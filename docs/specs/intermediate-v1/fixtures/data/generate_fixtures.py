"""Materialize the authored intermediate-v1 fixtures. Standard library only.

Materialized files contain authored inputs and labels only. Computed floating-point values are never
materialized, so retained bytes do not depend on the interpreter's float summation or libm. TF-IDF scores are
checked against fixtures/applied/retrieval-query-oracle-v1.json within the APP-013 absolute tolerance."""
from __future__ import annotations
import argparse, hashlib, json, math, platform, re, unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CANONICAL = ROOT / "materialized"
SOURCE = json.loads((ROOT / "fixture-source.json").read_text(encoding="utf-8"))
SCORE_ORACLE = json.loads((ROOT.parent / "applied" / "retrieval-query-oracle-v1.json").read_text(encoding="utf-8"))
SCORE_TOLERANCE = 1e-15
OUTPUT_FILES = {}

def line(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"

def write_rows(relative, rows):
    raw = "".join(line(row) for row in rows).encode("utf-8")
    OUTPUT_FILES[relative.as_posix()] = raw
    return {"path": relative.as_posix(), "records": len(rows), "utf8_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}

def norm(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    kept = [token for token in text.split() if not re.fullmatch(r"[a-z]{1,3}[0-9]+", token)]
    return " ".join(re.sub(r"[0-9]+", " ", " ".join(kept)).split())

def content(row):
    """DAT-003 compared content: document text, or instruction message contents joined by LF."""
    return row["text"] if "text" in row else "\n".join(m["content"] for m in row["messages"])

def audit(splits):
    named = [(split, row) for split, rows in splits.items() for row in rows]
    exact = near = 0
    for i, (sa, a) in enumerate(named):
        for sb, b in named[i + 1:]:
            if sa == sb:
                continue
            if content(a).encode("utf-8") == content(b).encode("utf-8"):
                exact += 1
            elif norm(content(a)) == norm(content(b)):
                near += 1
    groups = {}
    for split, row in named:
        groups.setdefault(row["scenario_group_id"], set()).add(split)
    return {"records": len(named), "exact_duplicate_pairs": exact,
            "normalized_near_duplicate_pairs": near,
            "group_overlap_count": sum(len(v) > 1 for v in groups.values())}

def clinic_rows():
    variants = SOURCE["data_clinic"]["variants"]
    result = {"train": [], "validation": [], "test": []}
    for scenario in SOURCE["data_clinic"]["scenarios"]:
        number = int(scenario["group"][2:])
        split = "train" if number <= 6 else "validation" if number <= 9 else "test"
        for variant in variants:
            result[split].append({"record_id": f"{scenario['group'].lower()}-{variant['id']}",
                "scenario_group_id": scenario["group"],
                "text": variant["template"].format(**scenario)})
    return result

def instruction_rows(prefix, ordinals, capstone=False):
    """One record per authored paraphrase; ordinals run through the scenario groups in order."""
    result = {"train": [], "validation": [], "sealed_test" if capstone else "test": []}
    for spec in SOURCE["instruction_sets"]["slices"]:
        flat = [(number, user, group["reply"])
                for number, group in enumerate(spec["capstone_groups" if capstone else "applied_groups"], 1)
                for user in group["users"]]
        assert len(flat) == len(ordinals), (spec["slice"], len(flat))
        for ordinal in ordinals:
            group_number, user, reply = flat[ordinal - 1]
            if capstone:
                split = "train" if ordinal <= 5 else "validation" if ordinal == 6 else "sealed_test"
            else:
                split = "train" if ordinal <= 10 else "validation" if ordinal <= 12 else "test"
            full = f"INTENT={spec['slice']}\nREPLY={reply}"
            result[split].append({"record_id": f"{prefix}-{spec['slice']}-{ordinal:02d}",
                "scenario_group_id": f"{prefix.upper()}-{spec['slice']}-G{group_number:02d}",
                "slice": spec["slice"], "messages": [{"role": "user", "content": user}],
                "expected": {"intent": spec["slice"], "response": full}})
    return result

def tokens(text):
    return re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).casefold())

def retrieval_rows():
    docs = SOURCE["retrieval_manual"]["documents"]
    df = Counter(token for doc in docs for token in set(tokens(doc["text"])))
    idf = {token: math.log((1 + len(docs)) / (1 + count)) + 1 for token, count in df.items()}
    def vector(text):
        terms = tokens(text); counts = Counter(terms)
        values = {t: counts[t] / len(terms) * idf[t] for t in sorted(counts) if t in idf}
        length = math.sqrt(math.fsum(v * v for v in values.values()))
        return {t: v / length for t, v in values.items()} if length else {}
    doc_vectors = {doc["record_id"]: vector(doc["text"]) for doc in docs}
    oracle = {q["query_id"]: q for q in SCORE_ORACLE["queries"]}
    assert sorted(oracle) == sorted(q["query_id"] for q in SOURCE["retrieval_manual"]["queries"]), "oracle/source query IDs differ"
    queries = []
    for source in SOURCE["retrieval_manual"]["queries"]:
        query_vector = vector(source["text"])
        scores = {doc_id: math.fsum(query_vector[t] * values.get(t, 0.0) for t in sorted(query_vector))
                  for doc_id, values in doc_vectors.items()}
        ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
        assert ordered[:3] == source["top3"], (source["query_id"], ordered[:3], source["top3"])
        expected = oracle[source["query_id"]]
        assert expected["query"] == source["text"], ("oracle query text differs", source["query_id"])
        assert [hit["record_id"] for hit in expected["expected_hits"]] == source["top3"], ("oracle ranking differs", source["query_id"])
        for hit in expected["expected_hits"]:
            assert abs(scores[hit["record_id"]] - hit["score"]) <= SCORE_TOLERANCE, (source["query_id"], hit["record_id"], scores[hit["record_id"]], hit["score"])
        queries.append(dict(source))
    return docs, queries

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="compare regenerated bytes with retained fixtures (default)")
    group.add_argument("--output", type=Path, help="write to a new, nonexistent directory")
    args = parser.parse_args()
    OUTPUT_FILES.clear()
    manifest = {"format": "llm-foundations-materialized-fixtures-v1", "files": [], "audits": {}}
    clean = clinic_rows()
    assert audit(clean) == {"records": 48, "exact_duplicate_pairs": 0,
                            "normalized_near_duplicate_pairs": 0, "group_overlap_count": 0}
    leaky = json.loads(json.dumps(clean))
    by_id = {r["record_id"]: r for rows in leaky.values() for r in rows}
    clean_by_id = {r["record_id"]: r for rows in clean.values() for r in rows}
    by_id["dc07-summary"]["text"] = clean_by_id["dc01-summary"]["text"]
    by_id["dc08-timeline"]["text"] = clean_by_id["dc02-timeline"]["text"].upper() + " !!!"
    by_id["dc10-symptom"]["text"] = clean_by_id["dc03-symptom"]["text"].upper() + " !!!"
    by_id["dc10-resolution"]["scenario_group_id"] = "DC06"
    expected_leaky = {"records": 48, "exact_duplicate_pairs": 1,
                      "normalized_near_duplicate_pairs": 2, "group_overlap_count": 1}
    assert audit(leaky) == expected_leaky, audit(leaky)
    applied = instruction_rows("ai", range(1, 15))
    capstone = instruction_rows("cs", range(1, 9), True)
    for rows, count in ((applied, 84), (capstone, 48)):
        assert audit(rows) == {"records": count, "exact_duplicate_pairs": 0,
                               "normalized_near_duplicate_pairs": 0, "group_overlap_count": 0}, audit(rows)
        # self-test: each seeded leak must be detected exactly once
        held_out = "test" if "test" in rows else "sealed_test"
        first = rows["train"][0]["messages"][0]["content"]
        copied = json.loads(json.dumps(rows)); copied[held_out][0]["messages"] = [{"role": "user", "content": first}]
        assert audit(copied)["exact_duplicate_pairs"] == 1
        recased = json.loads(json.dumps(rows)); recased[held_out][0]["messages"] = [{"role": "user", "content": first.upper() + " !!!"}]
        assert (audit(recased)["exact_duplicate_pairs"], audit(recased)["normalized_near_duplicate_pairs"]) == (0, 1)
        moved = json.loads(json.dumps(rows)); moved[held_out].append(moved["train"].pop(0))
        assert audit(moved)["group_overlap_count"] == 1
    manifest["audits"] = {"data-clinic-v1": audit(clean), "data-clinic-leaky-v1": audit(leaky),
                          "applied-intents-v1": audit(applied), "capstone-support-v1": audit(capstone)}
    for fixture, splits in (("data-clinic-v1", clean), ("data-clinic-leaky-v1", leaky),
                            ("applied-intents-v1", applied), ("capstone-support-v1", capstone)):
        for split, rows in splits.items():
            manifest["files"].append(write_rows(Path(fixture) / f"{split}.jsonl", rows))
    docs, queries = retrieval_rows()
    manifest["files"].append(write_rows(Path("retrieval-manual-v1/documents.jsonl"), docs))
    manifest["files"].append(write_rows(Path("retrieval-manual-v1/queries.jsonl"), queries))
    manifest["files"].sort(key=lambda item: item["path"])
    raw = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    OUTPUT_FILES["materialized-manifest.json"] = raw
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        for relative, content in OUTPUT_FILES.items():
            target = args.output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        print(f"wrote {len(manifest['files'])} fixtures plus manifest to new directory {args.output}")
    else:
        retained = {p.relative_to(CANONICAL).as_posix(): p.read_bytes()
                    for p in CANONICAL.rglob("*") if p.is_file()}
        assert set(retained) == set(OUTPUT_FILES), (sorted(set(retained) - set(OUTPUT_FILES)), sorted(set(OUTPUT_FILES) - set(retained)))
        for relative, content in OUTPUT_FILES.items():
            assert retained[relative] == content, f"fixture differs: {relative}"
        digest = hashlib.sha256(retained["materialized-manifest.json"]).hexdigest()
        print(f"checked {len(manifest['files'])} fixtures; audits/rankings/oracle scores and retained bytes match; "
              f"manifest_sha256={digest}; python={platform.python_version()} ({platform.system()})")

if __name__ == "__main__":
    main()
