"""Refresh presentation data from the untouched, canonical course package.
Only standard library dependencies. Run from any working directory.
"""
from pathlib import Path
import hashlib, json, shutil, zipfile, sys
ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT.parent / 'llm-foundations-v2'
OUT = ROOT / 'dist'
COPY = ROOT / 'course'
FILES = [p for p in SOURCE.rglob('*') if p.is_file() and not any(x in p.relative_to(SOURCE).parts for x in ('runs', 'data', '.venv', '__pycache__', '.git'))]
for p in FILES:
    rel = p.relative_to(SOURCE)
    for base in (COPY, OUT / 'course'):
        dest = base / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
manifest = {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in FILES}
receipt = ROOT / 'qa' / 'source-hashes.json'
if receipt.exists() and json.loads(receipt.read_text(encoding='utf-8')) != manifest:
    raise SystemExit('Course source changed since the initial receipt. Review before updating this receipt.')
receipt.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
content = json.loads((SOURCE / 'curriculum.json').read_text(encoding='utf-8'))
content['lessons'] = {m['id']: (SOURCE / m['path']).read_text(encoding='utf-8') for m in content['modules']}
content['references'] = {str(p.relative_to(SOURCE)): p.read_text(encoding='utf-8') for p in FILES if p.suffix == '.md' and 'lessons' not in p.parts}
content['cases'] = json.loads((SOURCE / 'examples/tokenizer-cases.json').read_text(encoding='utf-8'))
content['runs'] = {}
for name in ('byte-training', 'bpe-check'):
    folder = SOURCE / 'examples' / name
    content['runs'][name] = {'name': name, 'manifest': json.loads((folder/'manifest.json').read_text(encoding='utf-8')), 'metrics': [json.loads(x) for x in (folder/'metrics.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()], 'result': json.loads((folder/'result.json').read_text(encoding='utf-8')), 'sample': (folder/'sample.txt').read_text(encoding='utf-8') if (folder/'sample.txt').exists() else None}
(OUT / 'content.json').write_text(json.dumps(content, ensure_ascii=False), encoding='utf-8')
with zipfile.ZipFile(OUT / 'course.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in FILES:
        z.write(p, 'llm-foundations-v2/' + p.relative_to(SOURCE).as_posix())
print(f'Compiled {len(content["modules"])} lessons, {len(content["runs"])} recorded runs; preserved {len(FILES)} source files.')
