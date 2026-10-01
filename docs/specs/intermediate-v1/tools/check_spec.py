"""Validate this specification package; never executes product cases or model code."""
from pathlib import Path
import argparse, datetime, hashlib, json, re, sys
from urllib.parse import unquote, urldefrag
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {'static', 'win-cpu', 'win-cuda', 'wsl-cpu', 'wsl-cuda'}
REPORT = {'kind':'specification_document_check','product_tests_executed':False,
          'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'checks':[], 'errors':[]}
def fail(message):
    REPORT['errors'].append(str(message))
def check(name, detail):
    REPORT['checks'].append({'name':name,'result':'PASS','detail':detail})
def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Non-JSON constant '+x)))
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def pointer(value, fragment):
    if not fragment: return value
    if not fragment.startswith('/'): raise ValueError('Unsupported non-pointer fragment '+fragment)
    for part in fragment[1:].split('/'):
        part=unquote(part).replace('~1','/').replace('~0','~')
        value=value[int(part)] if isinstance(value,list) else value[part]
    return value
def walk(value):
    if isinstance(value,dict):
        yield value
        for child in value.values(): yield from walk(child)
    elif isinstance(value,list):
        for child in value: yield from walk(child)

ap=argparse.ArgumentParser()
ap.add_argument('--report',required=True,help='Fresh report path relative to specification root')
mode=ap.add_mutually_exclusive_group()
mode.add_argument('--write-index',action='store_true',help='Authoring only: rewrite requirements.json and acceptance-cases.json; requires --revision')
mode.add_argument('--check-index',action='store_true',help='Compare requirements.json and acceptance-cases.json with the generated index without writing')
ap.add_argument('--revision',help='Specification revision written by --write-index, for example 1.1')
args=ap.parse_args()
if args.write_index and not args.revision: raise SystemExit('--write-index requires --revision')
report_path=(ROOT/args.report).resolve()
if not report_path.is_relative_to(ROOT.resolve()) or report_path.exists():
    raise SystemExit('Report must be a fresh path within this package.')
documents={}
for path in ROOT.rglob('*.json'):
    if path.relative_to(ROOT).parts[0]=='reviews': continue
    try: documents[path.resolve()]=load(path)
    except Exception as e: fail(f'JSON {path.relative_to(ROOT)}: {e}')
check('json_parse',f'{len(documents)} JSON documents parsed; failures listed separately')

requirements={}
for path in sorted(ROOT.rglob('*.md')):
    if path.relative_to(ROOT).parts[0]=='reviews': continue
    text=path.read_text(encoding='utf-8-sig')
    for ident,title in re.findall(r'^## ([A-Z]+-\d{3}) — (.+)$',text,re.M):
        if ident in requirements: fail(f'Duplicate requirement heading {ident}')
        requirements[ident]={'id':ident,'title':title,'document':path.name}
    if re.search(r'\b(TODO|TBD|TBC|FIXME)\b',text):
        fail(f'Unresolved placeholder marker in {path.name}')
    for target in re.findall(r'\[[^\]\n]+\]\(([^)\n]+)\)',text):
        target=target.strip('<>')
        if target.startswith(('https://','http://','mailto:','#')): continue
        base,fragment=urldefrag(unquote(target))
        if not base: continue
        if not (path.parent/base).exists() and not (args.write_index and base in {'requirements.json','acceptance-cases.json'}): fail(f'Broken local link {path.name}: {target}')
check('requirements_and_links',f'{len(requirements)} normative headings indexed')

traces=[]
cases=[]
seen=set()
trace_requirements=set()
for path in sorted((ROOT/'trace').glob('*.json')):
    data=documents.get(path.resolve())
    if not isinstance(data,dict): fail(f'Invalid trace {path.name}'); continue
    section=data.get('section')
    if not (ROOT/str(section)).is_file(): fail(f'Unknown trace section {path.name}:{section}')
    for item in data.get('requirements',[]):
        ident=item.get('id')
        if ident in trace_requirements: fail(f'Duplicate trace requirement {ident}')
        trace_requirements.add(ident)
        if ident not in requirements: fail(f'Trace requirement absent from prose {ident}')
        elif requirements[ident]['document']!=section or requirements[ident]['title']!=item.get('title'):
            fail(f'Trace heading mismatch {ident}')
    for case in data.get('cases',[]):
        ident=case.get('id')
        if not re.fullmatch(r'AC-[A-Z]+-\d{3}',str(ident)): fail(f'Invalid case ID {ident}')
        if ident in seen: fail(f'Duplicate case ID {ident}')
        seen.add(ident)
        for field in ['name','method','requirements','profiles','preconditions','steps','expected','status']:
            if not case.get(field): fail(f'{ident} missing {field}')
        if case.get('status')!='NOT_RUN': fail(f'{ident} asserts product status {case.get("status")}')
        if case.get('method') not in ['unit','integration','browser','manual','source']:
            fail(f'{ident} invalid method')
        for req in case.get('requirements',[]):
            if req not in requirements: fail(f'{ident} refers to missing {req}')
        if not set(case.get('profiles',[]))<=PROFILES: fail(f'{ident} invalid profiles')
        if len(set(case.get('profiles',[])))!=len(case.get('profiles',[])): fail(f'{ident} duplicate profile')
        case=dict(case,section=section)
        cases.append(case)
    traces.append(path.name)
covered={r for c in cases for r in c.get('requirements',[])}
for ident in requirements:
    if ident not in trace_requirements: fail(f'Missing trace requirement {ident}')
    if ident not in covered: fail(f'Missing acceptance coverage {ident}')
units=[{'execution_id':f'{c["id"]}@{p}'+(f'@{b}' if b else ''),
        'case_id':c['id'],'profile':p,'browser':b,'status':'NOT_RUN'}
       for c in cases for p in c.get('profiles',[])
       for b in (['Edge','Firefox'] if c.get('method')=='browser' else [None])]
check('traceability',f'{len(requirements)} requirements; {len(cases)} cases; {len(units)} frozen case/profile/browser execution units')

by_id={}
schema_docs=[]
for path,data in documents.items():
    if 'contracts' in path.parts and isinstance(data,dict):
        if data.get('$id'):
            if data['$id'] in by_id: fail('Duplicate schema id '+data['$id'])
            by_id[data['$id']]=(path,data)
        if data.get('$schema') == 'https://json-schema.org/draft/2020-12/schema':
            schema_docs.append((path,data))
            try: Draft202012Validator.check_schema(data)
            except Exception as e: fail(f'Invalid JSON Schema {path.name}: {str(e)[:400]}')
for path,data in documents.items():
    if 'contracts' not in path.parts: continue
    for node in walk(data):
        if '$ref' not in node: continue
        ref=node['$ref']; base,frag=urldefrag(ref)
        try:
            if not base: target=data
            elif base in by_id: target=by_id[base][1]
            elif re.match(r'^[a-z]+:',base): raise ValueError('Unknown external schema '+base)
            else: target=documents[(path.parent/base).resolve()]
            pointer(target,frag)
        except Exception as e: fail(f'Unresolved reference {path.relative_to(ROOT)} {ref}: {str(e)[:200]}')
check('schemas_and_refs',f'{len(schema_docs)} draft-2020-12 schemas checked; all contract refs traversed')

registry=Registry()
for path,data in schema_docs:
    resource=Resource.from_contents(data)
    registry=registry.with_resource(path.as_uri(),resource)
    if data.get('$id'): registry=registry.with_resource(data['$id'],resource)
fixture_count=0
case_path=ROOT/'fixtures/data/validation-cases.json'
if case_path.exists():
    payload=documents.get(case_path.resolve())
    fixture_cases=payload if isinstance(payload,list) else payload.get('cases',[])
    for item in fixture_cases:
        if item.get('method')=='semantic': continue
        fixture_count+=1
        try:
            # Schema and instance paths are relative to the validation-cases manifest.
            sp=(case_path.parent/item['schema']).resolve()
            ip=(case_path.parent/item['instance']).resolve()
            schema=documents[sp]; instance=documents[ip]
            validator=Draft202012Validator(schema,registry=registry,format_checker=FormatChecker())
            errors=list(validator.iter_errors(instance))
            actual=not errors
            if actual!=item['valid']:
                fail(f'Fixture {item["id"]}: expected valid={item["valid"]}, actual={actual}; '+(errors[0].message[:180] if errors else ''))
        except Exception as e: fail(f'Fixture {item.get("id")}: {str(e)[:250]}')
else: fail('Missing fixtures/data/validation-cases.json')
profile_path=ROOT/'fixtures/applied/smollm2-135m-instruct-v1.json'
profile_schema=ROOT/'contracts/model-profile.json'
if profile_path.resolve() in documents and profile_schema.resolve() in documents:
    try:
        profile=documents[profile_path.resolve()]
        errors=list(Draft202012Validator(documents[profile_schema.resolve()],registry=registry,format_checker=FormatChecker()).iter_errors(profile))
        if errors: fail('Pinned model profile instance: '+errors[0].message[:300])
        manifest=ROOT/'fixtures/applied/model-download-manifest.json'
        if profile.get('download_manifest_sha256')!=digest(manifest): fail('Pinned profile download_manifest_sha256 does not hash exact manifest bytes')
        fixture_count+=1
    except Exception as e: fail('Pinned profile instance validation: '+str(e)[:250])
check('schema_fixtures',f'{fixture_count} positive/negative schema fixture cases evaluated')
REPORT.setdefault('not_document_checkable',[])
semantic_count=0
materialized=ROOT/'fixtures/data/materialized'
try:
    sem_cases=documents[case_path.resolve()].get('semantic_cases',[])
    fixture_manifest=documents[(materialized/'materialized-manifest.json').resolve()]
    manifest_digest=digest(materialized/'materialized-manifest.json')
    def jsonl_count(rel): return sum(1 for line in (materialized/rel).read_text(encoding='utf-8').splitlines() if line.strip())
    for item in sem_cases:
        oracle=item['oracle']; semantic_count+=1
        if item['id']=='DATA-SEM-001':
            audit=fixture_manifest['audits']['data-clinic-leaky-v1']
            leaks=audit['exact_duplicate_pairs']+audit['normalized_near_duplicate_pairs']+audit['group_overlap_count']
            if audit!={k:v for k,v in oracle.items() if k!='eligibility'} or (oracle['eligibility']=='audit_only')!=(leaks>0):
                fail(f'{item["id"]}: leaky audit {audit} does not match oracle {oracle}')
        elif item['id']=='DATA-SEM-002':
            counts={s:jsonl_count(f'capstone-support-v1/{s}.jsonl') for s in ('train','validation','sealed_test')}
            if counts!={s:oracle[s] for s in ('train','validation','sealed_test')}: fail(f'{item["id"]}: capstone split counts {counts}')
            REPORT['not_document_checkable'].append(item['id']+'.one_paired_evaluation: product rule, NOT_RUN')
        elif item['id']=='DATA-SEM-003':
            if (manifest_digest!=oracle['manifest_sha256'] or len(fixture_manifest['files'])!=oracle['files']
                    or fixture_manifest['audits'].get('data-clinic-v1')!=oracle['clean_audit']
                    or fixture_manifest['audits'].get('data-clinic-leaky-v1')!=oracle['leaky_audit']):
                fail(f'{item["id"]}: materialized manifest identity/audits differ from oracle')
        else: fail(f'Unknown semantic fixture case {item["id"]}')
except Exception as e: fail(f'Semantic fixture cases: {str(e)[:250]}')
check('semantic_fixture_cases',f'{semantic_count} semantic/source fixture cases evaluated; product-only clauses listed as not_document_checkable')

openapi=documents.get((ROOT/'contracts/openapi.json').resolve())
if not openapi: fail('Missing OpenAPI document')
else:
    if not str(openapi.get('openapi','')).startswith('3.1.'): fail('OpenAPI must be3.1')
    op_ids=set();op_count=0
    for route,item in openapi.get('paths',{}).items():
        pathparams=set(re.findall(r'\{([^}]+)\}',route))
        for method,op in item.items():
            if method not in ['get','post','put','patch','delete','head','options']:continue
            op_count+=1;oid=op.get('operationId')
            if not oid or oid in op_ids:fail(f'Missing/duplicate operationId {route} {method}')
            op_ids.add(oid)
            if not op.get('responses'):fail(f'Missing responses {route} {method}')
            params=item.get('parameters',[])+op.get('parameters',[])
            declared={p.get('name') for p in params if p.get('in')=='path' and p.get('required') is True}
            if not pathparams<=declared:fail(f'Undeclared path parameter {route} {method}')
    check('openapi_structure',f'{len(openapi.get("paths",{}))} paths and {op_count} operations structurally checked; not full OpenAPI runtime validation')
vocab=documents.get((ROOT/'contracts/error-vocabulary.json').resolve())
if not vocab: fail('Missing contracts/error-vocabulary.json')
elif openapi:
    comps=openapi['components']['schemas']; rows=vocab['codes']
    kinds={k:{r['code'] for r in rows if k in r['kinds']} for k in ('top_level','reason','job','terminal')}
    enums={'top_level':set(comps['Error']['properties']['error']['properties']['code']['enum']),
           'reason':set(comps.get('ReasonCode',{}).get('enum',[])),'job':set(comps.get('JobErrorCode',{}).get('enum',[])),
           'terminal':set(comps.get('TerminalReason',{}).get('enum',[]))}
    for k in kinds:
        if kinds[k]!=enums[k]: fail(f'OpenAPI {k} enum differs from error-vocabulary.json: {sorted(kinds[k]^enums[k])[:10]}')
    for r in rows:
        if 'reason' in r['kinds'] and (r.get('top_level_code') not in kinds['top_level'] or not r.get('http_status')):
            fail(f'Reason code {r["code"]} lacks a top-level code or HTTP status')
    known={r['code'] for r in rows}|set(vocab.get('ignored_tokens',{}))
    known|={r['id'] for r in documents.get((ROOT/'contracts/semantic-rules.json').resolve(),{}).get('rules',[])}
    token_re=re.compile(r'(?<![A-Za-z0-9_\-/.])([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)(?![A-Za-z0-9_])')
    unknown={}
    for path in sorted(list(ROOT.glob('0*.md'))+list((ROOT/'contracts').glob('*.md'))):
        for n,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
            for tok in token_re.findall(line):
                if tok not in known: unknown.setdefault(tok,f'{path.name}:{n}')
    for case in cases:
        for text in case.get('steps',[])+case.get('expected',[])+case.get('preconditions',[]):
            for tok in token_re.findall(text):
                if tok not in known: unknown.setdefault(tok,case['id'])
    for tok,where in sorted(unknown.items()): fail(f'Uppercase code {tok} at {where} is not in error-vocabulary.json, semantic-rules.json or ignored_tokens')
    check('error_vocabulary',f'{len(rows)} vocabulary codes equal the OpenAPI enums; prose and case census clean')

baseline=documents.get((ROOT/'evidence/source-baseline.json').resolve())
if baseline:
    repo=ROOT.parents[2]
    for rel,expected in baseline['files_sha256'].items():
        path=repo/rel
        if not path.is_file() or digest(path)!=expected: fail('Protected baseline changed: '+rel)
    check('baseline_preservation',f'{len(baseline["files_sha256"])} pre-existing tracked files checked byte-for-byte')
else:fail('Missing baseline inventory')

recorded=documents.get((ROOT/'requirements.json').resolve(),{})
revision=args.revision if args.write_index else recorded.get('spec_revision')
if not re.fullmatch(r'\d+\.\d+',str(revision)): fail(f'Invalid or missing specification revision {revision!r}')
index={'requirements.json':{'spec_revision':revision,'requirements':list(requirements.values())},
       'acceptance-cases.json':{'spec_revision':revision,'product_status':'NOT_RUN','cases':cases,'execution_units':units}}
if args.check_index:
    for name,value in index.items():
        if (ROOT/name).read_bytes()!=(json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode():
            fail(f'{name} differs from the index generated from trace/*.json and prose headings')
    check('index_consistency','requirements.json and acceptance-cases.json equal the generated index byte-for-byte')
if args.write_index and not REPORT['errors']:
    for name,value in index.items():
        (ROOT/name).write_bytes((json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode())
rules_doc=documents.get((ROOT/'contracts/semantic-rules.json').resolve(),{})
rule_ids=[r.get('id') for r in rules_doc.get('rules',[])]
if len(rule_ids)!=len(set(rule_ids)): fail('Duplicate semantic rule IDs')
for r in rules_doc.get('rules',[]):
    if not r.get('applies_to') or not (r.get('predicate') or r.get('decision')) or not (r.get('rejection') or r.get('invariant')): fail(f'Semantic rule {r.get("id")} lacks applies_to, predicate/decision or rejection/invariant')
    rej=r.get('rejection') or {}
    if rej.get('reason_code') and vocab and rej['reason_code'] not in {v['code'] for v in vocab['codes']}: fail(f'Semantic rule {r["id"]} reason {rej["reason_code"]} not in error vocabulary')
annotated=set()
for path,data in documents.items():
    for node in walk(data):
        for rid in node.get('x-semantic-rules',[]) if isinstance(node.get('x-semantic-rules',[]),list) else []:
            annotated.add(rid)
            if rid not in rule_ids: fail(f'x-semantic-rules references unknown rule {rid} in {path.name}')
check('semantic_rules',f'{len(rule_ids)} semantic rules well-formed; {len(annotated)} referenced by schema annotations')
roles=documents.get((ROOT/'contracts/evidence-roles.json').resolve())
if not roles: fail('Missing contracts/evidence-roles.json')
elif openapi:
    enum_profiles=set(openapi['components']['schemas']['EvidenceVerifyRequest']['properties']['verification_profile_id']['enum'])
    if set(roles['profiles'])!=enum_profiles: fail(f'evidence-roles.json profiles differ from EvidenceVerifyRequest enum: {sorted(set(roles["profiles"])^enum_profiles)}')
    ids=[c['id'] for p in roles['profiles'].values() for c in p['check_ids']]
    if len(ids)!=len(set(ids)): fail('Duplicate evidence check IDs')
    check('evidence_roles',f'{len(roles["profiles"])} profiles and {len(ids)} frozen check IDs')
REPORT['result']='FAIL' if REPORT['errors'] else 'PASS'
REPORT['requirement_count']=len(requirements)
REPORT['case_count']=len(cases)
REPORT['execution_unit_count']=len(units)
REPORT['schema_count']=len(schema_docs)
REPORT['schema_fixture_count']=fixture_count
report_path.parent.mkdir(parents=True,exist_ok=True)
report_path.write_bytes((json.dumps(REPORT,indent=2,ensure_ascii=False)+'\n').encode())
print(json.dumps({k:REPORT[k] for k in ['result','requirement_count','case_count','execution_unit_count','schema_count','schema_fixture_count','errors']},ensure_ascii=False))
sys.exit(bool(REPORT['errors']))
