"""Offline checks of authored specification fixtures; does not execute product code."""
from pathlib import Path
import argparse, collections, datetime, hashlib, json, math, re, sys, unicodedata
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser()
ap.add_argument('--report',required=True)
args=ap.parse_args()
out=(ROOT/args.report).resolve()
if not out.is_relative_to(ROOT.resolve()) or out.exists():raise SystemExit('Use a fresh report path inside the specification package')
checks=[];errors=[]
def read(rel):return json.loads((ROOT/rel).read_text(encoding='utf-8-sig'))
def assert_case(name,actual,expected):
    passed=actual==expected
    checks.append({'name':name,'result':'PASS' if passed else 'FAIL','actual':actual,'expected':expected})
    if not passed:errors.append(name)
def close(name,actual,expected,tolerance=1e-12):
    passed=abs(actual-expected)<=tolerance
    checks.append({'name':name,'result':'PASS' if passed else 'FAIL','actual':actual,'expected':expected,'absolute_tolerance':tolerance})
    if not passed:errors.append(name)
def terms(text):return re.findall('[a-z0-9]+',unicodedata.normalize('NFKC',text).casefold())
oracle=read('fixtures/applied/metric-oracle-v1.json')
intent_total=schema_total=0
for row in oracle['records']:
    lines=[line.strip() for line in row['raw_text'].splitlines() if line.strip()]
    intent=lines[0][7:] if lines and lines[0].startswith('INTENT=') and lines[0][7:] in oracle['allowed_intents'] else None
    exact=int(bool(lines and lines[0]=='INTENT='+row['expected_intent']))
    schema=int(len(lines)==2 and intent is not None and lines[1].startswith('REPLY=') and 1<=len(lines[1][6:])<=240)
    actual={'parsed_intent':intent,'exact_intent_match':exact,'response_schema_valid':schema}
    assert_case('metric parser '+row['record_id'],actual,row['expected'])
    intent_total+=exact;schema_total+=schema
for key,total in [('exact_intent_match',intent_total),('response_schema_valid',schema_total)]:
    assert_case('metric aggregate '+key,{'numerator':total,'denominator':len(oracle['records']),'ratio':total/len(oracle['records'])},oracle['expected_overall'][key])
for slice_id,expected in oracle.get('expected_per_slice',{}).items():
    rows=[r for r in oracle['records'] if r['slice']==slice_id]
    for key in ('exact_intent_match','response_schema_valid'):
        num=sum(r['expected'][key] for r in rows)
        assert_case('metric per-slice '+slice_id+' '+key,{'numerator':num,'denominator':len(rows),'ratio':num/len(rows)},expected[key])
corpus=read('fixtures/applied/retrieval-corpus-v1.json')['documents']
queries=read('fixtures/applied/retrieval-query-oracle-v1.json')['queries']
counts={d['record_id']:collections.Counter(terms(d['text'])) for d in corpus}
df=collections.Counter(t for c in counts.values() for t in c)
idf={t:math.log((1+len(counts))/(1+n))+1 for t,n in df.items()}
def vector(c):
    total=sum(c.values())
    v={t:n/total*idf[t] for t,n in c.items() if t in idf}
    norm=math.sqrt(math.fsum(x*x for t,x in sorted(v.items())))
    return {t:x/norm for t,x in sorted(v.items())} if norm else {}
vectors={k:vector(v) for k,v in counts.items()}
ranking=[]
for q in queries:
    query=q.get('query',q.get('text'))
    v=vector(collections.Counter(terms(query)))
    scores={ident:math.fsum(value*v.get(t,0.0) for t,value in sorted(d.items())) for ident,d in vectors.items()}
    ranked=sorted(scores,key=lambda k:(-scores[k],k))[:q.get('top_k',3)]
    expected=[hit['record_id'] for hit in q['expected_hits']]
    assert_case('retrieval '+q['query_id'],ranked,expected)
    for hit in q['expected_hits']:close('retrieval score '+q['query_id']+' '+hit['record_id'],scores[hit['record_id']],hit['score'],1e-15)
    ranking.append({'query_id':q['query_id'],'ranked':[{'record_id':k,'score':scores[k]} for k in ranked]})
assert_case('retrieval top-3 scores all above zero',all(h['score']>0 for q in queries for h in q['expected_hits']),True)
assert_case('retrieval queries whose supporting document is not ranked first',sum(q['expected_hits'][0]['record_id'] not in q['supported_record_ids'] for q in queries)>=2,True)
assert_case('LoRA unique trainable scalar count',30*((8*576+576*8)+(8*576+192*8)),460800)
assert_case('Tiny standard parameter count',2*257*64+64*64+2*(12*64*64+13*64)+2*64,137088)
assert_case('Tiny tied delta',257*64,16448)
close('mean of three illustrative results',sum([1.2,1.4,1.3])/3,1.3)
close('sample standard deviation',math.sqrt(sum((x-1.3)**2 for x in [1.2,1.4,1.3])/2),0.1)
close('byte aggregate is weighted',(6+4)/(3+8),10/11)
materialized=ROOT/'fixtures/data/materialized'
fixture_manifest=json.loads((materialized/'materialized-manifest.json').read_text())
for entry in fixture_manifest['files']:
    path=materialized/entry['path']
    assert_case('materialized digest '+entry['path'],hashlib.sha256(path.read_bytes()).hexdigest(),entry['sha256'])
    if path.suffix=='.jsonl' and entry['path'].startswith(('applied-intents-v1/','capstone-support-v1/')):
        records=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        for record in records:
            response=record['expected']['response']
            lines=response.splitlines()
            valid=len(lines)==2 and lines[0]=='INTENT='+record['slice'] and lines[1].startswith('REPLY=') and 1<=len(lines[1][6:])<=240
            if not valid:errors.append('SFT target format '+record['record_id'])
        assert_case('all SFT target formats '+entry['path'],all(len(x['expected']['response'].splitlines())==2 and x['expected']['response'].startswith('INTENT='+x['slice']+'\nREPLY=') for x in records),True)
for fixture_id in ('applied-intents-v1','capstone-support-v1'):
    def _words(text):
        text=unicodedata.normalize('NFKC',text).casefold()
        text=''.join(' ' if unicodedata.category(ch)[0] in 'PS' else ch for ch in text)
        return {w for w in text.split() if not any(c.isdigit() for c in w)}
    rows=[]
    for entry in fixture_manifest['files']:
        if entry['path'].startswith(fixture_id+'/') and entry['path'].endswith('.jsonl'):
            split=entry['path'].split('/')[1][:-6]
            rows+=[(split,json.loads(l)['messages'][-1]['content']) for l in (materialized/entry['path']).read_text(encoding='utf-8').splitlines()]
    assert_case('distinct user messages '+fixture_id,len({u for _,u in rows}),len(rows))
    worst=max(len(_words(a)&_words(b))/len(_words(a)|_words(b)) for i,(sa,a) in enumerate(rows) for sb,b in rows[i+1:] if sa!=sb)
    assert_case('cross-split Jaccard at most 0.5 '+fixture_id,worst<=0.5,True)
    records=[json.loads(l) for entry in fixture_manifest['files'] if entry['path'].startswith(fixture_id+'/') and entry['path'].endswith('.jsonl') for l in (materialized/entry['path']).read_text(encoding='utf-8').splitlines()]
    groups=collections.defaultdict(list)
    for r in records: groups[r['scenario_group_id']].append(_words(r['messages'][-1]['content']))
    assert_case('scenario groups '+fixture_id,(len(groups),sum(len(v)>=2 for v in groups.values())),{'applied-intents-v1':(42,42),'capstone-support-v1':(24,18)}[fixture_id])
    assert_case('paraphrases in a group differ in at least 3 words '+fixture_id,all(len(a^b)>=3 for v in groups.values() for i,a in enumerate(v) for b in v[i+1:]),True)
meta=read('evidence/model-config-and-license.json')
config=next(i['parsed'] for i in meta['items'] if i['name']=='config.json')
tok=next(i['parsed'] for i in meta['items'] if i['name']=='tokenizer_config.json')
profile=read('fixtures/applied/smollm2-135m-instruct-v1.json')
assert_case('pinned template digest',hashlib.sha256(tok['chat_template'].encode()).hexdigest(),profile['chat_template_sha256'])
assert_case('model architecture class',config['architectures'][0],profile['architecture']['class'])
for key,pkey in [('hidden_size','hidden_size'),('num_hidden_layers','layers'),('num_attention_heads','attention_heads'),('num_key_value_heads','key_value_heads'),('max_position_embeddings','native_context'),('vocab_size','vocabulary_size'),('intermediate_size','intermediate_size'),('rope_theta','rope_theta'),('tie_word_embeddings','tie_word_embeddings')]:
    assert_case('architecture '+key,config[key],profile['architecture'][pkey])
manifest=read('fixtures/applied/model-download-manifest.json')
assert_case('download total',sum(f['bytes'] for f in manifest['files']),manifest['total_bytes'])
for item in meta['items']:
    match=next((f for f in manifest['files'] if f['name']==item['name']),None)
    if match:
        assert_case('metadata size '+item['name'],item['bytes'],match['bytes'])
        assert_case('metadata digest '+item['name'],item['sha256'],match['sha256'])
def luminance(color):
    c=[int(color[i:i+2],16)/255 for i in (1,3,5)]
    v=[x/12.92 if x<=0.04045 else ((x+0.055)/1.055)**2.4 for x in c]
    return sum(x*w for x,w in zip(v,[0.2126,0.7152,0.0722]))
def contrast(a,b):
    x,y=sorted([luminance(a),luminance(b)])
    return (y+.05)/(x+.05)
ux=(ROOT/'03-LEARNER-EXPERIENCE.md').read_text(encoding='utf-8')
token_text=re.search(r'Light tokens are (.*?)\. Dark tokens are (.*?)\. ',ux)
button_text=re.search(r'light primary action uses <code>(#[0-9a-f]{6})</code> text on action; a dark primary action uses <code>(#[0-9a-f]{6})</code> text on action',ux)
themes={theme:dict(re.findall(r'<code>([a-z]+) (#[0-9a-f]{6})</code>',token_text.group(i))) for i,theme in ((1,'light'),(2,'dark'))}
buttons={'light':button_text.group(1),'dark':button_text.group(2)}
TOKENS=('background','surface','navigation','text','muted','border','action','focus','recorded','simulation','error')
palette=[]
for theme,tok in themes.items():
    assert_case(theme+' token list parsed from UX-009',sorted(tok),sorted(TOKENS))
    for name in ('text','muted','action','focus','recorded','simulation','error'):
        for bgname in ('background','surface','navigation'):
            ratio=contrast(tok[name],tok[bgname]);threshold=3 if name=='focus' else 4.5
            assert_case(f'{theme} {name} on {bgname} contrast threshold',ratio>=threshold,True)
            palette.append({'theme':theme,'foreground':name,'color':tok[name],'background':bgname,'background_color':tok[bgname],'ratio':ratio,'threshold':threshold})
    ratio=contrast(buttons[theme],tok['action'])
    assert_case(f'{theme} primary action label contrast threshold',ratio>=4.5,True)
    palette.append({'theme':theme,'foreground':'primary_action_label','color':buttons[theme],'background':'action','background_color':tok['action'],'ratio':ratio,'threshold':4.5})
dec=read('fixtures/applied/decode-oracle-v1.json')
for row in dec['selection_cases']:
    if row['temperature']==0:
        best=max(row['logits'])
        assert_case('decode '+row['case_id'],min(i for i,x in enumerate(row['logits']) if x==best),row['greedy_token'])
    else:
        sc=[x/row['temperature'] for x in row['logits']];mx=max(sc);ex=[math.exp(x-mx) for x in sc];z=math.fsum(ex);pr=[e/z for e in ex]
        order=sorted(range(len(pr)),key=lambda i:(-pr[i],i));kept=[]
        for i in order:
            kept.append(i)
            if math.fsum(pr[j] for j in kept)>=row['top_p']:break
        assert_case('decode '+row['case_id'],{'sorted':order,'retained':kept},{'sorted':row['sorted_token_ids'],'retained':row['retained_token_ids']})
for row in dec['stop_cases']:
    ids=row['generated_token_ids'];eos=row['eos_id']
    assert_case('decode stop '+row['case_id'],{'stop_reason':'eos' if ids and ids[-1]==eos else 'length','generated_token_count':len(ids),'visible_token_ids':[i for i in ids if i!=eos]},row['expected'])
for name in ('chat-no-truncation-v1','chat-drop-oldest-v1','chat-drop-oldest-two-pairs-v1'):
    fx=read('fixtures/applied/'+name+'.json');msgs=fx['request']['messages'];roles=[m['role'] for m in msgs if m['role']!='system']
    shape=(all(0<len(m['content'].encode())<=8192 for m in msgs) and sum(len(m['content'].encode()) for m in msgs)<=32768
           and [m['role'] for m in msgs].count('system')<=1 and roles[0]=='user' and roles[-1]=='user' and all(a!=b for a,b in zip(roles,roles[1:])))
    assert_case('chat fixture shape '+name,shape,True)
report={'kind':'specification_oracle_check','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'product_tests_executed':False,'result':'FAIL' if errors else 'PASS','case_count':len(checks),'errors':errors,'checks':checks,'retrieval_rankings':ranking,'palette_calculations':palette,'limitations':['No browser rendering or accessibility behavior tested.','No tokenizer/model/dependency runtime installed or executed.','These are independent calculations over authored specification fixtures, not product tests.']}
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:report[k] for k in ['result','case_count','errors']}))
sys.exit(bool(errors))
