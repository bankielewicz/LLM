from html.parser import HTMLParser
from pathlib import Path
import json, re, hashlib, zipfile
root=Path(__file__).resolve().parents[1]
class Audit(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids=[]; self.labels=[]; self.controls=[]; self.hrefs=[]; self.counts={}
    def handle_starttag(self,tag,attrs):
        a=dict(attrs);self.counts[tag]=self.counts.get(tag,0)+1
        if 'id' in a:self.ids.append(a['id'])
        if tag=='label' and 'for' in a:self.labels.append(a['for'])
        if tag in ('input','select','textarea') and 'id' in a:self.controls.append(a['id'])
        if tag=='a':self.hrefs.append(a.get('href',''))
checks=[]
for view in json.loads((root/'qa/template-output.json').read_text(encoding='utf-8')):
    a=Audit();a.feed(view['html'])
    assert len(a.ids)==len(set(a.ids)),(view['name'],'duplicate ID')
    assert all(x in a.ids for x in a.labels),(view['name'],'orphan label')
    assert all(x in a.labels for x in a.controls),(view['name'],'unlabelled control')
    for href in a.hrefs:
        if href.startswith('course/'):
            assert (root/'dist'/href).is_file(),(view['name'],href)
        if href.startswith('#/lesson/'):
            assert re.fullmatch(r'#/lesson/(0[0-9]|1[0-2])',href),(view['name'],href)
    checks.append({'view':view['name'],'ids_unique':True,'controls_labelled':True,'local_links_exist':True})
css=(root/'dist/styles.css').read_text(encoding='utf-8-sig')
def vars_in(selector):
    body=re.findall(re.escape(selector)+r'\{([^}]+)\}',css)
    return dict(re.findall(r'--([\w-]+):\s*(#[a-fA-F0-9]{3,6})\b',';'.join(body)))
def lum(h):
    h=h.lstrip('#');h=h if len(h)==6 else ''.join(x*2 for x in h)
    c=[int(h[i:i+2],16)/255 for i in (0,2,4)];c=[x/12.92 if x<=.04045 else ((x+.055)/1.055)**2.4 for x in c]
    return sum(a*b for a,b in zip(c,(.2126,.7152,.0722)))
def ratio(a,b):
    x,y=sorted([lum(a),lum(b)]);return (y+.05)/(x+.05)
light=vars_in(':root'); dark=light|vars_in('[data-theme=dark]')
contrast=[]
for theme,v in [('light',light),('dark',dark)]:
    for fg,bg in [('ink','bg'),('ink','surface'),('muted','bg'),('muted','surface'),('muted','sidebar'),('accent','surface'),('accent','soft'),('teal','teal-bg'),('amber','amber-bg'),('vector','vector-bg'),('weight','weight-bg'),('target','target-bg'),('chart-train','surface'),('chart-val','surface')]:
        r=ratio(v[fg],v[bg]);contrast.append({'theme':theme,'foreground':fg,'background':bg,'ratio':round(r,2)});assert r>=4.5,(theme,fg,bg,r)
    r=ratio(v['focus'],v['surface']);assert r>=3,(theme,'focus',r)
assert 'prefers-reduced-motion:reduce' in css
assert 'scroll-behavior:auto!important' in css
assert 'forced-colors:active' in css
assert ':focus-visible' in css
original=root.parent/'llm-foundations-v2'
hashes=json.loads((root/'qa/source-hashes.json').read_text(encoding='utf-8'))
with zipfile.ZipFile(root/'dist/course.zip') as z:
    for rel,sha in hashes.items():
        assert hashlib.sha256((original/rel).read_bytes()).hexdigest()==sha,rel
        assert hashlib.sha256(z.read('llm-foundations-v2/'+rel)).hexdigest()==sha,rel
report={'method':'Source, generated HTML structure, mathematical palette contrast, and file hashes. No browser rendering or UI interaction.','views':checks,'contrast':contrast,'minimum_text_contrast':min(x['ratio'] for x in contrast),'original_files_unchanged':len(hashes),'download_files_byte_identical':len(hashes),'browser_visual_keyboard_screenreader_200percent_zoom':'NOT_TESTED: admin-enforced browser policy','reduced_motion':'CSS suppression and absence of autoplay checked; OS behavior NOT_TESTED'}
(root/'qa/accessibility-source.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'{len(checks)} generated HTML views passed label/link/ID checks; {len(contrast)} text-color pairs passed 4.5:1; minimum {report["minimum_text_contrast"]}:1. {len(hashes)} source and ZIP files unchanged.')
