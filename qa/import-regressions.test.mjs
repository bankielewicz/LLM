import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';
import * as C from '../dist/core.js';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const course=JSON.parse(fs.readFileSync(path.join(root,'dist/content.json'),'utf8'));
const failedFixture=JSON.parse(fs.readFileSync(path.join(root,'qa/fixtures/failed-result.json'),'utf8'));
const fresh=()=>structuredClone(course.runs['byte-training']);
const files=run=>[
  {name:'manifest.json',text:JSON.stringify(run.manifest)},
  {name:'metrics.jsonl',text:run.metrics.map(x=>JSON.stringify(x)).join('\n')},
  {name:'result.json',text:JSON.stringify(run.result)}
];
const backup=run=>JSON.parse(JSON.stringify({...C.blankState(),runs:[run]}));
const inertMarkup='<b data-review-marker>unescaped parameter count</b>';
// A source fixture of the exact failure-artifact structure, not an induced training failure.
test('failed fixture matches the preserved producer and omits the requested limit',()=>{
  const producer=fs.readFileSync(path.join(root,'course/labs/lab.py'),'utf8');
  assert.match(producer,/json\.dumps\(\{"status": "failed", "step": step, "error": str\(error\)\}\)/);
  assert.deepEqual(Object.keys(failedFixture).sort(),['error','status','step']);
});
test('file import rejects parameter markup and alternate JSON types before returning a run',()=>{
  for(const parameters of [inertMarkup,[inertMarkup],{value:inertMarkup},'137088',null,true,-1,1.5,Number.MAX_SAFE_INTEGER+1]){
    const run=fresh();run.manifest.parameters=parameters;
    assert.throws(()=>C.parseRunFiles(files(run)),/Parameter count/);
  }
  for(const parameters of [NaN,Infinity,-Infinity]){const run=fresh();run.manifest.parameters=parameters;assert.throws(()=>C.validateRun(run),/Parameter count/);}
  const encoded=files(fresh());encoded[0].text=encoded[0].text.replace('"parameters":137088','"parameters":"\\u003cb data-review-marker\\u003eencoded\\u003c/b\\u003e"');
  assert.throws(()=>C.parseRunFiles(encoded),/Parameter count/);
});
test('progress backup and stored-state restoration reject the same markup without overwriting the stored bytes',()=>{
  for(const parameters of [inertMarkup,[inertMarkup],null,'100']){
    const run=fresh();run.manifest.parameters=parameters;const imported=backup(run),raw=JSON.stringify(imported);
    assert.throws(()=>C.validateState(imported),/Parameter count/);
    const current=C.blankState();current.notes[C.NOTE_FIELDS[0]]='Keep this hypothesis';const before=JSON.stringify(current);
    assert.throws(()=>C.mergeState(current,imported),/Parameter count/);assert.equal(JSON.stringify(current),before);
    let writes=0;const restored=C.loadState({getItem:()=>raw,setItem(){writes++;}});
    assert.ok(restored.error);assert.deepEqual(restored.state.runs,[]);assert.equal(writes,0);
  }
});
test('nonnegative safe counts and omitted count retain legitimate file and backup behavior',()=>{
  for(const parameters of [0,137088,Number.MAX_SAFE_INTEGER]){const run=fresh();run.manifest.parameters=parameters;assert.equal(C.parseRunFiles(files(run)).manifest.parameters,parameters);assert.equal(C.validateState(backup(run)).runs[0].manifest.parameters,parameters);}
  const absent=fresh();delete absent.manifest.parameters;assert.equal(C.parseRunFiles(files(absent)).manifest.parameters,undefined);assert.equal(C.validateState(backup(absent)).runs[0].manifest.parameters,undefined);
});
test('actual failed artifact imports and restores with failure text and unknown requested limit',()=>{
  const run=fresh();run.result=structuredClone(failedFixture);const imported=C.parseRunFiles(files(run));
  assert.equal(imported.result.status,'failed');assert.equal(imported.result.step,201);assert.equal(imported.result.requested_final_step,undefined);assert.equal(imported.result.error,failedFixture.error);
  assert.deepEqual(C.validateState(backup(imported)).runs[0].result,failedFixture);
  const restored=C.loadState({getItem:()=>JSON.stringify(backup(imported))});assert.equal(restored.error,null);assert.equal(restored.state.runs[0].result.status,'failed');
});
test('failed records retain step consistency and validate any supplied requested limit',()=>{
  const base=fresh();base.result=structuredClone(failedFixture);
  for(const mutate of [r=>r.result.step=199,r=>r.result.step=-1,r=>r.result.step=201.5,r=>r.result.requested_final_step=200,r=>r.result.requested_final_step=null,r=>r.result.requested_final_step='300']){const run=structuredClone(base);mutate(run);assert.throws(()=>C.validateRun(run),/recorded steps/);}
  for(const step of [200,201]){const run=structuredClone(base);run.result.step=step;assert.equal(C.validateRun(run).result.step,step);}
  base.result.requested_final_step=300;assert.equal(C.validateRun(base).result.requested_final_step,300);
});
test('completed and interrupted records retain strict required limits and completed exactness',()=>{
  for(const status of ['completed','interrupted']){const run=fresh();run.result.status=status;delete run.result.requested_final_step;assert.throws(()=>C.validateRun(run),/recorded steps/);}
  const completed=fresh();completed.result.step=201;completed.result.requested_final_step=201;assert.throws(()=>C.validateRun(completed),/recorded steps/);
  const interrupted=fresh();interrupted.result={status:'interrupted',step:201,requested_final_step:300};assert.equal(C.validateRun(interrupted).result.step,201);
  assert.equal(C.validateRun(fresh()).result.status,'completed');
});
test('error text accepts empty/absent legacy messages and rejects nonstrings',()=>{
  for(const error of ['',undefined,'A harmless error']){const run=fresh();run.result={status:'failed',step:201};if(error!==undefined)run.result.error=error;assert.equal(C.validateRun(run).result.status,'failed');}
  for(const error of [null,1,[],{},false]){const run=fresh();run.result={status:'failed',step:201,error};assert.throws(()=>C.validateRun(run),/plain text/);}
});
// Template evaluation only: no browser, DOM renderer, active payload, or network.
function render(run){
  const context=vm.createContext({C,document:{addEventListener(){},querySelectorAll(){return []}},window:{localStorage:{getItem(){return null}}},URLSearchParams,setTimeout,clearTimeout});
  const source=fs.readFileSync(path.join(root,'dist/app.js'),'utf8').replace("import * as C from './core.js';",'').replace(/init\(\);\s*$/,'');
  vm.runInContext(source,context);context.inputRun=run;vm.runInContext("selectedRun='import-0'",context);return vm.runInContext('resultsDetail(inputRun)',context);
}
test('result template escapes formatted count even when validation is deliberately bypassed',()=>{
  for(const parameters of [inertMarkup,[inertMarkup]]){const run=fresh();run.manifest.parameters=parameters;const html=render(run);assert.ok(!html.includes(inertMarkup));assert.ok(html.includes(C.escapeHTML(inertMarkup)));}
  const zero=fresh();zero.manifest.parameters=0;assert.match(render(zero),/<dt>Parameters<\/dt><dd>0<\/dd>/);
  const absent=fresh();delete absent.manifest.parameters;assert.match(render(absent),/<dt>Parameters<\/dt><dd>Unknown<\/dd>/);
  assert.ok(render(fresh()).includes(fresh().manifest.parameters.toLocaleString()));
});
test('failed result displays the error as escaped text and the requested limit as unknown',()=>{
  const run=fresh();run.result={...failedFixture,error:'<em data-review-marker>failed & retained</em>\nSecond line'};const html=render(C.parseRunFiles(files(run)));
  assert.match(html,/Recorded run · failed/);assert.match(html,/Training failed\./);assert.match(html,/Requested final step: unknown/);assert.ok(!html.includes(run.result.error));assert.ok(html.includes(C.escapeHTML(run.result.error)));assert.match(html,/Reported final step<\/dt><dd>201/);
  run.result.error='';assert.match(render(run),/No error text was supplied/);
});


test('supplied result scalars cannot bypass file, backup, or stored-state validation',()=>{
  for(const result of [false,0,'',true,1,[],{}]){
    const run=fresh();run.result=result;
    assert.throws(()=>C.parseRunFiles(files(run)),/Result must|Unknown result status/);
    assert.throws(()=>C.validateState(backup(run)),/Result must|Unknown result status/);
    const restored=C.loadState({getItem:()=>JSON.stringify(backup(run))});
    assert.ok(restored.error);assert.deepEqual(restored.state.runs,[]);
  }
  const run=fresh();run.result=null;
  assert.equal(C.validateState(backup(run)).runs[0].result,null);
  assert.equal(C.parseRunFiles(files(run).filter(f=>f.name!=='result.json')).result,null);
});

test('an empty selected result file is rejected rather than treated as absent',()=>{
  for(const text of ['', '   ', '\n']){
    const input=files(fresh());input.find(f=>f.name==='result.json').text=text;
    assert.throws(()=>C.parseRunFiles(input),/malformed/);
  }
});
