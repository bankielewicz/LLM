import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import * as C from '../dist/core.js';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const course=JSON.parse(fs.readFileSync(path.join(root,'dist/content.json'),'utf8'));
// Pure template evaluation only: no browser, DOM renderer, automation, or network.
const inertDocument={querySelectorAll:()=>[],querySelector:()=>null,addEventListener:()=>{}};
const sandbox={C,document:inertDocument,window:{localStorage:{getItem:()=>null,setItem:()=>{}}},URLSearchParams,setTimeout,clearTimeout,console};
const context=vm.createContext(sandbox);
let source=fs.readFileSync(path.join(root,'dist/app.js'),'utf8').replace("import * as C from './core.js';",'').replace(/init\(\);\s*$/,'');
vm.runInContext(source,context);context.courseData=course;vm.runInContext('course=courseData;',context);
const cases=[['home','home()'],['roadmap','roadmap()'],...course.modules.map(m=>['lesson-'+m.id,`lesson('${m.id}')`]),...['tokenizer','prediction','attention','shapes'].map(t=>['explore-'+t,`explorer('${t}','02')`]),['results','results()'],['notebook','notebook()'],['capstone','capstone()'],['labs','labs()'],['checkpoints','checkpoints()'],['glossary','glossary()'],...Object.keys(course.references).map(p=>['reference-'+p,`reference(${JSON.stringify(p)})`])];
const outputs=[];for(const [name,expression] of cases){const html=vm.runInContext(expression,context);assert.ok(typeof html==='string'&&html.length>300,name);assert.ok(!html.includes('undefined'),name+' undefined');outputs.push({name,html});}
vm.runInContext("selectedRun='none'",context);assert.match(vm.runInContext('results()',context),/No measurements yet/);
vm.runInContext("selectedRun='bpe-check'",context);assert.match(vm.runInContext('results()',context),/4 recorded|four-update|4 updates/);
vm.runInContext("shapeConfig.heads=6",context);assert.match(vm.runInContext('shapesView()',context),/Width must divide/);
vm.runInContext("tokenText='<script>alert(1)</script>'",context);assert.ok(!vm.runInContext('tokenizerView()',context).includes('<script>'));
fs.writeFileSync(path.join(root,'qa/template-output.json'),JSON.stringify(outputs));
console.log(`${cases.length} page-template cases generated successfully; empty run, BPE identity, invalid shape, and text-escaping checks passed. This is not browser testing.`);
