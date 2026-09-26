// Bounded fallback for this Site when the bundled publication helper is unavailable.
// Credentials enter through hidden stdin, live only in memory and a Git child environment.
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
const root=process.cwd();
const expected=JSON.parse(fs.readFileSync('.openai/hosting.json','utf8').replace(/^\uFEFF/,''));
if(!expected.project_id||expected.static?.directory!=='dist')throw Error('Invalid Site manifest');
if(process.stdin.isTTY)process.stdin.setRawMode(true);
process.stdin.setEncoding('utf8');process.stdin.resume();
console.log('Ready for publication JSON on stdin (input is hidden).');
let input='';
process.stdin.on('data',chunk=>{input+=chunk;if(!/[\r\n]/.test(input))return;process.stdin.pause();if(process.stdin.isTTY)process.stdin.setRawMode(false);try{publish(JSON.parse(input.trim()));}catch(error){console.error('Publication stopped: '+String(error.message).replace(/Bearer\s+\S+/g,'[redacted]'));process.exitCode=1;}});
function publish({credential,archivePath}){
 if(credential.provider!=='cloudflare_artifact'||credential.auth_mode!=='http_extra_header'||credential.repository!==expected.project_id||!/^https:\/\/git\.chatgpt-team\.site\//.test(credential.remote_url)||credential.branch!=='main')throw Error('Unexpected repository identity or authentication mode');
 const env={...process.env,GIT_TERMINAL_PROMPT:'0',GIT_CONFIG_COUNT:'2',GIT_CONFIG_KEY_0:'http.extraHeader',GIT_CONFIG_VALUE_0:'Authorization: Bearer '+credential.token,GIT_CONFIG_KEY_1:'safe.directory',GIT_CONFIG_VALUE_1:root.replaceAll('\\','/')};
 const run=(exe,args,opts={})=>{const r=spawnSync(exe,args,{cwd:root,env,encoding:'utf8',windowsHide:true,...opts});if(r.status!==0)throw Error((r.stderr||r.error?.message||`${exe} exited ${r.status}`).replaceAll(credential.token,'[redacted]').slice(0,1400));return r.stdout.trim();};
 const git=(...args)=>run('git',args);
 const remote=git('ls-remote','--heads',credential.remote_url,'refs/heads/'+credential.branch);
 if(remote)throw Error('Remote source already exists. No overwrite attempted. Reconcile explicitly.');
 if(!fs.existsSync('.git'))git('init','--initial-branch',credential.branch,'.');
 const top=git('rev-parse','--show-toplevel');if(fs.realpathSync(top)!==fs.realpathSync(root))throw Error('Unexpected Git root');
 git('add','--','.openai/hosting.json','.gitignore','dist','course','scripts','qa','README.md','DESIGN.md','QA_REPORT.md');
 git('-c','user.name=Sites','-c','user.email=sites@localhost','commit','-m','Build source-grounded LLM Foundations learner interface');
 const sha=git('rev-parse','HEAD');
 if(git('status','--porcelain'))throw Error('Uncommitted source remains');
 git('push',credential.remote_url,'HEAD:refs/heads/'+credential.branch);
 const readback=git('ls-remote','--heads',credential.remote_url,'refs/heads/'+credential.branch).split(/\s/)[0];if(readback!==sha)throw Error('Remote commit readback mismatch');
 if(fs.existsSync(archivePath))throw Error('Archive already exists; choose a fresh path');
 run('tar',['-cf',archivePath,'.openai/hosting.json','dist']);
 const listing=run('tar',['-tf',archivePath]);if(!listing.includes('dist/index.html')||!listing.includes('.openai/hosting.json'))throw Error('Archive missing required files');
 if(git('rev-parse','HEAD')!==sha||git('status','--porcelain'))throw Error('Source changed while packaging');
 console.log(JSON.stringify({project_id:expected.project_id,checkout_path:root,commit_sha:sha,archive:archivePath,archive_bytes:fs.statSync(archivePath).size}));
}
