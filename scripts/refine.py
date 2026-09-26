from pathlib import Path
r=Path(__file__).resolve().parents[1]
p=r/'dist/app.js';s=p.read_text(encoding='utf-8-sig')
s=s.replace("el.setAttribute('aria-expanded',String(menu));}","el.setAttribute('aria-expanded',String(menu));if(menu)$('#sidebar a').focus();}")
s=s.replace("if(file.size>5_000_000)throw Error('Backup is too large. Maximum size is 5 MB.')", "if(file.size>25_000_000)throw Error('Backup is too large. Maximum size is 25 MB.')")
s=s.replace("if(el.files.length>4||[...el.files].some(f=>f.size>2_000_000))throw Error('Select up to four files, each under 2 MB.')", "if(el.files.length>4||[...el.files].reduce((n,f)=>n+f.size,0)>1_000_000)throw Error('Select up to four recorded files, under 1 MB combined.')")
s=s.replace('${rows.map(r=>`<text x="${x(r.step)}"', '${rows.filter((r,i)=>i===0||i===rows.length-1||i%Math.max(1,Math.ceil(rows.length/5))===0).map(r=>`<text x="${x(r.step)}"')
p.write_text(s,encoding='utf-8')
p=r/'qa/audit.py';s=p.read_text(encoding='utf-8-sig').replace('.read_text()', ".read_text(encoding='utf-8')");p.write_text(s,encoding='utf-8')
p=r/'scripts/sync_course.py';s=p.read_text(encoding='utf-8-sig').replace('.read_text()', ".read_text(encoding='utf-8')");s=s.replace("receipt.write_text(json.dumps(manifest, indent=2) + '\\n')", "receipt.write_text(json.dumps(manifest, indent=2) + '\\n', encoding='utf-8')");s=s.replace(".write_text(json.dumps(content, ensure_ascii=False))", ".write_text(json.dumps(content, ensure_ascii=False), encoding='utf-8')");p.write_text(s,encoding='utf-8')
