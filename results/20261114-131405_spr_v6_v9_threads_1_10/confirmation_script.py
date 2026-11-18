#!/usr/bin/env python3
"""Repeat selected V6/V9 comparisons in two fresh processes per setting; no verification."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
out=Path(sys.argv[1]).resolve()
manifest=json.loads((out/'manifest.json').read_text())
assert hashlib.sha256((ROOT/'build/bench_cpu').read_bytes()).hexdigest()==manifest['binary_hashes']['bench_cpu']
with (out/'comparison.csv').open() as f: primary=list(csv.DictReader(f))
cases=[('high_k8',64),('high_k6',512),('clustered',320),('tail',63)]
env=dict(os.environ,OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE',KMP_BLOCKTIME='0',OMP_PROC_BIND='close',OMP_PLACES='cores')
raw=out/'confirmation.csv'
if raw.exists():raise SystemExit('Preserve/rename confirmation.csv before rerunning')
commands=[]
with (out/'confirmation.log').open('w') as log:
 for rep in range(2):
  for category,M in cases:
   p=next(r for r in primary if r['category']==category and int(r['M'])==M and float(r['sparsity'])==.5 and int(r['threads'])==1)
   for t in (1,10):
    cmd=[str(ROOT/'build/bench_cpu'),'--M',str(M),'--K',p['K'],'--N',p['N'],'--s','.5','--k',p['param'].split('=')[1],
         '--threads',str(t),'--sparsity-pattern','column_clustered' if category=='clustered' else 'iid',
         '--kernels','=spr.v6.b32,=spr.v9.b32','--rounds','61','--warmup','3','--verify','none','--ref-rows','0',
         '--seed','42','--tag',f'confirm{rep}_{category}','--out',str(raw)]
    print('Confirm',rep+1,category,'M',M,'T',t,flush=True)
    commands.append(cmd)
    log.write('\n$ '+' '.join(cmd)+'\n');log.flush()
    subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=log,check=True)
(out/'confirmation_manifest.json').write_text(json.dumps(dict(commands=commands,rounds=61,repetitions=2,
    correctness_run=False,binary_sha256=manifest['binary_hashes']['bench_cpu']),indent=2)+'\n')
with raw.open() as f:rows=list(csv.DictReader(f))
assert len(rows)==32 and all(float(r['max_err'])==-1 and int(r['rounds'])==61 for r in rows)
print('Confirmation complete:',len(rows)*61,'timed inference calls; no correctness checks.')
