"""Detached two-GPU smoke -> gated full runs, with immutable code manifest."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=Path('/path/to/project/exp/direction_t3_20260907')
PY='/path/to/project/.venv/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stamp():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def status(stage,**kwargs):
 (OUT/'status.json').write_text(json.dumps(dict(time=stamp(),stage=stage,**kwargs),indent=2));print(stage,kwargs,flush=True)
def verify_outputs(folder,n,epochs):
 paths=sorted(folder.rglob('run_summary.json'));assert len(paths)==n,(folder,len(paths))
 for p in paths:
  import pandas as pd
  r=json.loads(p.read_text());assert r['script_sha256']==manifest['files']['run_direction_t3.py']
  assert (p.parent/'DONE.txt').exists()
  assert len(pd.read_csv(p.parent/'test_predictions.csv'))==331
  metrics=pd.read_csv(p.parent.parent/'logs/metrics.csv');assert int(metrics.epoch.max())==epochs
def run_phase(phase):
 for name,h in manifest['files'].items():assert sha(HERE/name)==h,name
 folder=OUT/phase;assert not folder.exists(),f'Refusing overwrite {folder}'
 folder.mkdir();processes=[];logs=[]
 for gpu in [0,1]:
  env=os.environ.copy();env.update(BATCH_SIZE='6',NUM_WORKERS='4',USE_CKPT='0',NEPTUNE_ON='0',ACCUM_STEPS='1',CUDA_VISIBLE_DEVICES=str(gpu),DIRECTION_MODE='centered',DIRECTION_PATCH_CACHE=str(OUT/'rgb_cache'),WINDOW_CONFIRM_BASE_EXP=f'direction_t3_{phase}',WINDOW_CONFIRM_OUT_DIR=str(folder),EPOCHS='1' if phase=='smoke' else '50',FREEZE_EPOCHS='0' if phase=='smoke' else '2',MIN_EPOCHS_FOR_EARLYSTOP='1' if phase=='smoke' else '50',SKIP_DONE='0',RESUME='0')
  cmd=[PY,str(HERE/'run_direction_t3.py'),'--only_T','3','--only_seeds','0,1' if phase=='smoke' else '0,1,2,3,4,5,6,7,8,9','--num_jobs','2','--job_index',str(gpu)]
  log=open(folder/f'gpu{gpu}.log','w');logs.append(log)
  p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);processes.append(p)
 status(phase+'_running',pids=[p.pid for p in processes],gpus=[0,1])
 codes=[p.wait() for p in processes]
 for log in logs:log.close()
 (folder/'exit_codes.json').write_text(json.dumps(codes));assert codes==[0,0],codes
 verify_outputs(folder,2 if phase=='smoke' else 10,1 if phase=='smoke' else 50)
 status(phase+'_verified')
try:
 assert not (OUT/'frozen_manifest.json').exists(),'Already launched; inspect status instead of duplicating'
 assert (OUT/'preflight_report.json').exists() and (OUT/'tensor_parity.json').exists()
 subprocess.run([PY,str(HERE/'test_direction.py')],check=True)
 manifest={'frozen_at':stamp(),'files':{p.name:sha(p) for p in HERE.iterdir() if p.suffix in ['.py','.md']},'preflight_sha256':sha(OUT/'preflight_report.json'),'tensor_parity_sha256':sha(OUT/'tensor_parity.json'),'gpus':[0,1],'new_training_runs':10,'baseline_reused_runs':10,'T':3,'seeds':list(range(10))}
 (OUT/'frozen_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 run_phase('smoke');run_phase('formal')
 status('complete_training_requires_analysis')
except Exception as exc:
 status('failed',error=repr(exc));raise
