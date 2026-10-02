"""Audit baseline reuse, frame support, exact RGB cache, and freeze before training."""
import ast, hashlib, json, os, sys, typing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch, cv2
from PIL import Image
from torch.utils.data import Dataset
from direction_support import history_pick, select_centered, patch_key

HERE=Path(__file__).resolve().parent
BASE=Path('/path/to/project/exp/agent_initiated_supplementary_20260829/window_gui_only_repro_20260829_frozen/ablation_agent_window_gui_only_r50_cbam_tfm')
NEW=Path('/path/to/project/exp/direction_t3_20260907/preflight/ablation_direction_t3')
OUT=Path('/path/to/project/exp/direction_t3_20260907')
CACHE=OUT/'rgb_cache'
SRC=Path('/path/to/project/notebook_script/run_paired_window_t3_t9_repro_20260829.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
expected='0f467f5cdeef647120b4098822a230ed552e20794961acb56cbdca9530939715'
assert sha(SRC)==expected
tree=ast.parse(SRC.read_text())
names={'safe_imread_bgr','build_big_path','SeqDataset','to_tensor_rgb','is_black_patch'}
nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
env=dict(globals(),List=typing.List,Tuple=typing.Tuple,Union=typing.Union,Optional=typing.Optional,IM_SIZE=224,random=__import__('random'))
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SRC),'exec'),env)
crop=env['SeqDataset'].crop_rgb
build=env['build_big_path']; read=env['safe_imread_bgr']; big=Path('/path/to/project/Bovine')
for file in ['sequence_rows_used.csv','fixed_split_seq_id.csv']:
 a=pd.read_csv(BASE/'meta'/file);b=pd.read_csv(NEW/'meta'/file)
 pd.testing.assert_frame_equal(a,b)
rows=pd.read_csv(BASE/'meta/sequence_rows_used.csv');split=pd.read_csv(BASE/'meta/fixed_split_seq_id.csv')
assert len(split)==3260
baseline=[]
for seed in range(10):
 run=BASE/'runs'/f'exp_agent_window_gui_only_r50_cbam_tfm__hist_center_T03__seed{seed:02d}'
 assert (run/'eval/DONE.txt').is_file()
 summ=json.loads((run/'eval/run_summary.json').read_text())
 assert summ['script_sha256']==expected and summ['run_seed']==seed
 cfg=json.loads((run/'meta/run_config.json').read_text())
 assert cfg['T']==3
 preds=pd.read_csv(run/'eval/test_predictions.csv');assert len(preds)==331
 pd.testing.assert_frame_equal(pd.read_csv(run/'meta/fixed_split_seq_id.csv'),split)
 pd.testing.assert_frame_equal(pd.read_csv(run/'meta/sequence_rows_used.csv'),rows)
 baseline.append({'seed':seed,'run':str(run),'summary':summ,'prediction_sha256':sha(run/'eval/test_predictions.csv')})
entries=[]; tasks={}; no_future=0; full_history=0; exact=0
for sid,sub in rows.groupby('seq_id',sort=False):
 sub=sub.sort_values('tmin').reset_index(drop=True);cmin=int(sub.center_min.iloc[0])
 h=history_pick(sub,cmin);c=select_centered(sub,h,cmin)
 ht=[int(sub.tmin.iloc[i]) for i in h];ct=[int(sub.tmin.iloc[i]) for i in c]
 assert all(t<=cmin for t in ht)
 target=ht[-1];desired=2*target-ht[-2] if len(ht)>1 else target
 ft=[t for t in ct if t>cmin]
 no_future+=not bool(ft);full_history+=len(h)==3;exact+=bool(ft) and ft[0]==desired
 entries.append({'seq_id':sid,'history_times':ht,'centered_times':ct,'center_min':cmin,'desired_future':desired,'future_offset_error':ft[0]-desired if ft else None,'has_future':bool(ft)})
 for i in set(h+c):
  r=sub.iloc[i];path=build(r,big);box=tuple(int(r[k]) for k in ['x1','y1','x2','y2'])
  tasks[patch_key(path,box)]=(path,box)
CACHE.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
def process(item):
 key,(path,box)=item
 im=read(str(path));assert im is not None,str(path)
 patch=crop(None,im,box);out=CACHE/(key+'.npy')
 if out.exists():assert np.array_equal(np.load(out),patch)
 else:np.save(out,patch,allow_pickle=False)
 # All cached patches checked against original crop/resize/conversion.
 assert np.array_equal(np.load(out),patch)
 return key
with ThreadPoolExecutor(max_workers=8) as pool:
 for n,_ in enumerate(pool.map(process,tasks.items()),1):
  if n%1000==0:print('cached and pixel-verified',n,'/',len(tasks),flush=True)
report={'baseline_sha256':expected,'cohort':len(split),'split_counts':split.split.value_counts().to_dict(),'baseline_runs':baseline,'sequence_rows_sha256':sha(BASE/'meta/sequence_rows_used.csv'),'split_sha256':sha(BASE/'meta/fixed_split_seq_id.csv'),'history_has_3':full_history,'no_future':no_future,'exact_reflected_future':exact,'cached_patches':len(tasks),'pixel_checks':'every selected crop matches original crop function exactly','selection':entries,'code_sha256':{p.name:sha(p) for p in HERE.glob('*.py')}}
(OUT/'preflight_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['baseline_runs','selection']},ensure_ascii=False,indent=2),flush=True)
