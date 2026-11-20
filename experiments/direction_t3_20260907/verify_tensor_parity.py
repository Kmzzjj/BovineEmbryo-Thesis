import ast, random, typing, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch, cv2
from PIL import Image
from torch.utils.data import Dataset
from direction_support import select_centered,cached_patch

root=Path('/path/to/project/exp/direction_t3_20260907')
base=Path('/path/to/project/exp/agent_initiated_supplementary_20260829/window_gui_only_repro_20260829_frozen/ablation_agent_window_gui_only_r50_cbam_tfm/meta')
rows=pd.read_csv(base/'sequence_rows_used.csv')
mani=rows.groupby('seq_id',sort=False).agg({'label':'first','center_min':'first'}).reset_index()
classes=sorted(mani.label.unique())
def load(path):
 tree=ast.parse(Path(path).read_text());names={'SeqDataset','safe_imread_bgr','build_big_path','to_tensor_rgb','is_black_patch'}
 nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 env=dict(globals(),List=typing.List,IM_SIZE=224,BIG_ROOT=Path('/path/to/project/Bovine'),DIRECTION_MODE='history',PATCH_CACHE=str(root/'rgb_cache'),IMAGENET_MEAN=np.array([.485,.456,.406],dtype=np.float32).reshape(1,1,3),IMAGENET_STD=np.array([.229,.224,.225],dtype=np.float32).reshape(1,1,3))
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
 return env['SeqDataset']
old=load('/path/to/project/notebook_script/run_paired_window_t3_t9_repro_20260829.py')
new=load(Path(__file__).with_name('run_direction_t3.py'))
checks=0
for train in [False,True]:
 a=old(rows,mani,classes,3,train);b=new(rows,mani,classes,3,train)
 for idx in [0,1,2,3,7,10,20,50,100,250,500,1000,2000,3000,3259]:
  random.seed(678+idx);x,y,m=a[idx]
  random.seed(678+idx);xx,yy,mm=b[idx]
  assert torch.equal(x,xx) and y==yy and torch.equal(m['mask'],mm['mask'])
  checks+=1
out={'history_tensor_bitwise_checks':checks,'training_and_evaluation':True,'cached_RGB_changes_no_tensor_values':True}
(root/'tensor_parity.json').write_text(json.dumps(out,indent=2));print(out)
