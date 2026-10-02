"""Run only after formal completion; recompute predictions, pairing and statistics."""
import hashlib,itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score

ROOT=Path('/path/to/project/exp/direction_t3_20260907')
status=json.loads((ROOT/'status.json').read_text())
assert status['stage']=='complete_training_requires_analysis',status
audit=json.loads((ROOT/'preflight_report.json').read_text())
manifest=json.loads((ROOT/'frozen_manifest.json').read_text())
assert hashlib.sha256((ROOT/'preflight_report.json').read_bytes()).hexdigest()==manifest['preflight_sha256']
classes=['2-4 cell','5-16 cell','Blast','Dead/Degenerating/Bad','Morula','Unfertilised']
selection={x['seq_id']:x for x in audit['selection']}
rows=[];detail=[]
for b in audit['baseline_runs']:
 seed=b['seed'];old=Path(b['run'])/'eval/test_predictions.csv'
 assert hashlib.sha256(old.read_bytes()).hexdigest()==b['prediction_sha256']
 matches=list((ROOT/'formal').glob(f'*/runs/*seed{seed:02d}/eval/test_predictions.csv'));assert len(matches)==1
 new=matches[0]
 assert (new.parent/'DONE.txt').exists()
 config=json.loads((new.parent.parent/'meta/run_config.json').read_text())
 assert config['T']==3 and config['run_seed']==seed and config['direction_mode']=='centered'
 assert config['script_sha256']==manifest['files']['run_direction_t3.py']
 a=pd.read_csv(old);c=pd.read_csv(new)
 assert len(a)==len(c)==331 and a.seq_id.is_unique and c.seq_id.is_unique
 assert a.seq_id.tolist()==c.seq_id.tolist() and a.true.tolist()==c.true.tolist()
 metrics=pd.read_csv(new.parent.parent/'logs/metrics.csv');assert metrics.epoch.max()==50
 full=np.array([len(selection[s]['history_times'])==3 and selection[s]['has_future'] and selection[s]['future_offset_error']==0 for s in a.seq_id])
 record={'seed':seed,'history_macro_f1':f1_score(a.true,a.pred,labels=classes,average='macro',zero_division=0),'centered_macro_f1':f1_score(c.true,c.pred,labels=classes,average='macro',zero_division=0),'history_accuracy':accuracy_score(a.true,a.pred),'centered_accuracy':accuracy_score(c.true,c.pred),'test_n':len(a),'complete_mirrored_subset_n':int(full.sum())}
 record['delta_macro_f1']=record['centered_macro_f1']-record['history_macro_f1']
 assert abs(record['history_macro_f1']-b['summary']['test_f1@best'])<1e-10
 for name,pred in [('history',a),('centered',c)]:
  record[name+'_mirrored_subset_macro_f1_descriptive']=f1_score(pred.true[full],pred.pred[full],labels=classes,average='macro',zero_division=0)
  for label,value in zip(classes,f1_score(pred.true,pred.pred,labels=classes,average=None,zero_division=0)):
   detail.append({'seed':seed,'condition':name,'class':label,'f1':float(value)})
 rows.append(record)
assert sorted(r['seed'] for r in rows)==list(range(10))
d=np.array([r['delta_macro_f1'] for r in rows]);signs=np.array(list(itertools.product([-1,1],repeat=10)))
p=float(np.mean(np.abs((signs*d).mean(axis=1))>=abs(d.mean())-1e-12))
rng=np.random.default_rng(20260907);ci=np.quantile(d[rng.integers(0,10,size=(100000,10))].mean(axis=1),[.025,.975])
out={'contrast':'centered minus history, T3 only; follow-up design','n_pairs':10,'history_mean':float(np.mean([r['history_macro_f1'] for r in rows])),'centered_mean':float(np.mean([r['centered_macro_f1'] for r in rows])),'paired_delta':float(d.mean()),'bootstrap_95_ci':ci.tolist(),'exact_two_sided_sign_flip_p':p,'bootstrap_seed':20260907,'bootstrap_resamples':100000,'cohens_dz':float(d.mean()/d.std(ddof=1)) if d.std(ddof=1)>0 else None,'equivalence_test':'not performed; nonsignificance is not equivalence','limitations':['T3 only; chosen after previous results','fixed internal split, not external validation','boundary/irregular frame availability retained and recorded','paired seeds measure optimisation variability, not independent embryo cohorts'],'input_preflight_sha256':manifest['preflight_sha256'],'per_seed':rows}
dest=ROOT/'analysis';dest.mkdir(exist_ok=False)
(dest/'paired_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
pd.DataFrame(rows).to_csv(dest/'paired_results.csv',index=False)
pd.DataFrame(detail).to_csv(dest/'per_class_descriptive.csv',index=False)
print(json.dumps({k:v for k,v in out.items() if k!='per_seed'},ensure_ascii=False,indent=2))
