"""Reproduce frozen baseline on cached embeddings for diagnosis, without training."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import numpy as np
from diagnostic_frozen_kernel_20260908 import kernel,sha

root=Path('/path/to/project/exp/repair_supplement_20260907_v53_cached_v2')
baseline=Path('/path/to/project/exp/track_repair_v53_guarded_orphanfix/run_20260324_094127_fullrun_release')
data=pd.read_pickle(root/'selected_complete_wells.pkl')
audit=pd.read_csv('/path/to/project/repair_supplement_20260907/audit_v53_done_only.csv')
args=json.loads((baseline/'args.json').read_text())
mod,program,_=kernel(Path('/path/to/project/notebook_script/repair_pipeline53_guarded_orphanfix_release.py'))
allout=[]
for case in audit[audit.fix_success=='fail'].itertuples():
    frame=data[(data.dish==case.dish)&(data.p==case.p)&(data.fp==case.fp)].copy()
    wid=int(frame.well_id.iloc[0]); shards=list((baseline/'emb_shards').glob(f'wid{wid:05d}_*.npz'));assert len(shards)==1
    cache=mod.load_emb_cache_npz(shards[0])
    s=dict(vars(mod));s.update(args=SimpleNamespace(**args),wid=wid,df_w=frame.copy(),emb_cache=cache,
        track_id_seg=pd.Series(-1,index=frame.index),n_clusters_well=pd.Series(-1,index=frame.index))
    exec(program,s)
    mappings={seg.seg_id:s['ids_safe'][i]['track_id_final_strict'] for i,seg in enumerate(s['segs'])}
    assert (s['df_w'].track_id_seg==frame.track_id_seg).all()
    assert (s['df_w'].track_id_seg.map(mappings).fillna(-1).astype(int)==frame.track_id_final_strict_safe).all()
    stats=mod.build_seg_stats_from_df(s['df_w']); sg=[]
    for i,seg in enumerate(s['segs']):
        sg.append(dict(index=i,seg=seg.seg_id,stats=stats[seg.seg_id],orig=s['colors_orig'][i],rescued=s['colors_rescued'][i],safe=s['colors_safe'][i],
            conflicts=[s['segs'][j].seg_id for j in s['conflicts'][i]],rescue=s['rescue_meta'][i],promotion=s['promote_meta'][i]))
    transitions=[]
    for tid,g in frame.groupby('track_id'):
        g=g.sort_values('tmin').reset_index(drop=True)
        for k in range(1,len(g)):
            prev=g.iloc[k-1];cur=g.iloc[k]
            if not (case.t_prev-250<=cur.tmin<=case.t_cur+40):continue
            left=mod._window_mean([cache.get(p) for p in g.patch_path.astype(str)],max(0,k-int(args['split_win'])),k)
            right=mod._window_mean([cache.get(p) for p in g.patch_path.astype(str)],k,min(len(g),k+int(args['split_win'])))
            cos=None if left is None or right is None else mod.cosine_np(left,right)
            geom,dbg=mod._boundary_geom_break(g,k,s['split_cfg'])
            transitions.append(dict(raw=int(tid),t0=float(prev.tmin),t1=float(cur.tmin),seg0=int(prev.track_id_seg),seg1=int(cur.track_id_seg),cos=cos,geom_break=geom,**dbg))
    rewards=[]
    for (i,j),reward in s['succ_rewards'].items():
        p,q,r=mod.reward_pair(s['segs'][i],s['segs'][j],args['w_proto'],args['w_seq'],args['w_gap'],args['seq_topk'])
        rewards.append(dict(prev=s['segs'][i].seg_id,next=s['segs'][j].seg_id,proto=p,seq=q,reward=r))
    allout.append(dict(event=int(case.event_idx),well_id=wid,K=int(s['K_real']),baseline_reproduced=True,segments=sg,transitions=transitions,rewards=rewards))
    print(f'event {case.event_idx}: baseline reproduced, {len(sg)} segments',flush=True)
out=Path('/path/to/project/exp/failure_diagnostics_20260908_v1/algorithm_trace.json')
with out.open('x') as f:json.dump(allout,f,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))
