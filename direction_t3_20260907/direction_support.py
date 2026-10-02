"""Fixed-budget T3 direction contrast; no labels/results enter frame selection."""
from pathlib import Path
import hashlib
import numpy as np

def select_centered(sub, history_picked, center_min):
    
    
    
    if not history_picked:
        raise ValueError('No historical observations')
    times = sub.tmin.astype(int)
    past = [i for i in history_picked if int(times.iloc[i]) <= center_min]
    if len(past) != len(history_picked):
        raise ValueError('Historical input contains a future frame')
    target = past[-1]
    retained = past[-2:] if len(past) >= 2 else past[:]
    reference = retained[0]
    desired = 2 * int(times.iloc[target]) - int(times.iloc[reference])
    future = [i for i in range(len(sub)) if int(times.iloc[i]) > center_min]
    if future:
        chosen = min(future, key=lambda i: (abs(int(times.iloc[i])-desired), int(times.iloc[i]), i))
        retained.append(chosen)
    return sorted(retained, key=lambda i: int(times.iloc[i]))

def patch_key(path, box):
    return hashlib.sha256((str(path)+'|'+','.join(map(str,map(int,box)))).encode()).hexdigest()

def cached_patch(cache, path, box):
    return np.load(Path(cache)/(patch_key(path,box)+'.npy'), allow_pickle=False)

def history_pick(sub, cmin, T=3):
    idx = [i for i,t in enumerate(sub.tmin.astype(int)) if t <= cmin]
    if not idx:
        raise ValueError('No historical frame; refusing original future fallback')
    if len(idx) <= T:
        return idx
    picked = [idx[int(round(i*(len(idx)-1)/(T-1)))] for i in range(T)]
    return picked
