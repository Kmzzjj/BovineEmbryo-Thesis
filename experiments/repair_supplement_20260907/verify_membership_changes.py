"""Verify assignment changes independent of numeric ID renaming, incl. singletons."""
import argparse
import json
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
complete = json.loads((args.run / 'COMPLETE.json').read_text())

def identity_sets(path):
    frame = pd.read_csv(path, usecols=['track_id_seg','safe'])
    if (frame.groupby('track_id_seg').safe.nunique() != 1).any():
        raise ValueError('One segment assigned inconsistently')
    groups = frozenset(frozenset(map(int,g.track_id_seg.unique()))
                       for _,g in frame[frame.safe>=0].groupby('safe'))
    unresolved = frozenset(map(int,frame.loc[frame.safe<0,'track_id_seg'].unique()))
    return groups, unresolved

changes = []
for row in complete['results']:
    wid = row['well_id']
    folder = args.run / f'well_{wid:05d}'
    old, old_unresolved = identity_sets(folder / 'A0_rows.csv')
    for variant in row['variants']:
        if variant == 'A0':
            continue
        new, new_unresolved = identity_sets(folder / f'{variant}_rows.csv')
        if (old, old_unresolved) == (new, new_unresolved):
            continue
        changes.append(dict(well_id=wid, condition=variant,
                            added_groups=sorted([sorted(g) for g in new-old]),
                            removed_groups=sorted([sorted(g) for g in old-new]),
                            newly_assigned_segments=sorted(old_unresolved-new_unresolved),
                            newly_unresolved_segments=sorted(new_unresolved-old_unresolved),
                            human_judgement='pending'))
report = dict(verified_wells=len(complete['results']), changed_wells=sorted({r['well_id'] for r in changes}),
              changed_well_conditions=len(changes), changes=changes)
with args.output.open('x') as output:
    json.dump(report, output, indent=2)
print(json.dumps(report, indent=2))
