"""Supplementary cached-feature replay, not a new v53 implementation.

Execute hash-locked, unchanged snippets from the original main loop. Require
row-wise baseline reproduction before any parameter variants. Outputs never
overwrite historical runs. No biological accuracy is inferred here.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import random
import shutil
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

EXPECTED = '0eb86c8887251fc107483fd9187c865220a8b9fbad9a264abfc805512897c386'
VARIANTS = {
    'A0': {}, 'A1': {'unassigned_penalty': .20},
    'A2': {'min_attach_reward': .30},
    'A3': {'unassigned_penalty': .20, 'min_attach_reward': .30},
    'S1': {'rescue_min_proto': .55}, 'S2': {'rescue_min_proto': .65},
    'S3': {'rescue_min_seq': .55}, 'S4': {'rescue_min_seq': .65},
    'S5': {'rescue_min_reward': .75}, 'S6': {'rescue_min_reward': .85},
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as src:
        for block in iter(lambda: src.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as dst:
        json.dump(value, dst, ensure_ascii=False, indent=2, allow_nan=False)


def prepare(cli):
    out = cli.output
    out.mkdir(parents=True, exist_ok=False)
    source_hash = sha(cli.code)
    if source_hash != EXPECTED:
        raise ValueError('Original v53 source does not match frozen baseline')
    args = json.loads((cli.baseline / 'args.json').read_text())
    for k, v in dict(unassigned_penalty=.10, min_attach_reward=.25,
                     rescue_min_proto=.60, rescue_min_seq=.60,
                     rescue_min_reward=.80).items():
        if args[k] != v:
            raise ValueError(f'Baseline parameter differs: {k}')
    audit = pd.read_csv(cli.audit)
    if len(audit) != 56 or audit.event_key.duplicated().any():
        raise ValueError('Historical audit identity mismatch')
    stats = pd.read_csv(cli.baseline / 'per_well_stats.csv')
    cols = ['dish', 'p', 'fp']
    selected = stats.merge(audit[cols].drop_duplicates(), on=cols, validate='one_to_one')
    if len(selected) != len(audit[cols].drop_duplicates()):
        raise ValueError('Some historical event wells are missing')
    historic = sorted(map(int, selected.well_id))
    other = sorted(set(map(int, stats.well_id)) - set(historic))
    guard = sorted(random.Random(20260907).sample(other, min(20, len(other))))
    well_ids = historic + guard
    # Freeze selection before reading any newly computed variant outcome.
    save(out / 'protocol.json', {
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'status': 'frozen_followup_not_original_prespecified_experiment',
        'code': str(cli.code), 'code_sha256': source_hash,
        'runner_sha256': sha(__file__), 'audit_sha256': sha(cli.audit),
        'baseline': str(cli.baseline), 'baseline_args': args,
        'historical_event_wells': historic, 'reference_wells': guard,
        'reference_selection': 'seed 20260907; remaining wells; not assumed normal',
        'variants': VARIANTS, 'evaluation': 'structural_and_membership_only',
        'human_identity_review_required': True,
    })
    if cli.reuse_prepared:
        previous = json.loads((cli.reuse_prepared / 'input_manifest.json').read_text())
        previous_protocol = json.loads((cli.reuse_prepared / 'protocol.json').read_text())
        if previous_protocol['baseline_args'] != args or previous_protocol['baseline'] != str(cli.baseline):
            raise ValueError('Prepared source configuration differs')
        if sha(cli.reuse_prepared / 'selected_complete_wells.pkl') != previous['selected_sha256']:
            raise ValueError('Prepared source hash differs')
        data = pd.read_pickle(cli.reuse_prepared / 'selected_complete_wells.pkl')
        total = previous['input_rows']
    else:
        chunks = []
        total = 0
        for chunk in pd.read_csv(cli.baseline / 'tracks_repaired.csv', chunksize=100000, low_memory=False):
            chunk['source_row'] = np.arange(total, total + len(chunk))
            total += len(chunk)
            chunks.append(chunk[chunk.well_id.isin(well_ids)].copy())
        data = pd.concat(chunks, ignore_index=True)
    if set(map(int, data.well_id.unique())) != set(well_ids):
        raise ValueError('Selected well data incomplete')
    if cli.reuse_prepared:
        shutil.copy2(cli.reuse_prepared / 'selected_complete_wells.pkl', out / 'selected_complete_wells.pkl')
    else:
        data.to_pickle(out / 'selected_complete_wells.pkl')
    save(out / 'input_manifest.json', {
        'input_rows': total, 'selected_rows': len(data), 'selected_wells': len(well_ids),
        'tracks_sha256': sha(cli.baseline / 'tracks_repaired.csv'),
        'selected_sha256': sha(out / 'selected_complete_wells.pkl'),
        'stats_sha256': sha(cli.baseline / 'per_well_stats.csv'),
        'args_sha256': sha(cli.baseline / 'args.json'),
    })
    print(json.dumps({'prepared_wells': len(well_ids), 'rows': len(data)}), flush=True)


def kernel(code):
    if sha(code) != EXPECTED:
        raise ValueError('Source changed')
    spec = importlib.util.spec_from_file_location('frozen_v53', code)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    source = code.read_text()
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    estimator = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == 'estimate_k_from_timeline')
    cfg_start = source.index('    split_cfg = SplitCfg(')
    cfg_end = source.index('    track_id_seg = pd.Series', cfg_start)
    start = source.index('        df_w["track_id_raw"] =')
    end = source.index('        dish0 = ', start)
    program = '\n'.join([textwrap.dedent(source[cfg_start:cfg_end]),
                         ast.get_source_segment(source, estimator),
                         'for _cached_iteration in range(1):\n' + textwrap.indent(textwrap.dedent(source[start:end]), '    ')])
    return module, compile(program, str(code) + ':unchanged_cached_loop', 'exec'), program


def evaluate(module, program, frame, cache, args, wid):
    scope = dict(vars(module))
    scope.update(args=SimpleNamespace(**args), wid=wid, df_w=frame.copy(),
                 emb_cache=cache, track_id_seg=pd.Series(-1, index=frame.index),
                 n_clusters_well=pd.Series(-1, index=frame.index))
    exec(program, scope)
    result = frame[['source_row', 'dish', 'p', 'fp', 'tmin', 'track_id_seg']].copy()
    regenerated = scope['df_w']['track_id_seg']
    if not np.array_equal(frame.track_id_seg, regenerated):
        raise ValueError(f'Appearance split reproduction failed for well {wid}')
    if not (frame.n_clusters_well.astype(int) == int(scope['K_real'])).all():
        raise ValueError(f'Capacity reproduction failed for well {wid}')
    segs = scope['segs']
    for stage in ('orig', 'rescued', 'safe'):
        mapping = {s.seg_id: scope['ids_' + stage][i]['track_id_final_strict'] for i, s in enumerate(segs)}
        result[stage] = regenerated.map(mapping).fillna(-1).astype('int64')
    # Exclude unresolved -1 from identity edges; compare segment memberships, not ID names.
    links = []
    for identity in sorted(set(result.safe) - {-1}):
        members = [s for i, s in enumerate(segs) if scope['ids_safe'][i]['track_id_final_strict'] == identity]
        members.sort(key=lambda s: (s.t0, s.t1, s.seg_id))
        links.extend((int(a.seg_id), int(b.seg_id)) for a, b in zip(members, members[1:]))
    assigned = result[result.safe >= 0]
    collisions = int((assigned.groupby(['tmin', 'safe']).size() > 1).sum())
    summary = dict(well_id=wid, rows=len(result), segments=len(segs),
                   unresolved_rows=int((result.safe < 0).sum()),
                   identity_time_collision_groups=collisions,
                   chronological_identity_links=links)
    return result, summary


def run(cli):
    out = cli.output
    protocol = json.loads((out / 'protocol.json').read_text())
    manifest = json.loads((out / 'input_manifest.json').read_text())
    if sha(out / 'selected_complete_wells.pkl') != manifest['selected_sha256']:
        raise ValueError('Selected data changed')
    if sha(__file__) != protocol['runner_sha256']:
        raise ValueError('Runner changed after protocol freeze')
    args = protocol['baseline_args']
    module, program, text = kernel(Path(protocol['code']))
    destination = out / cli.mode
    destination.mkdir(exist_ok=False)
    (destination / 'executed_original_snippets.py').write_text(text)
    data = pd.read_pickle(out / 'selected_complete_wells.pkl')
    wells = sorted(map(int, data.well_id.unique()))
    if cli.mode == 'smoke':
        wells = wells[:3]
    totals = []
    for wid in wells:
        frame = data[data.well_id == wid].copy()
        paths = sorted(frame.patch_path.astype(str).unique().tolist())
        shard = Path(protocol['baseline']) / 'emb_shards' / f'wid{wid:05d}_{module._hash_list(paths)}.npz'
        cache = module.load_emb_cache_npz(shard)
        if not all(p in cache for p in paths):
            raise ValueError(f'Missing frozen embeddings for well {wid}')
        start = time.monotonic()
        baseline, base_summary = evaluate(module, program, frame, cache, args, wid)
        mismatches = {stage: int((baseline[stage].to_numpy() != frame['track_id_final_strict_' + stage].to_numpy()).sum())
                      for stage in ('orig', 'rescued', 'safe')}
        if any(mismatches.values()):
            save(destination / f'BASELINE_MISMATCH_{wid}.json', mismatches)
            raise ValueError(f'Historical baseline mismatch at well {wid}: {mismatches}')
        repeat, _ = evaluate(module, program, frame, cache, args, wid)
        if not baseline.equals(repeat):
            raise ValueError(f'Baseline repeat is not deterministic at well {wid}')
        well_dir = destination / f'well_{wid:05d}'
        well_dir.mkdir()
        baseline.to_csv(well_dir / 'A0_rows.csv', index=False)
        save(well_dir / 'A0_summary.json', base_summary)
        entry = {'well_id': wid, 'baseline_exact': True, 'repeat_exact': True,
                 'shard_sha256': sha(shard), 'variants': {'A0': base_summary}}
        if cli.mode == 'run':
            for name, changes in VARIANTS.items():
                if name == 'A0':
                    continue
                result, summary = evaluate(module, program, frame, cache, dict(args, **changes), wid)
                old = set(map(tuple, base_summary['chronological_identity_links']))
                new = set(map(tuple, summary['chronological_identity_links']))
                summary['added_links'] = sorted(new - old)
                summary['removed_links'] = sorted(old - new)
                summary['biological_review'] = 'not_performed'
                result.to_csv(well_dir / f'{name}_rows.csv', index=False)
                save(well_dir / f'{name}_summary.json', summary)
                entry['variants'][name] = summary
        entry['elapsed_seconds'] = time.monotonic() - start
        save(well_dir / 'COMPLETE.json', entry)
        totals.append(entry)
        print(json.dumps({'completed_well': wid, 'completed': len(totals), 'total': len(wells),
                          'elapsed_seconds': entry['elapsed_seconds']}), flush=True)
    save(destination / 'COMPLETE.json', {'completed_wells': len(totals), 'results': totals,
                                       'human_identity_review_required': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'smoke', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--code', type=Path)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--reuse-prepared', type=Path)
    cli = parser.parse_args()
    if cli.mode == 'prepare':
        if not all((cli.baseline, cli.code, cli.audit)):
            parser.error('prepare requires baseline, code and audit')
        prepare(cli)
    else:
        run(cli)
