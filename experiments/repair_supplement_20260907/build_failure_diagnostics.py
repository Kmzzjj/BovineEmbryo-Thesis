"""Read-only diagnostic panels for the 12 historical human-labelled failures."""
import argparse,json,hashlib
from pathlib import Path
import pandas as pd
from PIL import Image,ImageDraw,ImageFont

ap=argparse.ArgumentParser()
ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--audit',type=Path,required=True)
a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
d=pd.read_pickle('/path/to/project/exp/repair_supplement_20260907_v53_cached_v2/selected_complete_wells.pkl')
audit=pd.read_csv(a.audit); failures=audit[audit.fix_success=='fail']
assert len(failures)==12
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
records=[]
for c in failures.itertuples():
    w=d[(d.dish==c.dish)&(d.p==c.p)&(d.fp==c.fp)].copy()
    folder=Path('/path/to/project/Bovine')/f'MTL-0034-{c.dish}'
    prefix=f'{c.dish}_P{c.p}_FP{c.fp}_'
    paths={int(p.stem[len(prefix):]):p for p in folder.glob(prefix+'*.jpg') if p.stem[len(prefix):].isdigit()}
    ts=sorted(paths); times=sorted(set([t for t in ts if t<=c.t_prev][-6:]+[t for t in ts if t>=c.t_cur][:6]))
    tiles=[]
    for t in times:
        im=Image.open(paths[t]).convert('RGB'); sc=min(410/im.width,410/im.height)
        tile=Image.new('RGB',(440,490),'white');tile.paste(im.resize((round(im.width*sc),round(im.height*sc))),(15,70))
        dr=ImageDraw.Draw(tile);dr.text((8,5),f'E{c.event_idx} {c.dish} P{c.p} t={t}',font=font,fill='black')
        dr.text((8,26),'labels: safe ID / raw track / segment',font=font,fill='black')
        for r in w[w.tmin==t].itertuples():
            sid=int(r.track_id_final_strict_safe);col='red' if sid<0 else ['#00ff00','#ff5555','#00ffff','#ffff00','#ff77ff'][sid%5]
            box=(15+r.x1*sc,70+r.y1*sc,15+r.x2*sc,70+r.y2*sc)
            dr.rectangle(box,outline=col,width=2)
            dr.text((box[0],max(48,box[1]-18)),f'{sid}/{int(r.track_id)}/{int(r.track_id_seg)}',font=font,fill=col,stroke_width=1,stroke_fill='black')
        tiles.append(tile)
    sheet=Image.new('RGB',(1760,1470),'#dddddd')
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%4)*440,(i//4)*490))
    sheet.save(a.output/f'failure_{c.event_idx}.jpg',quality=92)
    subset=w[w.tmin.isin(times)]
    segs=set(w[w.tmin.isin([c.t_prev,c.t_cur])].track_id_seg)
    segsummary=[]
    for seg,g in w[w.track_id_seg.isin(segs)].groupby('track_id_seg'):
        r=g.iloc[0].to_dict(); r.update(t0=float(g.tmin.min()),t1=float(g.tmin.max()),rows=len(g));segsummary.append(r)
    records.append(dict(audit=c._asdict(),well_id=int(w.well_id.iloc[0]),times=times,rows=json.loads(subset.to_json(orient='records')),segments=segsummary))
    print(f'event {c.event_idx}: {len(times)} original frames',flush=True)
(a.output/'diagnostics.json').write_text(json.dumps(records,indent=2,default=str))
(a.output/'audit_sha256.txt').write_text(hashlib.sha256(a.audit.read_bytes()).hexdigest())
