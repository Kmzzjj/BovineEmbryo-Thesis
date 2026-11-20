"""Single experiment dependency: analyse after its training gate, then exit."""
import json,subprocess,time,hashlib
from pathlib import Path
root=Path('/path/to/project/exp/direction_t3_20260907')
script=Path(__file__).with_name('analyse_direction.py')
job={'analysis_script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'status':'waiting_for_training','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
(root/'analysis_job.json').write_text(json.dumps(job,indent=2))
deadline=time.monotonic()+24*3600
try:
 while time.monotonic()<deadline:
  status=json.loads((root/'status.json').read_text())
  if status['stage']=='failed':raise RuntimeError(status)
  if status['stage']=='complete_training_requires_analysis':break
  time.sleep(15)
 else:raise TimeoutError('24h pipeline limit exceeded; inspect training, do not restart automatically')
 assert hashlib.sha256(script.read_bytes()).hexdigest()==job['analysis_script_sha256']
 with open(root/'analysis.log','w') as log:
  subprocess.run(['/path/to/project/.venv/bin/python',str(script)],stdout=log,stderr=subprocess.STDOUT,check=True)
 job['status']='analysis_complete_requires_interpretation'
except Exception as exc:
 job.update(status='failed',error=repr(exc));raise
finally:
 job['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
 (root/'analysis_job.json').write_text(json.dumps(job,indent=2))
