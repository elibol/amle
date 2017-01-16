import os
from subprocess import call

from amle.data import dataset_ids
from amle import settings

test_dataset_ids = dataset_ids.test_dids
for did in test_dataset_ids:
    # data also available for 1.5 hour run.
    dur_hours = 3.0
    duration = str(dur_hours*60*60)
    cmd = ["python", "asklearn.py", "-t", duration, "--dids", did]
    print cmd
    call(cmd)
