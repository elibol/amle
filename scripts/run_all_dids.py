
import os
from subprocess import call

from amle.data import dataset_ids
from amle import settings

test_dataset_ids = dataset_ids.test_dids

# certain datasets need more time to generate 200 iterations...
dur_hours = 4.0
duration = str(int(dur_hours*60*60))

dids = test_dataset_ids

cmd = ["python", settings.pj(settings.repo_root, "scripts", "asklearn.py"),
       "-d", "4hr",
       "-t", duration,
       # uncomment this to use default parameters.
       # the filename will automatically be appended with '_defaults'.
       # "--use_default_params",
       "--dids", " ".join(dids),
]

print " ".join(cmd)
# print call(cmd)
