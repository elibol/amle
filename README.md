# SETUP

Please use pip to install requirements.
1. `pip install -r requirements.txt`
2. `pip install -r requirements-git.txt`
3. `python setup.py develop` (to expose amle package to scripts)

# Notes

Everything was tested with Anaconda (https://www.continuum.io/downloads) in a conda environment created with command `conda create --name amle python=2.7`.

Please notify us of any missing dependencies in different environments.

# Run Auto-SKLearn Experiment

1. `python scripts/run_all_dids.py`
2. with default values in `amle/settings.py`, results will be written to `tmp/{did}_results.json`, where
   `{did}` corresponds to an openml dataset id.
