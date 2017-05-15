# SETUP

Please use pip to install requirements.

1. `pip install -r requirements.txt`
2. `pip install -r requirements-git.txt`
3. `python setup.py develop` (to expose amle package to scripts)

# Run auto-sklearn Experiment

1. `python scripts/run_all_dids.py`
2. Results will be written to the directory `{results_dir}` specified as parameter `-d` in `run_all_dids.py`. See `scripts/asklearn.py` for parameter descriptions.
3. Generate prediction results by running `python scripts/process_results.py {results_dir}`

# Notes

Everything was tested with Anaconda (https://www.continuum.io/downloads) in a conda environment created with command `conda create --name amle python=2.7`.

Please notify us of any missing dependencies in different environments.
