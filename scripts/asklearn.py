import faulthandler
import argparse

from amle import runner
from amle import settings

if __name__ == '__main__':
    faulthandler.enable()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', '-d', default=settings.default_working_dir, help="Working directory.")
    parser.add_argument('--seed', '-s', default=1, type=int, help="Random seed.")
    parser.add_argument('--total_time_limit', '-t', type=int, default=120,
                        help="Total time (secs) given to search hyperparameters for a given dataset.")
    parser.add_argument('--dids', default=None, nargs='*', type=str,
                        help="OpenML dataset IDs to use (all test dataset ids are used by default).")
    parser.add_argument('--use_default_params', default=False, action='store_true', help="Run Autosklearn with default parameters.")
    args = parser.parse_args()

    print "-"*75
    print "ARGUMENTS"
    for k, v in vars(args).iteritems():
        print k, v
    print "-"*75

    runner.ParallelRunner(**vars(args)).execute()
