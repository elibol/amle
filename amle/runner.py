import sys
import os
import json
from functools import partial
from operator import itemgetter
import math

import numpy as np

import sklearn
import sklearn.cross_validation
import sklearn.datasets
import sklearn.metrics

import autosklearn
import autosklearn.classification
from autosklearn.constants import *
from autosklearn.metrics import classification_metrics

from amle.data import dataset_ids
from amle import settings

from amle import utils


# make sure the right version of autosklearn is being used...
assert autosklearn.__version__ == '0.1.0-miro', 'wrong version of auto-sklearn, expected 0.1.0-miro got %s' \
                                                '\nsee requirements-git.txt for correct version.' % autosklearn.__version__


class ParallelRunner(object):

    def __init__(self, working_dir, seed, total_time_limit, dids, use_default_params):
        self.working_dir = working_dir
        self.seed = seed
        self.total_time_limit = total_time_limit
        self.use_default_params = use_default_params

        if self.use_default_params:
            self.working_dir += "_defaults"

        if not os.path.exists(self.working_dir):
            os.mkdir(self.working_dir)

        if dids is None:
            self.dids = dataset_ids.test_dids
        else:
            self.dids = dids
        assert len(set(self.dids) - set(dataset_ids.all_dids)) == 0, "invalid dids: %s" % " ".join(self.dids)

    def execute(self, num_procs=30):
        if num_procs >= len(self.dids):
            return self.execute_batch(self.dids)

        batch_size = num_procs
        batches = int(math.ceil(float(len(self.dids))/batch_size))

        check_dids = []
        for batch in range(batches):
            s = batch*batch_size
            e = (batch+1)*batch_size
            if batch == batches-1:
                e = len(self.dids)
            # print s,e
            dids = self.dids[s:e]
            self.execute_batch(dids)
            check_dids += dids
        assert check_dids == self.dids

    def execute_batch(self, dids):
        init_params = {
            "working_dir": self.working_dir,
            "seed": self.seed,
            "total_time_limit": self.total_time_limit,
            "use_default_params": self.use_default_params,
        }

        param_set = []
        for did in dids:
            params = init_params.copy()
            params['did'] = did
            param_set.append(params)

        farg_pairs = [(execute_single, [args]) for args in param_set]
        results = utils.execute_parallel(farg_pairs, returns=True)
        return results


def execute_single(kwargs):
    # log output and err to file
    out_file = settings.pj(kwargs['working_dir'], "out_"+kwargs['did']+".log")
    err_file = settings.pj(kwargs['working_dir'], "err_"+kwargs['did']+".log")
    sys.stdout = open(out_file, 'w')
    sys.stderr = open(err_file, 'w')
    runner = ASKLOpenMLRunner(**kwargs)
    return runner.execute()

    
class ASKLOpenMLRunner(object):

    def __init__(self, working_dir, seed, total_time_limit, did, use_default_params):
        self.working_dir = working_dir
        self.seed = seed
        self.total_time_limit = total_time_limit
        self.use_default_params = use_default_params
        self.did = did

    def execute(self):
        assert self.did in set(dataset_ids.all_dids), "invalid did: %s" % self.did
        results = self.run_openml_dataset(self.did)
        file_path = settings.pj(self.working_dir, "askl_final_results_" + str(self.did) + ".json")
        if not os.path.exists(self.working_dir):
            os.mkdir(self.working_dir)
        with open(file_path, "w") as fh:
            json.dump(results, fh)
        return results

    def run_openml_dataset(self, did):
        try:
            X, y, categorical = utils.get_dataset(did)
            return self.run_dataset(str(did), X, y)
        except Exception as e:
            print e
            return np.nan, []

    def run_dataset(self, dataset_name, X, y):

        num_classes = np.unique(y).shape[0]

        # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, random_state=self.seed)
        # replicate 20k split
        np.random.seed(123)
        X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=0.10)

        askl_tmp = settings.pj(self.working_dir, "internal_%s" % dataset_name, "tmp")
        askl_out = settings.pj(self.working_dir, "internal_%s" % dataset_name, "out")

        default_parameters = {
            "time_left_for_this_task": self.total_time_limit,
            "tmp_folder": askl_tmp,
            "output_folder": askl_out,
            "delete_tmp_folder_after_terminate": False,
            "delete_output_folder_after_terminate": False
        }

        if self.use_default_params:
            metric = 'acc_metric'
            additional_parameters = {}
        else:
            metric = 'auc_metric'
            additional_parameters = {
                "ensemble_size": 1,
                "resampling_strategy": 'holdout',
                "include_preprocessors": ['polynomial', 'pca'],
                "include_estimators": ['lda',
                                     'xgradient_boosting',
                                     'qda',
                                     'extra_trees',
                                     'decision_tree',
                                     'gradient_boosting',
                                     'k_nearest_neighbors',
                                     'multinomial_nb',
                                     'libsvm_svc',
                                     'gaussian_nb',
                                     'random_forest',
                                     'bernoulli_nb'],
                "per_run_time_limit": 30,
                "ml_memory_limit": 1000 * 16,
            }

        parameters = default_parameters.copy()
        parameters.update(additional_parameters)
        automl = autosklearn.classification.AutoSklearnClassifier(**parameters)

        # set metric to auc to match our performance metric
        miro_extra = {
            'did': dataset_name,
            'working_dir': self.working_dir,
            'X_test': X_test,
            'y_test': y_test,
            'num_classes': num_classes,
        }

        print
        print "-"*75
        print "FITTING DATASET %s with PARAMETERS" % dataset_name
        for k, v in parameters.iteritems():
            print k, v
        print "-"*75
        print

        automl.fit(X_train, y_train, dataset_name=dataset_name, metric=metric, miro_extra=miro_extra)

        # run_tests()
        stats = utils.stats_per_iteration(automl)
        test_predictions = automl.predict(X_test)
        test_auc = utils.roc_auc_score(y_test, test_predictions, num_classes=num_classes)

        return test_auc, stats

    def test_dataset(self):
        self.working_dir = "digits_test"
        dataset_name = "digits"
        digits = sklearn.datasets.load_digits()
        X = digits.data
        y = digits.target
        return self.run_dataset(dataset_name, X, y)
