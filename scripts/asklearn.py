import os
import json
from functools import partial

from operator import itemgetter

import numpy as np

import sklearn
import sklearn.cross_validation
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification
from autosklearn.constants import *

from openml import datasets

from amle.data import dataset_ids
from amle import settings

# use this for stats
def model_stats(model):
    automl = model._automl._automl
    cv_results = automl.cv_results_
    print('name %s' % automl._dataset_name)
    print('metric %s' % METRIC_TO_STRING[automl._metric])
    top_val_score = cv_results['mean_test_score'][np.argmax(cv_results['mean_test_score'])]
    print('top_val_score %f' % top_val_score)
    print('num_runs %d' % len(cv_results['status']))
    print('num_success %d' % sum([s == 'Success' for s in cv_results['status']]))
    print('num_crash %d' % sum([s == 'Crash' for s in cv_results['status']]))
    print('num_timeout %d' % sum([s == 'Timeout' for s in cv_results['status']]))
    print('num_memout %d' % sum([s == 'Memout' for s in cv_results['status']]))


def stats_per_iteration(model):
    automl = model._automl._automl
    runhistory = automl._proc_smac.runhistory
    res = []
    for key, val in runhistory.data.iteritems():
        # val.cost = 1 - auc
        auc = 1 - val.cost
        # val.time = duration in seconds of this run
        print "AUC =", auc, "SECS =", val.time
        res.append((auc, val.time))
    return res


def to_matrix(a, num_classes=None):
    if num_classes is None:
        # infer
        num_classes = np.max(a) - np.min(a) + 1
    r = np.zeros(shape=(a.shape[0], num_classes))
    r[np.arange(a.shape[0]), a] = 1
    return r


def roc_auc_score(y_truth, y_pred, num_classes=None):
    return sklearn.metrics.roc_auc_score(*map(partial(to_matrix, num_classes=num_classes), [y_truth, y_pred]))


def run_dataset(dataset_name, X, y, seed, total_time_limit):
    num_classes = np.unique(y).shape[0]
    X_train, X_test, y_train, y_test = \
        sklearn.cross_validation.train_test_split(X, y, random_state=seed)

    automl = autosklearn.classification.AutoSklearnClassifier(
        # important parameters
        ensemble_size=1,
        resampling_strategy='holdout',
        include_preprocessors=['polynomial', 'pca'],
        include_estimators=['lda',
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

        per_run_time_limit=30,
        ml_memory_limit=1000*16,

        # default parameters
        time_left_for_this_task=total_time_limit,
        tmp_folder='/tmp/autoslearn_holdout_example_tmp',
        output_folder='/tmp/autosklearn_holdout_example_out',
        delete_tmp_folder_after_terminate=True,
        delete_output_folder_after_terminate=True,
    )

    # set metric to auc to match our performance metric
    automl.fit(X_train, y_train, dataset_name=dataset_name, metric="auc_metric")
    
    # import ipdb; ipdb.set_trace()
    print(model_stats(automl))
    stats = stats_per_iteration(automl)

    print "num_classes", num_classes
    
    train_predictions = automl.predict(X_train)
    train_auc = roc_auc_score(y_train, train_predictions, num_classes=num_classes)
    print("train data auc score", train_auc)
    from autosklearn.metrics import classification_metrics
    
    test_predictions = automl.predict(X_test)
    test_auc = roc_auc_score(y_test, test_predictions, num_classes=num_classes)
    print("test data auc score", test_auc)
    
    # they expect a one-hot encoding
    test_predictions = to_matrix(test_predictions)
    if num_classes == 2:
        their_auc = classification_metrics.auc_metric(y_test, test_predictions, task=BINARY_CLASSIFICATION)
    else:
        their_auc = classification_metrics.auc_metric(y_test, test_predictions, task=MULTICLASS_CLASSIFICATION)
    
    # they compute Gini index
    # 2*AUC-1
    # e.g. 2*0.8-1 = 0.6
    # verified for binary and multiclass datasets.
    print("their test data auc score (2*auc-1)", their_auc)
    print("their test data auc score (reverted from Gini index)", (their_auc+1)/2)

    return test_auc, stats


def run_openml_dataset(did, seed, total_time_limit):
    try:
        # import ipdb;ipdb.set_trace()
        dataset = datasets.get_dataset(did)
        X, y, categorical = dataset.get_data(target=dataset.default_target_attribute,
                                             return_categorical_indicator=True)
        return run_dataset(str(did), X, y, seed, total_time_limit)
    except Exception as e:
        print e
        return np.nan, []


def test_dataset():
    dataset_name = "digits"
    digits = sklearn.datasets.load_digits()
    X = digits.data
    y = digits.target
    return run_dataset(dataset_name, X, y, seed=1, total_time_limit=120)


def main(working_dir, seed, total_time_limit, dids):
    if dids is None:
        test_dataset_ids = dataset_ids.test_dids
    else:
        test_dataset_ids = dids

    assert len(set(test_dataset_ids) - set(dataset_ids.all_dids)) == 0, "invalid dids: %s" % " ".join(test_dataset_ids)
    
    aucs_iterstats = map(partial(run_openml_dataset, seed=seed, total_time_limit=total_time_limit), test_dataset_ids)

    results = zip(test_dataset_ids, aucs_iterstats)
    print results
    
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
    file_path = settings.pj(working_dir, "askl_results.json")
    if len(dids) == 1:
        file_path = settings.pj(working_dir, dids[0]+"_results.json")
    with open(file_path, "w") as fh:
        json.dump(results, fh)

if __name__ == '__main__':
    import faulthandler
    import argparse
    faulthandler.enable()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', '-d', default=settings.default_working_dir, help="Working directory.")
    parser.add_argument('--seed', '-s', default=1, type=int, help="Random seed.")
    parser.add_argument('--total_time_limit', '-t', type=int, default=120,
                        help="Total time (secs) given to search hyperparameters for a given dataset.")
    parser.add_argument('--dids', default=None, nargs='*', type=str,
                        help="OpenML dataset IDs to use (all test dataset ids are used by default).")
    args = parser.parse_args()
    # print "\n", "ARGS", args, "\n"
    main(**vars(args))
