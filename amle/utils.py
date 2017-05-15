import os
import json
import time
from functools import partial
from operator import itemgetter

import numpy as np

import sklearn
import sklearn.cross_validation
import sklearn.datasets
import sklearn.metrics

from openml import datasets

import autosklearn.classification
from autosklearn.constants import *


#########################
# OpenML UTILS
#########################


def get_dataset(did):
    dataset = datasets.get_dataset(did)
    X, y, categorical = dataset.get_data(target=dataset.default_target_attribute,
                                         return_categorical_indicator=True)
    return X, y, categorical


def get_num_classes(did):
    X, y, categorical = get_dataset(did)
    num_classes = np.unique(y).shape[0]
    return num_classes


#########################
# OUR AUC METRIC
#########################


def to_matrix(a, num_classes=None):
    if num_classes is None:
        # infer
        num_classes = np.max(a) - np.min(a) + 1
    r = np.zeros(shape=(a.shape[0], num_classes))
    r[np.arange(a.shape[0]), a] = 1
    return r


def roc_auc_score(y_truth, y_pred, num_classes=None):
    return sklearn.metrics.roc_auc_score(*map(partial(to_matrix, num_classes=num_classes), [y_truth, y_pred]))


#########################
# AUTOSKLEARN UTILS
#########################


def run_tests(automl, num_classes, X_train, y_train, X_test, y_test):
    # import ipdb; ipdb.set_trace()
    print(model_stats(automl))

    print "num_classes", num_classes

    train_predictions = automl.predict(X_train)
    train_auc = roc_auc_score(y_train, train_predictions, num_classes=num_classes)
    print("train data auc score", train_auc)

    test_predictions = automl.predict(X_test)
    test_auc = roc_auc_score(y_test, test_predictions, num_classes=num_classes)
    print("test data auc score", test_auc)

    # they expect a one-hot encoding
    try:
        test_predictions = to_matrix(test_predictions, num_classes=num_classes)
        if num_classes == 2:
            their_auc = classification_metrics.auc_metric(y_test, test_predictions, task=BINARY_CLASSIFICATION)
        else:
            their_auc = classification_metrics.auc_metric(y_test, test_predictions, task=MULTICLASS_CLASSIFICATION)

        # they compute Gini index
        # 2*AUC-1
        # e.g. 2*0.8-1 = 0.6
        # verified for binary and multiclass datasets.
        print("their test data auc score (2*auc-1)", their_auc)
        print("their test data auc score (reverted from Gini index)", (their_auc + 1) / 2)
    except Exception as e:
        print e


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


#########################
# GENERAL UTILS
#########################


def execute_parallel(farg_pairs, returns=False):
    # see https://blog.ionelmc.ro/2014/12/21/compiling-python-extensions-on-windows/
    # from multiprocess import Process, Queue
    from multiprocessing import Process, Queue

    processes = []
    q = None
    results = None
    if returns:
        results = []
        q = Queue()

    for i, farg_pair in enumerate(farg_pairs):
        if returns:
            def target_func(*args, **kwargs):
                q.put((i, farg_pair[0](*args, **kwargs)))
        else:
            target_func = farg_pair[0]

        if len(farg_pair) > 1:
            p = Process(target=target_func, args=farg_pair[1])
        else:
            p = Process(target=target_func)
        p.start()
        processes.append(p)

    if returns:
        while len(results) < len(farg_pairs):
            results.append(q.get())
            time.sleep(0.01)

        assert len(results) == len(farg_pairs)

    # join all processes before exiting
    for i, p in enumerate(processes):
        p.join()

    if returns:
        # print "all processes executed"
        results = zip(*sorted(results, key=lambda x: x[0]))[1]
        return results
