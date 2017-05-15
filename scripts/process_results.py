import os
import json
from functools import partial
import glob

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
from amle import utils


def main(working_dir):
    print "processing %s" % working_dir
    
    final_files = glob.glob(working_dir + "/askl_final_results_*")
    final_info = map(lambda x: {"did": x.split("/")[-1].split(".")[0].split("_")[-1]}, final_files)
    
    iter_files = glob.glob(working_dir + "/askl_iter_results_*")
    iter_info = map(lambda x: x.split("/")[-1].split(".")[0].split("_"), iter_files)
    iter_info = map(lambda x: {"did": x[-2], "iteration": int(x[-1])}, iter_info)

    dids = list(set(map(lambda x: x['did'], iter_info)))
    print "num_final", len(final_info)
    print "num_dids ", len(dids)

    num_classes = map(utils.get_num_classes, dids)
    num_classes = dict(zip(dids, num_classes))

    print "processing %d iter files" % len(iter_files)
    for i in range(len(iter_files)):
        with open(iter_files[i]) as fh:
            data = json.load(fh)
            # print i, len(iter_info), len(iter_files)
            # import ipdb; ipdb.set_trace()
            iter_info[i]['y_test'] = np.array(data['y_test'])
            iter_info[i]['y_pred'] = np.array(data['y_pred'])
            # iter_num_classes = iter_info[i]['num_classes']
            iter_num_classes = num_classes[iter_info[i]['did']]
            iter_info[i]['iter_auc'] = utils.roc_auc_score(iter_info[i]['y_test'], iter_info[i]['y_pred'], iter_num_classes)

    print "processing %d final files" % len(final_files)
    for i in range(len(final_files)):
        with open(final_files[i]) as fh:
            data = json.load(fh)
            if data[0] is None:
                print final_info[i]['did'], 'has nan final_auc'
            final_info[i]['final_auc'] = data[0]

    results = {did: {'final_auc': None, 'iter_auc': []} for did in dids}

    for item in final_info:
        results[item['did']]['final_auc'] = item['final_auc']

    for item in sorted(iter_info, key=lambda x: (x['did'], x['iteration'])):
        results[item['did']]['iter_auc'].append((item['iteration'], item['iter_auc']))

    for did, item in results.iteritems():
        item['iter_num'], item['iter_auc'] = map(list, zip(*sorted(item['iter_auc'], key=lambda x: x[0])))

    print "\niterations per dataset"
    for did, item in sorted(results.items(), key=lambda x: int(x[0])):
        num_iters =  len(item['iter_auc'])
        fauc = item['final_auc']
        fauc = None if fauc is None else ("%.3f" % fauc)
        if num_iters < 200:
            print did, fauc, num_iters, "LOW"
        else:
            print did, fauc, num_iters

    with open("paper_output.json", "w") as fh:
        json.dump(results, fh, indent=4)

    print "\nscore comparison"
    for did, item in sorted(results.items(), key=lambda x: ((x[1]['final_auc'] or 0) - max(x[1]['iter_auc']))):
        fauc = item['final_auc']
        fauc = None if fauc is None else ("%.3f" % fauc)
        print did, fauc, "%.3f" % max(item['iter_auc'])


if __name__ == '__main__':
    import faulthandler
    import argparse
    faulthandler.enable()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('working_dir', default=settings.default_working_dir, help="Working directory.")
    args = parser.parse_args()
    # print "\n", "ARGS", args, "\n"
    main(**vars(args))
