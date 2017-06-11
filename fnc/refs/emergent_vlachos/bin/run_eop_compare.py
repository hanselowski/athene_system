import os

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from model.utils import calc_confusion_matrix, calc_measures


def _load_eop_results_file(f):
    df_eop = pd.read_csv(f, delimiter='\t', header=None)
    df_eop.columns = ['id', 'benchmark', 'predicted', 'confidence']

    df_eop.loc[df_eop.benchmark == 'ENTAILMENT', 'benchmark'] = 'for'
    df_eop.loc[df_eop.benchmark == 'CONTRADICTION', 'benchmark'] = 'against'
    df_eop.loc[df_eop.benchmark == 'UNKNOWN', 'benchmark'] = 'observing'

    df_eop.loc[df_eop.predicted == 'Entailment', 'predicted'] = 'for'
    df_eop.loc[df_eop.predicted == 'NonEntailment', 'predicted'] = 'against'
    df_eop.loc[df_eop.predicted == 'Unknown', 'predicted'] = 'observing'
    return df_eop


_no_folds = 10


def _set_threshold(df, t):
    df_t = df.copy()
    df_t['thresholded_predicted'] = df_t.predicted
    if t is not None:
        df_t.loc[df_t.confidence < t, 'thresholded_predicted'] = 'observing'
    return df_t


def _calc_accuracy(df):
    y = df.benchmark
    y_hat = df.thresholded_predicted
    return accuracy_score(y, y_hat)


def _cv_threshold(folds, thresholds):
    results = np.zeros((thresholds.size, _no_folds))
    for i, t in enumerate(thresholds):
        for j in range(_no_folds):
            df_eop = _set_threshold(folds[j+1], t)
            results[i, j] = _calc_accuracy(df_eop)
    return np.mean(results, axis=1)


_eop_results_filename = 'MaxEntClassificationEDA_Base+WN+VO+TP+TPPos_EN.xml_results.txt'


if __name__ == '__main__':
    d = {}
    for i in range(1, _no_folds + 1):
        f = os.path.join('..', 'output', 'eop', 'fold-{0:d}'.format(i), _eop_results_filename)
        d[i] = _load_eop_results_file(f)

    thresholds = np.arange(0.51, 1.0, 0.001)
    accuracies = _cv_threshold(d, thresholds)
    idx = np.argmax(accuracies)
    best_threshold = thresholds[idx]
    print 'Best threshold chosen by 10-fold CV:', best_threshold

    f_rte_clean_test = os.path.join('..', 'output', 'eop', 'rte-clean-test', _eop_results_filename)
    rte_clean_test = _set_threshold(_load_eop_results_file(f_rte_clean_test), best_threshold)
    print 'RTE3 accuracy:', _calc_accuracy(rte_clean_test)

    f_emergent_clean_test_fa = os.path.join('..', 'output', 'eop', 'rte-clean-test-fa', _eop_results_filename)
    emergent_clean_test = _set_threshold(_load_eop_results_file(f_emergent_clean_test_fa), None)
    print 'RTE3 accuracy for-against only:', _calc_accuracy(emergent_clean_test)

    f_emergent_clean_test = os.path.join('..', 'output', 'eop', 'emergent-clean-test', _eop_results_filename)
    emergent_clean_test = _set_threshold(_load_eop_results_file(f_emergent_clean_test), None)
    print 'Emergent accuracy:', _calc_accuracy(emergent_clean_test)


