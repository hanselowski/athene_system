import os

try:
    import cPickle as pickle
except:
    import pickle

import numpy as np


def to_float(s):
    return np.nan if s == 'NA' else float(s)


def process_ppdb_data():
    with open(os.path.join('..', 'data', 'ppdb', 'ppdb-2.0-priliminary-release.tar'), 'r') as f:
        ppdb = {}
        for line in f:
            data = line.split('|||')
            text_lhs = data[1].strip(' ,')
            text_rhs = data[2].strip(' ,')
            if len(text_lhs.split(' ')) > 1 or len(text_rhs.split(' ')) > 1:
                continue
            ppdb_score = to_float(data[3].strip().split()[0].split('=')[1])
            entailment = data[-1].strip()
            paraphrases = ppdb.setdefault(text_lhs, list())
            paraphrases.append((text_rhs, ppdb_score, entailment))
    return ppdb


if __name__ == '__main__':
    ppdb = process_ppdb_data()

    with open(os.path.join('..', 'data', 'pickled', 'ppdb.pickle', 'wb')) as f:
        pickle.dump(ppdb, f, pickle.HIGHEST_PROTOCOL)