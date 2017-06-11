import os
try:
    import cPickle as pickle
except:
    import pickle

import numpy as np
import pandas as pd

from munkres import Munkres, make_cost_matrix

from model.utils import get_tokenized_lemmas, compute_paraphrase_score, _max_ppdb_score, get_dataset


_munk = Munkres()


def calc_hungarian_alignment_score(s, t):
    """Calculate the alignment score between the two texts s and t
    using the implementation of the Hungarian alignment algorithm
    provided in https://pypi.python.org/pypi/munkres/."""
    s_toks = get_tokenized_lemmas(s)
    t_toks = get_tokenized_lemmas(t)

    df = pd.DataFrame(index=s_toks, columns=t_toks, data=0.)

    for c in s_toks:
        for a in t_toks:
            df.ix[c, a] = compute_paraphrase_score(c, a)

    matrix = df.values
    cost_matrix = make_cost_matrix(matrix, lambda cost: _max_ppdb_score - cost)

    indexes = _munk.compute(cost_matrix)
    total = 0.0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
    return indexes, total / float(np.min(matrix.shape))


if __name__ == "__main__":
    df = get_dataset()
    data = {}

    for _, row in df.iterrows():
        data[(row.claimId, row.articleId)] = calc_hungarian_alignment_score(row.claimHeadline,
                                                                            row.articleHeadline)

    with open(os.path.join('..', 'data', 'pickled', 'hungarian-alignment-score.pickle'), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)