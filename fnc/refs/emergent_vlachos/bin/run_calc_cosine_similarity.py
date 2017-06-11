import os
import operator as op
try:
    import cPickle as pickle
except:
    import pickle

import numpy as np

from model.utils import get_w2vec_data, get_dataset, cosine_sim

if __name__ == "__main__":
    w2v_add, w2v_mult = get_w2vec_data()
    df = get_dataset()
    data = {}, {}
    cos_add, cos_mult = data

    def cosn(v, w):
        c = cosine_sim(v, w)
        if np.isnan(c) or np.isinf(c):
            c = 0.0
        return c

    for _, row in df.iterrows():
        cos_add[(row.claimId, row.articleId)] = cosn(w2v_add[row.claimId], w2v_add[row.articleId])
        cos_mult[(row.claimId, row.articleId)] = cosn(w2v_mult[row.claimId], w2v_mult[row.articleId])

    with open(os.path.join('..', 'data', 'pickled', 'cosine-similarity.pickle'), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
