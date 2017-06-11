import os
import operator as op
try:
    import cPickle as pickle
except:
    import pickle

import numpy as np

from model.utils import get_dataset, get_w2v_model, convert_text_to_vec, W2VEC_SIZE

if __name__ == "__main__":
    df = get_dataset()
    data = ({}, {})
    add_data, mult_data = data
    model = get_w2v_model()

    for id, row in df.iterrows():
        add_data[row.claimId] = convert_text_to_vec(model, row.claimHeadline)
        add_data[row.articleId] = convert_text_to_vec(model, row.articleHeadline)

        grp_mult = (np.ones(W2VEC_SIZE), op.mul)
        mult_data[row.claimId] = convert_text_to_vec(model, row.claimHeadline, grp_mult)
        mult_data[row.articleId] = convert_text_to_vec(model, row.articleHeadline, grp_mult)

    with open(os.path.join('..', 'data', 'pickled', 'w2vec-data.pickle'), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
