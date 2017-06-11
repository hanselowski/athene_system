import os

try:
    import cPickle as pickle
except:
    import pickle

from aligner import align

from model.utils import get_dataset, get_tokenized_lemmas


def _get_unaligned_tokens(tokens, alignment):
    aligned = [a-1 for (a, _) in alignment]
    unaligned = [i for i in range(len(tokens)) if i not in aligned]
    return [tokens[i] for i in unaligned]


if __name__ == "__main__":
    df = get_dataset()
    data = {}

    for id, row in df.iterrows():
        article_hl_tok = get_tokenized_lemmas(row.articleHeadline)
        claim_hl_tok = get_tokenized_lemmas(row.claimHeadline)
        try:
            alignment = align(claim_hl_tok, article_hl_tok)
            data[(row.claimId, row.articleId)] = [(s-1, t-1) for (s, t) in alignment[0]]
        except:
            print 'Unable to align', article_hl_tok, 'and', claim_hl_tok
            print row.articleId,  row.claimId

    with open(os.path.join('..', 'data', 'pickled', 'aligned-data.pickle'), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

