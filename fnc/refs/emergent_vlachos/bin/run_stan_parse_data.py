import os
try:
    import cPickle as pickle
except:
    import pickle


from model.utils import get_dataset
from model.stanford import nlp


if __name__ == "__main__":
    df = get_dataset()
    data = {}
    for id, row in df.iterrows():
        try:
            data[row.claimId] = nlp.parse(row.claimHeadline)
            data[row.articleId] = nlp.parse(row.articleHeadline)
        except:
            print "Can't parse the following"
            print "Claim: " + row.claimHeadline
            print "Article: " + row.articleHeadline

    with open(os.path.join('..', 'data', 'pickled', 'stanparse-data.pickle'), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
