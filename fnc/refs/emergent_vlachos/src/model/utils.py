import os
import itertools as it
import functools as ft
import functools
import operator as op

try:
    from functools import lru_cache
except:
    from repoze.lru import lru_cache

try:
    import cPickle as pickle
except:
    import pickle

import numpy as np
import csv
import gensim
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer
from sklearn.cross_validation import cross_val_score

from model.cross_validation import ClaimKFold

VALID_STANCE_LABELS = ['for', 'against', 'observing']

_data_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


def flatten(L):
    return it.chain.from_iterable(L)


def calc_confusion_matrix(actual, predicted=VALID_STANCE_LABELS, labels=VALID_STANCE_LABELS):
    actual_lst = actual.tolist()
    predicted_lst = predicted.tolist()
    cm = pd.DataFrame(index=labels, columns=labels, data=0.0)
    for i in range(len(actual)):
        cm.ix[actual_lst[i], predicted_lst[i]] += 1
    return cm


def get_contraction_mappings():
    folder = os.path.join(_data_folder, 'misc')
    with open(os.path.join(folder, 'contraction_map.csv')) as f:
        return dict(filter(None, csv.reader(f)))


@lru_cache(maxsize=1)
def get_dataset(filename='url-versions-2015-06-14-clean.csv'):
    folder = os.path.join(_data_folder, 'emergent')
    return pd.DataFrame.from_csv(os.path.join(folder, filename))


def split_data(data):
    y = data.articleHeadlineStance
    X = data[['claimHeadline', 'articleHeadline', 'claimId', 'articleId']]
    return X, y


@lru_cache(maxsize=1000)
def get_antonyms(w):
    return set([y.name().lower() for y in flatten([x.antonyms() for x in wn.lemmas(w.lower())])])


@lru_cache(maxsize=1000)
def is_antonym(v, w):
    return w in get_antonyms(v) or v in get_antonyms(w)


@lru_cache(maxsize=1000)
def get_synonyms(w):
    return set(map(lambda x: x.lower(), flatten([ss.lemma_names() for ss in wn.synsets(w.lower())])))


@lru_cache(maxsize=1000)
def is_synonym(v, w):
    return w in get_synonyms(v) or v in get_synonyms(w)


def get_wordnet_list(func, seed):
    return list(set(ft.reduce(set.union,
                              map(func, seed)).union(seed)))


_wnl = nltk.WordNetLemmatizer()


def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]


@lru_cache(maxsize=1000)
def get_entailments(w):
    entailments = flatten([ss.entailments() for ss in wn.synsets(w)])
    entailment_lemmas = flatten([e.lemmas() for e in entailments])
    return set([l.name() for l in entailment_lemmas])


@lru_cache(maxsize=1000)
def entails(v, w):
    return w in get_entailments(v)


def generate_test_training_set(data, test_set_fraction=0.2):
    """
    Splits the given data into mutually exclusive test
    and training sets using the claim ids.
    :param data: DataFrame containing the data
    :param test_set_fraction: percentage of data to reserve for test
    :return: a tuple of DataFrames containing the test and training data
    """
    claim_ids = np.array(list(set(data.claimId.values)))
    claim_ids_rand = np.random.permutation(claim_ids)
    claim_ids_test = claim_ids_rand[:len(claim_ids_rand) * test_set_fraction]
    claim_ids_train = set(claim_ids_rand).difference(claim_ids_test)
    test_data = data[data.claimId.isin(claim_ids_test)]
    train_data = data[data.claimId.isin(claim_ids_train)]
    return test_data, train_data


_brown_cluster_data_files_by_size = {
    100: 'brown-rcv1.clean.tokenized-CoNLL03.txt-c100-freq1.txt',
    320: 'brown-rcv1.clean.tokenized-CoNLL03.txt-c320-freq1.txt',
    1000: 'brown-rcv1.clean.tokenized-CoNLL03.txt-c1000-freq1.txt',
    3200: 'brown-rcv1.clean.tokenized-CoNLL03.txt-c3200-freq1.txt'
}

_MAX_CLUSTER_SIZE = 20

@lru_cache(maxsize=1000)
def get_brown_cluster_data(num_classes):
    brown_clusters = _brown_cluster_data_files_by_size.keys()
    if num_classes not in brown_clusters:
        raise ValueError('Brown Cluster class size must be one of: {0:s}'.format(str(brown_clusters)))

    filename = _brown_cluster_data_files_by_size[num_classes]
    data = {}
    folder = os.path.join(_data_folder, 'brown-clusters')
    with open(os.path.join(folder, filename)) as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for name, word, _ in reader:
            data[normalize_word(word)] = int(name, 2)
    values = set(data.values())
    pairs = enumerate(it.product(values, values))
    return data, {p: n for (n, p) in pairs}


_pickled_data_folder = os.path.join(_data_folder, 'pickled')

@lru_cache(maxsize=1)
def get_aligned_data():
    with open(os.path.join(_pickled_data_folder, 'aligned-data.pickle'), 'rb') as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def get_hungarian_alignment_score_data():
    with open(os.path.join(_pickled_data_folder, 'hungarian-alignment-score.pickle'), 'rb') as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def get_w2vec_data():
    with open(os.path.join(_pickled_data_folder, 'w2vec-data.pickle'), 'rb') as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def get_stanparse_data():
    with open(os.path.join(_pickled_data_folder, 'stanparse-data.pickle'), 'rb') as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def get_stanparse_depths():
    with open(os.path.join(_pickled_data_folder, 'stanparse-depths.pickle'), 'rb') as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def get_cosine_similarity_data():
    with open(os.path.join(_pickled_data_folder, 'cosine-similarity.pickle'), 'rb') as f:
        return pickle.load(f)


W2VEC_SIZE = 300
_UNIT_ADD = np.zeros(W2VEC_SIZE)
_OP_ADD = op.add


def convert_text_to_vec(model, text, grp=(_UNIT_ADD, _OP_ADD)):
    unit, oper = grp

    if not text:
        return unit

    m = []
    for token in get_tokenized_lemmas(text):
        try:
            vec = model[token]
        except:
            vec = unit
        m.append(vec)
    return functools.reduce(oper, m) if m else unit


@lru_cache(maxsize=1)
def get_w2v_model():
    folder = os.path.join(_data_folder, 'google')
    return gensim.models.Word2Vec.load_word2vec_format(os.path.join(folder, "GoogleNews-vectors-negative300.bin"),
                                                       binary=True)

@lru_cache(maxsize=100000)
def get_dep_graph(token_pos, id):
    stanparse_data = get_stanparse_data()
    dep_parse = get_dep_graph(token_pos, stanparse_data[id])
    return _get_dep_graph(token_pos, dep_parse)


def _get_dep_graph(token_pos, dep_parse):
    tp = token_pos
    sentence = dep_parse['sentences'][0]

    for s in dep_parse['sentences']:
        sentence_len = len(s['words'])
        if tp - sentence_len < 0:
            sentence = s
            break
        else:
            tp -= sentence_len

    deps = sentence['dependencies']
    dep_subgraph = [(i, j, k) for (i, j, k) in deps if int(j.split('-')[1])-1 == tp]
    return dep_subgraph


def cosine_sim(u, v):
    """Returns the cosine similarity between two 1-D vectors, u and v"""
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


@lru_cache(maxsize=1)
def get_ppdb_data():
    with open(os.path.join(_pickled_data_folder, 'ppdb.pickle'), 'rb') as f:
        return pickle.load(f)


_stemmer = PorterStemmer()


@lru_cache(maxsize=100000)
def get_stem(w):
    return _stemmer.stem(w)


_max_ppdb_score = 10.0
_min_ppdb_score = -_max_ppdb_score


@lru_cache(maxsize=100000)
def compute_paraphrase_score(s, t):
    """Return numerical estimate of whether t is a paraphrase of s, up to
    stemming of s and t."""
    s_stem = get_stem(s)
    t_stem = get_stem(t)

    if s_stem == t_stem:
        return _max_ppdb_score

    # get PPDB paraphrases of s, and find matches to t, up to stemming
    s_paraphrases = set(get_ppdb_data().get(s, [])).union(get_ppdb_data().get(s_stem, []))
    matches = filter(lambda (a, b, c): a == t or a == t_stem, s_paraphrases)
    if matches:
        return max(matches, key=lambda (x, y, z): y)[1]
    return _min_ppdb_score


_stanparse_data = get_stanparse_data()


def get_stanford_idx(x):
    i = x.rfind('-')
    return x[:i].lower(), int(x[(i+1):])


def _is_not(w):
    return w.lower() == 'not' or w.lower() == "n't"


def find_negated_word_idxs(id):
    neg_word_idxs = []
    try:
        for sentence in _stanparse_data[id]['sentences']:
            for dependency in sentence['dependencies']:
                rel, head, dependent = dependency
                d, d_idx = get_stanford_idx(dependent)
                h, h_idx = get_stanford_idx(head)

                if rel == 'neg' \
                        or (rel == 'nn' and _is_not(d)):
                    neg_word_idxs.append(h_idx-1)

                if rel == 'pcomp' and _is_not(h):
                    neg_word_idxs.append(d_idx-1)
    except KeyError:
        pass
    return neg_word_idxs


def calc_measures(cm):
    df = pd.DataFrame(index=VALID_STANCE_LABELS, columns=['accuracy',
                                                          'precision',
                                                          'recall',
                                                          'F1'])
    for c in VALID_STANCE_LABELS:
        tp = float(cm.ix[c, c])
        fp = sum(cm[c]) - tp
        fn = sum(cm.ix[c, :]) - tp
        tn = sum(sum(cm.values)) - tp - fp - fn
        df.ix[c, 'accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        pr = df.ix[c, 'precision'] = tp / (tp + fp)
        rc = df.ix[c, 'recall'] = tp / (tp + fn)
        df.ix[c, 'F1'] = (2 * pr * rc) / (pr + rc)
    return df


class Score(object):

    def __init__(self, cm, accuracy, measures):
        self.cm = cm
        self.accuracy = accuracy
        self.measures = measures

    def __str__(self):
        s = 'Confusion matrix:\n'
        s += '=================\n'
        s += '{0:s}\n'.format(str(self.cm))
        s += '\nMeasures:\n'
        s += '=========\n'
        s += 'accuracy: {0:.4f}\n\n'.format(self.accuracy)
        s += 'Per class:\n'
        s += str(self.measures)
        return s


class RunCV(object):

    def __init__(self, X, y, predictor, display=False):
        self.X = X
        self.y = y
        self.predictor = predictor
        self.display = display
        self.fold = 1

    def run_cv(self, n_folds=10):
        if self.display:
            print('Running {0:d}-fold cross-validation'.format(n_folds))

        measures = []
        cms = []

        def scorer(estimator, XX, yy):
            if self.display:
                print('\n>> Fold: {0} <<'.format(self.fold))
            score = estimator.score(XX, yy)
            if self.display:
                print(str(score))
            self.fold += 1
            measures.append(score.measures)
            cms.append(score.cm)
            return score.accuracy

        skf = ClaimKFold(self.X, n_folds)
        scr = cross_val_score(self.predictor, self.X, self.y.values, cv=skf, scoring=scorer)

        if self.display:
            print('\n>> Averages across all folds <<')
        av_measures = (sum([m.unstack() for m in measures]).unstack() / len(measures)).T
        av_cms = (sum([m.unstack() for m in cms]).unstack() / len(cms)).T
        score = Score(av_cms, np.mean(scr), av_measures)
        if self.display:
            print(score)
            print('')
        return score


def run_test(X, y, test_data, predictor, display=False):
    print '>> Training classifier <<\n'

    predictor.fit(X, y)
    if display:
        print('>> Classifying test data <<\n')
    test_data_copy = test_data.copy()
    y_test = test_data_copy.articleHeadlineStance.values
    score = predictor.score(test_data_copy, y_test)

    if display:
        print(str(score))
    return score


_svo_labels = set(['nsubj', 'dobj'])


def get_svo(grph, grph_labels, node=0, svo=None):
    if svo is None:
        svo = []
    svo_match = dict([(l, (x, y)) for ((x, y), l) in grph_labels.items() \
                      if x == node and l in _svo_labels])

    def valid_match(m):
        if len(m) != 2:
            return False
        if set(m.keys()).intersection(_svo_labels) != _svo_labels:
            return False
        return True

    if valid_match(svo_match):
        svo.append(svo_match)
    for child in grph.get(node, []):
        get_svo(grph, grph_labels, child, svo)
    return svo


@lru_cache(maxsize=100000)
def get_svo_triples(id):
    stanparse_depths = get_stanparse_depths()
    d = []
    for i, x in stanparse_depths[id].items():
        grph, grph_labels, _ = x
        d.extend([(i, s) for s in get_svo(grph, grph_labels)])
    return d












