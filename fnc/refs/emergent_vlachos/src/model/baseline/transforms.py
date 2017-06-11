import itertools as it

import numpy as np
from scipy.sparse import dok_matrix
from nltk.util import ngrams
from nltk.corpus import stopwords
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_extraction.text import CountVectorizer

from model.base import StatelessTransform
from model.utils import get_tokenized_lemmas, get_stanparse_data, \
    get_brown_cluster_data, get_aligned_data, get_stem


class BoWTransform(StatelessTransform):

    def __init__(self, ngram_upper_range=2, max_features=500):
        self.cv = None
        self.ngram_upper_range = ngram_upper_range
        self.max_features = max_features

    def fit(self, X, y=None):
        text = X.articleHeadline.values
        self.cv = CountVectorizer(ngram_range=(1, self.ngram_upper_range),
                                  max_features=self.max_features)
        self.cv.fit_transform(text)
        return self

    def transform(self, X, y=None):
        text = X.articleHeadline.values
        return self.cv.transform(text)


_refuting_seed_words = [
                        'fake',
                        'fraud',
                        'hoax',
                        'false',
                        'deny', 'denies',
                        # 'refute',
                        'not',
                        'despite',
                        'nope',
                        'doubt', 'doubts',
                        'bogus',
                        'debunk',
                        'pranks',
                        'retract'
]

_refuting_words = _refuting_seed_words


class RefutingWordsTransform(StatelessTransform):

    def transform(self, X):
        mat = np.zeros((len(X), len(_refuting_words)))
        for i, (_, s) in enumerate(X.iterrows()):
            # article_headline = [get_stem(w) for w in get_tokenized_lemmas(s.articleHeadline)]
            article_headline = get_tokenized_lemmas(s.articleHeadline)
            # mat[i, :] = np.array([1 if get_stem(w) in article_headline else 0 for w in _refuting_words])
            mat[i, :] = np.array([1 if w in article_headline else 0 for w in _refuting_words])
        return mat


class NegationOfRefutingWordsTransform(StatelessTransform):

    def transform(self, X):
        mat = np.zeros((len(X), 1))
        stanparse_data = get_stanparse_data()

        def strip_idx(x):
            return x[:x.rfind('-')]

        for i, (_, s) in enumerate(X.iterrows()):
            try:
                for sentence in stanparse_data[s.articleId]['sentences']:
                    for dependency in sentence['dependencies']:
                        rel, head, dependent = dependency
                        if (rel == 'neg' or (rel == 'nn' and strip_idx(dependent).lower() == 'not')) \
                                and strip_idx(head).lower() in _refuting_words:
                            mat[i, 0] = 1
            except KeyError:
                pass
        return mat


class InteractionTransform(StatelessTransform):

    def __init__(self):
        self.refuting_words_transform = RefutingWordsTransform()
        self.question_mark_transform = QuestionMarkTransform()
        self.hedging_words_transform = HedgingWordsTransform()
        self.polynomial = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

    def transform(self, X):
        X1 = self.refuting_words_transform.transform(X)
        X2 = self.question_mark_transform.transform(X)
        X3 = self.hedging_words_transform.transform(X)

        return self.polynomial.fit_transform(np.hstack((X1, X2, X3)))

    def __str__(self):
        return 'I'


class PolarityTransform(StatelessTransform):

    @staticmethod
    def _calc_polarity(s):
        tokens = get_tokenized_lemmas(s)
        return sum([t in _refuting_words for t in tokens]) % 2

    def transform(self, X):
        mat = np.zeros((len(X), 2))
        for i, (_, s) in enumerate(X.iterrows()):
            mat[i, 0] = self._calc_polarity(s.claimHeadline)
            mat[i, 1] = self._calc_polarity(s.articleHeadline)
        return mat


class QuestionMarkTransform(StatelessTransform):

    def transform(self, X):
        mat = np.zeros((len(X), 1))
        for i, (_, s) in enumerate(X.iterrows()):
            if '?' in get_tokenized_lemmas(s.articleHeadline):
                mat[i, 0] = 1
        return mat

_hedging_seed_words = \
    [
        'alleged', 'allegedly',
        'apparently',
        'appear', 'appears',
        'claim', 'claims',
        'could',
        'evidently',
        'largely',
        'likely',
        'mainly',
        'may', 'maybe', 'might',
        'mostly',
        'perhaps',
        'presumably',
        'probably',
        'purported', 'purportedly',
        'reported', 'reportedly',
        'rumor', 'rumour', 'rumors', 'rumours', 'rumored', 'rumoured',
        'says',
        'seem',
        'somewhat',
        # 'supposedly',
        'unconfirmed']

_hedging_words = _hedging_seed_words


class HedgingWordsTransform(StatelessTransform):

    def transform(self, X):
        mat = np.zeros((len(X), len(_hedging_words)))
        for i, (_, s) in enumerate(X.iterrows()):
            # article_headline = [get_stem(w) for w in get_tokenized_lemmas(s.articleHeadline)]
            article_headline = get_tokenized_lemmas(s.articleHeadline)
            # mat[i, :] = np.array([1 if get_stem(w) in article_headline else 0 for w in _hedging_words])
            mat[i, :] = np.array([1 if w in article_headline else 0 for w in _hedging_words])
        return mat


class WordOverlapTransform(StatelessTransform):

    def transform(self, X):
        mat = np.zeros((len(X), 1))
        for i, (_, s) in enumerate(X.iterrows()):
            article_headline = get_tokenized_lemmas(s.articleHeadline)
            claim_headline = get_tokenized_lemmas(s.claimHeadline)
            mat[i, 0] = len(set(article_headline).intersection(claim_headline)) / \
                        float(len(set(article_headline).union(claim_headline)))
        return mat


class BrownClusterPairTransform(StatelessTransform):

    def __init__(self, cluster_size=100):
        self.cluster_size = cluster_size

    def transform(self, X):
        bc_data, bc_data_idx = get_brown_cluster_data(self.cluster_size)
        mat = dok_matrix((len(X), len(bc_data_idx.values())), dtype=np.float32)
        for i, (_, s) in enumerate(X.iterrows()):
            claim_headline = get_tokenized_lemmas(s.claimHeadline)
            article_headline = get_tokenized_lemmas(s.articleHeadline)
            word_pairs = it.product(article_headline, claim_headline)

            for v, w in word_pairs:
                v_cluster = bc_data.get(v)
                w_cluster = bc_data.get(w)
                if v_cluster is None or w_cluster is None:
                    continue

                idx = bc_data_idx[(v_cluster, w_cluster)]
                mat[i, idx] = 1
        return mat


def _get_bigram_clusters(s, bc_data):
    clusters = filter(None, [bc_data.get(l) for l in get_tokenized_lemmas(s)])
    return ngrams(clusters, 2)


class BrownClusterBigramTransform(StatelessTransform):

    def __init__(self, cluster_size=100):
        self.cluster_size = cluster_size

    def transform(self, X):
        bc_data, bc_data_idx = get_brown_cluster_data(self.cluster_size)

        y_dim = len(bc_data_idx.values())
        mat = dok_matrix((len(X), y_dim * 2), dtype=np.float32)

        def set_cluster_pair(i, s, offset=0):
            cx = _get_bigram_clusters(s, bc_data)
            for x in cx:
                idx = bc_data_idx[x]
                mat[i, idx + (y_dim * offset)] = 1

        for i, (_, s) in enumerate(X.iterrows()):
            set_cluster_pair(i, s.claimHeadline)
            set_cluster_pair(i, s.articleHeadline, 1)

        return mat


class AlignedWordsTransform(StatelessTransform):

    def _match(self, w):
        # m = w in ['some', 'none', 'not', 'all', 'every', 'each']
        m = (w == 'not')
        return m

    def transform(self, X):
        mat = np.zeros((len(X), 1))
        for i, (_, s) in enumerate(X.iterrows()):
            idx = get_aligned_data().get((s.claimId, s.articleId))
            f = 0
            if idx:
                claim_tok = get_tokenized_lemmas(s.claimHeadline)
                article_tok = get_tokenized_lemmas(s.articleHeadline)
                for x, y in idx:
                    if x > 0 and y == 0:
                        f = self._match(claim_tok[x-1])
                    elif x == 0 and y > 0:
                        f = self._match(article_tok[y-1])
                    elif [x-1, y-1] not in idx:
                        f = self._match(claim_tok[x-1]) or self._match(article_tok[y-1])
            mat[i, 0] = f

        return mat


_stopwords = stopwords.words('english')


class STSSimilarityTransform(StatelessTransform):

    @staticmethod
    def _sts(self, s1, s2, align):
        align1, align2 = zip(*align)
        ncsa1 = float(len([s1[i] for i in align1 if s1[i] not in _stopwords]))
        ncsa2 = float(len([s2[i] for i in align2 if s2[i] not in _stopwords]))
        nc1 = float(len([t for t in s1 if t not in _stopwords]))
        nc2 = float(len([t for t in s2 if t not in _stopwords]))
        return (ncsa1 + ncsa2) / (nc1 + nc2)

    def transform(self, X):
        mat = np.zeros((len(X), 1))
        for i, (_, s) in enumerate(X.iterrows()):
            idx = get_aligned_data().get((s.claimId, s.articleId))
            if idx:
                try:
                    claim_tok = get_tokenized_lemmas(s.claimHeadline)
                    article_tok = get_tokenized_lemmas(s.articleHeadline)
                    mat[i, 0] = self._sts(claim_tok, article_tok, idx)
                except:
                    pass
        return mat


class AlignedSimilarityTransform(StatelessTransform):

    def transform(self, X):
        mat = np.zeros((len(X), 2))
        # mat = np.zeros((len(X), 1))
        for i, (_, s) in enumerate(X.iterrows()):
            idx = get_aligned_data().get((s.claimId, s.articleId))
            if idx:
                try:
                    claim_tok = get_tokenized_lemmas(s.claimHeadline)
                    article_tok = get_tokenized_lemmas(s.articleHeadline)
                    # mat[i, 0] = _sts(claim_tok, article_tok, idx)
                    mat[i, 0] = len(idx)/float(len(claim_tok))
                    mat[i, 1] = len(idx)/float(len(article_tok))
                except:
                    pass
        return mat

