import itertools as it
from collections import Counter

import numpy as np
import scipy as sp

from model.base import AbstractPredictor
from model.utils import get_tokenized_lemmas


_label_map = {
    0: 'for',
    1: 'against',
    2: 'observing',
}


class ChancePredictor(AbstractPredictor):

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.array(map(lambda x: _label_map[x], it.starmap(np.random.randint, [(0, 3)] * X.shape[0])))

    def predict_proba(self, X):
        return np.ones(X.shape) * (1.0 / 3.0)


class MajorityPredictor(AbstractPredictor):

    def __init__(self):
        self.majority = None

    def fit(self, X, y=None):
        self.majority = sp.stats.mode(y)[0][0]
        return self

    def predict(self, X):
        return np.array([self.majority] * X.shape[0])


class ProbabilityPredictor(AbstractPredictor):

    def __init__(self):
        self.dist = None

    def fit(self, X, y=None):
        probabilities = Counter(y)
        norm_c = float(sum(probabilities.values()))
        probabilities = dict({k: n/norm_c for (k, n) in probabilities.items()})
        self.dist = sp.stats.rv_discrete(name='EMERGENTX',
                                         values=(_label_map.keys(), [probabilities[_label_map[i]]
                                                                     for i in _label_map.keys()]))
        return self

    def predict(self, X):
        return np.array(map(_label_map.get, self.dist.rvs(size=X.shape[0])))


class WordOverlapBaselinePredictor(AbstractPredictor):

    def __init__(self):
        self.thresholds = None

    @staticmethod
    def _compute_overlap(row):
        claim_lemmas = get_tokenized_lemmas(row.claimHeadline)
        article_lemmas = get_tokenized_lemmas(row.articleHeadline)
        intersect = set(claim_lemmas).intersection(article_lemmas)
        union = set(claim_lemmas).union(article_lemmas)
        return float(len(intersect)) / len(union)

    def fit(self, X, y=None):
        overlap = []
        for _, row in X.iterrows():
            overlap.append(self._compute_overlap(row))
        X_copy = X.copy()
        X_copy['overlap'] = overlap
        X_copy['stance'] = y
        self.thresholds = X_copy.groupby('stance').overlap.mean()
        self.thresholds.sort()
        return self

    def predict(self, X):
        labels = []
        for idx, row in X.iterrows():
            overlap = self._compute_overlap(row)
            label = self.thresholds.index[1]
            if overlap <= self.thresholds[0]:
                label = self.thresholds.index[0]
            if overlap > self.thresholds[2]:
                label = self.thresholds.index[2]
            labels.append(label)
        return np.array(labels)