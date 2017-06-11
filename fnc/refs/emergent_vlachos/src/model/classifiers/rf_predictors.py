from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, make_union

from model.base import AbstractPredictor


class RandomForestPredictor(AbstractPredictor):

    def __init__(self, transforms, n_estimators=2000, criterion='gini', min_samples_leaf=2, n_jobs=-1):
        self.transforms = transforms
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs

        union = make_union(*[t() for t in transforms])
        pipeline = [union]

        self.pipeline = make_pipeline(*pipeline)
        self.classifier = RandomForestClassifier(n_estimators, criterion, min_samples_leaf=min_samples_leaf, n_jobs=-1)

