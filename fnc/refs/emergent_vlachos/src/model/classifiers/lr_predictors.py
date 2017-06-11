from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from model.base import AbstractPredictor


class LogitPredictor(AbstractPredictor):

    def __init__(self, transforms):
        self.transforms = transforms

        union = make_union(*[t() for t in transforms])
        pipeline = [union]
        self.pipeline = make_pipeline(*pipeline)
        self.classifier = LogisticRegression(penalty='l1', class_weight='auto')


class ObservingPredictor(AbstractPredictor):

    def __init__(self, transforms):
        self.transforms = transforms

        union = make_union(*[t() for t in transforms])
        pipeline = [union]
        self.pipeline = make_pipeline(*pipeline)
        self.classifier = LogisticRegression(penalty='l1', class_weight='auto')

    def _transform(self, y):
        y1 = y.copy()
        y1[(y1 == 'for') | (y1 == 'against')] = 'not_observing'
        return y1

    def fit(self, X, y=None):
        # transform 'for' and 'against' labels to 'not_observing'
        # and fit a binary classifier using logistic regression
        y1 = self._transform(y)
        Z = self.pipeline.fit_transform(X, y1)
        self.classifier.fit(Z, y1)
        return self


class ForAgainstPredictor(AbstractPredictor):

    def __init__(self, transforms):
        self.transforms = transforms

        union = make_union(*[t() for t in transforms])
        pipeline = [union]
        self.pipeline = make_pipeline(*pipeline)
        self.classifier = LogisticRegression(penalty='l1', class_weight='auto')

    def _transform(self, X, y):
        mask = (y == 'for') | (y == 'against')
        y1 = y[mask]
        X1 = X[mask]
        return X1, y1

    def fit(self, X, y=None):
        # just work with the 'for' and 'against' data
        X1, y1 = self._transform(X, y)
        Z = self.pipeline.fit_transform(X1, y1)
        self.classifier.fit(Z, y1)
        return self


class CompoundPredictor(AbstractPredictor):

    def __init__(self, observing_transforms, for_against_transforms):
        self.observing_transforms = observing_transforms
        self.for_against_transforms = for_against_transforms

        self.observing_predictor = ObservingPredictor(observing_transforms)
        self.for_against_predictor = ForAgainstPredictor(for_against_transforms)

    def fit(self, X, y=None):
        self.observing_predictor.fit(X, y)
        self.for_against_predictor.fit(X, y)
        return self

    def predict(self, X):
        prediction = self.observing_predictor.predict(X)
        prediction_for_against = self.for_against_predictor.predict(X)
        not_observing = (prediction != 'observing')
        prediction[not_observing] = prediction_for_against[not_observing]
        return prediction

    def predict_proba(self, X):
        pass
