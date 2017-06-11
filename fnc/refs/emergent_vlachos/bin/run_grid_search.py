from itertools import chain, combinations

from sklearn import grid_search

from model.utils import get_dataset, split_data
from model.classifiers.lr_predictors import LogitPredictor
from model.cross_validation import ClaimKFold

from model.baseline.transforms import (
    RefutingWordsTransform,
    QuestionMarkTransform,
    HedgingWordsTransform,
    InteractionTransform,
    NegationOfRefutingWordsTransform,
    BoWTransform,
    PolarityTransform
)

from model.ext.transforms import (
    AlignedPPDBSemanticTransform,
    NegationAlignmentTransform,
    Word2VecSimilaritySemanticTransform,
    DependencyRootDepthTransform,
    SVOTransform
)


def do_grid_search(X, y, classifier, param_grid, cv):

    def scorer(estimator, XX, yy):
        return estimator.score(XX, yy).accuracy


    clf = grid_search.GridSearchCV(classifier, param_grid, cv=cv, scoring=scorer, verbose=True)
    clf.fit(X, y)

    print("Best parameters set found on development set:")
    print(clf.best_estimator_)
    print(clf.best_params_)
    print(clf.best_score_)
    return clf.best_estimator_


def powerset(iterable):
    s = list(iterable)
    return map(list, filter(lambda x: len(x) > 0, chain.from_iterable(combinations(s, r) for r in range(len(s)+1))))


if __name__ == "__main__":

    transforms = {
        Word2VecSimilaritySemanticTransform,
        AlignedPPDBSemanticTransform,
        NegationOfRefutingWordsTransform,
        NegationAlignmentTransform,
        DependencyRootDepthTransform,
        SVOTransform,
        # PolarityTransform
    }

    class CallableBowTransform(object):

        def __init__(self, ngram_ur, max_feat):
            self.ngram_ur = ngram_ur
            self.max_feat = max_feat

        def __call__(self, *args, **kwargs):
            return BoWTransform(self.ngram_ur, self.max_feat)

        def __str__(self):
            return 'CallableBowTransform({0:d}, {1:d})'.format(self.ngram_ur, self.max_feat)

        def __repr__(self):
            return self.__str__()

    bow_transform_funcs = []
    for ngram_ur in range(2, 3):
        for max_feat in (200, 300, 400, 500, 600, 700, 800):
            bow_transform_funcs.append(CallableBowTransform(ngram_ur, max_feat))

    transforms.union(bow_transform_funcs)

    train_data = get_dataset('url-versions-2015-06-14-clean-train.csv')
    X, y = split_data(train_data)
    ckf = ClaimKFold(X, n_folds=5)

    classifier = LogitPredictor([QuestionMarkTransform])

    param_grid = [
        {
            # 'transforms': powerset(transforms)
            'transforms': bow_transform_funcs
        }
    ]

    do_grid_search(X, y, classifier, param_grid, ckf)