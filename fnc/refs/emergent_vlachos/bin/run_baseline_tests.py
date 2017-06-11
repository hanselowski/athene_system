import sys
import os

sys.path.append(os.path.join('..', 'src'))

from model.utils import get_dataset, split_data, run_test
from model.baseline.baseline_predictors import ProbabilityPredictor, ChancePredictor, \
    MajorityPredictor, WordOverlapBaselinePredictor


if __name__ == '__main__':
    train_data = get_dataset('url-versions-2015-06-14-clean-train.csv')
    X, y = split_data(train_data)
    test_data = get_dataset('url-versions-2015-06-14-clean-test.csv')

    print('\n>> Chance predictor <<\n')
    print(run_test(X, y, test_data, ChancePredictor()))

    print('\n>> Majority predictor <<\n')
    print(run_test(X, y, test_data, MajorityPredictor()))

    print('\n>> Probability predictor <<\n')
    print(run_test(X, y, test_data, ProbabilityPredictor()))

    print('\n>> Word overlap predictor <<\n')
    print(run_test(X, y, test_data, WordOverlapBaselinePredictor()))