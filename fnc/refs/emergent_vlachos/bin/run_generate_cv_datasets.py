import os

from model.utils import get_dataset, split_data
from model.cross_validation import ClaimKFold

if __name__ == '__main__':
    train_data = get_dataset('url-versions-2015-06-14-clean-train.csv')
    X, y = split_data(train_data)

    ckf = ClaimKFold(X)

    fold = 1
    for train_index, test_index in ckf:
        Z_test = X.iloc[test_index, :].copy()
        Z_test['articleHeadlineStance'] = y.iloc[test_index]
        Z_test.to_csv(os.path.join('..', 'data', 'emergent',
                                   'url-versions-2015-06-14-clean-test-fold-{0:d}.csv'.format(fold)))

        Z_train = X.iloc[train_index, :].copy()
        Z_train['articleHeadlineStance'] = y.iloc[train_index]
        Z_train.to_csv(os.path.join('..', 'data', 'emergent',
                                    'url-versions-2015-06-14-clean-train-fold-{0:d}.csv'.format(fold)))

        fold += 1

