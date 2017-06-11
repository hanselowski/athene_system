import numpy as np

from sklearn.cross_validation import _BaseKFold, KFold


class ClaimKFold(_BaseKFold):

    def __init__(self, data, n_folds=10, shuffle=False):
        super(ClaimKFold, self).__init__(len(data), n_folds, None, False, None)
        self.shuffle = shuffle
        self.data = data.copy()
        self.data['iloc_index'] = range(len(self.data))

    def _iter_test_indices(self):
        claim_ids = np.unique(self.data.claimId)
        cv = KFold(len(claim_ids), self.n_folds, shuffle=self.shuffle)

        for _, test in cv:
            test_claim_ids = claim_ids[test]
            test_data = self.data[self.data.claimId.isin(test_claim_ids)]
            yield test_data.iloc_index.values

    def __len__(self):
        return self.n_folds