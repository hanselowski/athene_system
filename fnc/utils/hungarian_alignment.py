import os
import numpy as np
import pandas as pd
import pickle
from munkres import Munkres, make_cost_matrix
from fnc.utils.data_helpers import get_stem, get_tokenized_lemmas

# Implementation taken from: https://github.com/willferreira/mscproject

_max_ppdb_score = 10.0
_min_ppdb_score = -_max_ppdb_score
_data_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
_pickled_data_folder = os.path.join(_data_folder, 'pickled')
_ppdb_database_file = 'ppdb-1.0-xl-lexical'
_munk = Munkres()

class hungarian_alignment_calculator:
    def __init__(self):
        self.ppdb_data = self.get_ppdb_data()

    def to_float(self, s):
        return np.nan if s == 'NA' else float(s)


    # no reference file found for this procedure in original vlachos project
    # ppdb databases can be found under: http://www.cis.upenn.edu/~ccb/ppdb/
    # the following procedure seems not to load the whole file -> decided to omitt loading ppdb data and use predelivered pickle
    def process_ppdb_data(self):
        if not os.path.exists(os.path.join(_data_folder, 'ppdb', _ppdb_database_file)):
            print('Did not find \"' + os.path.join(_data_folder, 'ppdb', _ppdb_database_file))
            print("Please download from http://www.cis.upenn.edu/~ccb/ppdb/")
            return {}

        with open(os.path.join(_data_folder, 'ppdb', _ppdb_database_file), 'r', encoding='utf-8') as f:
            ppdb = {}
            for line in f:
                data = line.split('|||')
                text_lhs = data[1].strip(' ,')
                text_rhs = data[2].strip(' ,')
                #filtering out entries with more than one word
                if len(text_lhs.split(' ')) > 1 or len(text_rhs.split(' ')) > 1:
                    print('text_lhs: ' + text_lhs + "\ntext_rhs: " + text_rhs )
                    print("Problem while parsing line: " + line)
                    continue
                ppdb_score = self.to_float(data[3].strip().split()[0].split('=')[1])
                entailment = data[-1].strip()
                paraphrases = ppdb.setdefault(text_lhs, list())
                paraphrases.append((text_rhs, ppdb_score, entailment))
            print(str(len(ppdb)) + ' Lines parsed from ' + _ppdb_database_file)
        return ppdb


    def create_ppdb_pickle(self):
        if not os.path.exists(_pickled_data_folder):
            print('Creating base-directory: ' + _pickled_data_folder)
            os.makedirs(_pickled_data_folder)
        ppdb = self.process_ppdb_data()
        if ppdb:
            with open(os.path.join(_pickled_data_folder, 'ppdb.pickle'), 'wb') as f:
                pickle.dump(ppdb, f, pickle.HIGHEST_PROTOCOL)
                print('Done generating ppdb.pickle')
        else:
            quit("Aborting - an error occurred")


    def get_ppdb_data(self):
        if not os.path.exists(os.path.join(_pickled_data_folder, 'ppdb.pickle')):
            print('PPDB data did not exist, now generating ppdb.pickle')
            self.create_ppdb_pickle()
        with open(os.path.join(_pickled_data_folder, 'ppdb.pickle'), 'rb') as f:
            return pickle.load(f, encoding='utf-8')


    def compute_paraphrase_score(self, s, t):
        """Return numerical estimate of whether t is a paraphrase of s, up to
        stemming of s and t."""
        s_stem = get_stem(s)
        t_stem = get_stem(t)

        if s_stem == t_stem:
            return _max_ppdb_score

        # get PPDB paraphrases of s, and find matches to t, up to stemming
        s_paraphrases = set(self.ppdb_data.get(s, [])).union(self.ppdb_data.get(s_stem, []))
        matches = list(filter(lambda a: a[0] == t or a[0] == t_stem, s_paraphrases))
        if matches:
            return max(matches, key=lambda x: x[1])[1]
        return _min_ppdb_score


    def calc_hungarian_alignment_score(self, s, t):
        """Calculate the alignment score between the two texts s and t
        using the implementation of the Hungarian alignment algorithm
        provided in https://pypi.python.org/pypi/munkres/."""
        s_toks = get_tokenized_lemmas(s)
        t_toks = get_tokenized_lemmas(t)
        #print("#### new ppdb calculation ####")
        #print(s_toks)
        #print(t_toks)
        df = pd.DataFrame(index=s_toks, columns=t_toks, data=0.)

        for c in s_toks:
            for a in t_toks:
                df.ix[c, a] = self.compute_paraphrase_score(c, a)

        matrix = df.values
        cost_matrix = make_cost_matrix(matrix, lambda cost: _max_ppdb_score - cost)

        indexes = _munk.compute(cost_matrix)
        total = 0.0
        for row, column in indexes:
            value = matrix[row][column]
            total += value
        #print(s + ' || ' + t + ' :' + str(indexes) + ' - ' + str(total / float(np.min(matrix.shape))))

        # original procedure returns indexes and score - i do not see any use for the indexes as a feature
        # return indexes, total / float(np.min(matrix.shape))
        return total / float(np.min(matrix.shape))

    '''
    if __name__ == '__main__':
        testdata = [('he was walking outside of the house', 'he walked out of his home'),
                    ('those kinds', 'such'),
                    ('clock', 'watch'),
                    ('appendix', 'appendix'),
                    ('tutorial', 'erdbeertorte')]

        for s,t in testdata:
            print('Comparing: ' + s + ' <-> ' + t)
            print('score: ' + str(calc_hungarian_alignment_score(s,t)))
    '''