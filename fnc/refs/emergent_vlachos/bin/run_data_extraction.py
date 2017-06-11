# encoding: utf-8

import os
import argparse
import string
import re
import numpy as np

import pandas as pd

from model.utils import VALID_STANCE_LABELS, generate_test_training_set, get_contraction_mappings

_relevant_data_columns = [
    'claimId',
    'claimTruthiness',
    'claimHeadline',
    'articleId',
    'articleVersion',
    'articleHeadline',
    'articleHeadlineStance'
]

_final_data_columns = [
    'claimHeadline',
    'articleHeadline',
    'articleHeadlineStance',
    'articleId',
    'claimId',

]


def _clean_text(data):
    data = data.copy()

    # Remove specific articles with known issues
    to_drop = [
        '58eacfb0-7af4-11e4-b794-93ed794d7b91',  # Contains: "Sorry - this page has been removed."
        '8ac9a380-c383-11e4-9435-a96703525a9e'   # Contains only phrase: "Jonathan S. Geller"
    ]
    data = data[~data.articleId.isin(to_drop)]

    # Strip Claim: prefix from claim headline
    def strip_claim_suffix(s):
        if s.startswith('Claim:'):
            return s[7:]
        else:
            return s
    data['claimHeadline'] = data.claimHeadline.apply(strip_claim_suffix)

    # Collection of funcs to apply to claim and article, in order
    funcs = []

    # Convert chars to UTF
    def convert_to_utf(s):
        return s.decode('utf8')

    funcs.append(convert_to_utf)

    # Strip words containing article source prefix
    _strip_words = ['REPORT', 'REPORTS', 'PODCAST',
                    'CNN', 'CNBC', 'Net Extra', 'WSJ']

    def strip_source_prefix(s):
        for w in _strip_words:
            s = re.sub(w + ':', '', s, flags=re.IGNORECASE)
        return s

    funcs.append(strip_source_prefix)

    # Clean up some unicode quotations poop
    utf_quotation_marks = [u'\u2032', u'\u2019', u'\u2018',
                           u'\u201C', u'\u201D']

    def convert_quotations(s):
        for c in utf_quotation_marks:
            s = s.replace(c, "'")
        return s

    funcs.append(convert_quotations)

    # Re-introduce 's and 't which were encoded as ?s and ?t in some cases
    def convert_bad_apostrophe(s):
        s = s.replace('?s', '\'s')
        return s.replace('?t', '\'t')

    funcs.append(convert_bad_apostrophe)

    # Drop remaining non-ascii stuff we don't want
    def drop_non_ascii(s):
        return s.encode('utf-8').decode('ascii', 'ignore')

    funcs.append(drop_non_ascii)

    # Expand contractions
    def expand_contractions(s):
        for c, e in get_contraction_mappings().items():
            s = re.sub(c, e, s, flags=re.IGNORECASE)
        return s

    # funcs.append(expand_contractions)

    my_punctuation = ''.join(set(string.punctuation).difference(['?', ',', '.', ':', '-', '\'']))

    # Strip out any spurious punctuation
    # This should preserve apostrophes, comma, question-mark and full-stop.
    def strip_punctuation(s):
        s = [w.translate(dict.fromkeys(map(ord, my_punctuation))) for w in s.split(" ")]
        s = filter(None, s)
        return ' '.join(s)

    funcs.append(strip_punctuation)

    # Strip out any nested quotation marks
    def strip_internal_quotations(s):
        s = re.sub(r"^'", '', s)
        s = re.sub(r"\s'", " ", s)
        s = re.sub(r"'\s", " ", s)
        s = re.sub(r"'$", '', s)
        s = re.sub(r"[:.,;]'", lambda m: m.group(0)[0], s)
        s = re.sub(r"'[:.,;]", lambda m: m.group(0)[1], s)
        return s

    funcs.append(strip_internal_quotations)

    # Fix problem with words like ?abc?
    def drop_bracketing_question_marks(s):
        s = re.sub(r'\?\w[\w\s]*\?', lambda m: m.group(0)[1: -1], s)
        return s

    funcs.append(drop_bracketing_question_marks)

    for f in funcs:
        print 'Applying function:', f.__name__
        data['articleHeadline'] = data.articleHeadline.apply(f)
        data['claimHeadline'] = data.claimHeadline.apply(f)

    return data[_final_data_columns]


def _main(filepath, filename):
    # pull in all the latest claim data and take a subset the relevant columns
    all_data = pd.DataFrame.from_csv(os.path.join(filepath, filename), index_col=None)
    all_data = all_data.ix[:, _relevant_data_columns]

    # extract all the rows that have an article headline stance in ('for', 'against', 'observing')
    data = all_data[all_data.articleHeadlineStance.isin(VALID_STANCE_LABELS)]

    # subset the data to use on the most recent version of each article, by articleVersion
    idx = data.groupby(['claimId', 'articleId'], sort=False)['articleVersion'].transform(max) == data['articleVersion']
    data = data[idx]

    # Remove data where claim or article headline is NA
    data = data.dropna(subset=['articleHeadline', 'claimHeadline'])

    # Clean up some punctuation fails
    data = _clean_text(data)

    # Some claims or articles might be empty after clean-up
    # so remove them here
    data = data.drop(data[data.articleHeadline == ''].index)
    data = data.drop(data[data.claimHeadline == ''].index)

    # Reindex
    data.index = pd.Index(range(0, len(data.index)))

    # write out a new 'cleaned' data file
    split_filename = os.path.splitext(filename)
    filename_clean = os.path.join(filepath, split_filename[0] + '-clean')
    data.to_csv(filename_clean + split_filename[1], encoding='utf-8')

    # Generate test/train datasets
    test_data, train_data = generate_test_training_set(data)
    filename_test = filename_clean + '-test'
    test_data.to_csv(filename_test + split_filename[1], encoding='utf-8')

    filename_train = filename_clean + '-train'
    train_data.to_csv(filename_train + split_filename[1], encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run_data_extraction cmd-line arguments.')

    parser.add_argument('--filepath', default=os.path.join('..', 'data', 'emergent'), type=str)
    parser.add_argument('--filename', default='url-versions-2015-06-14.csv', type=str)

    args = parser.parse_args()
    _main(args.filepath, args.filename)
