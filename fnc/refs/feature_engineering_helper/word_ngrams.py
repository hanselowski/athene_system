from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from fnc.refs.utils.dataset import DataSet
import numpy as np
import nltk
import pickle
import string
import os
import os.path as path
from fnc.refs.utils.generate_test_splits import kfold_split
from sklearn import feature_extraction
from fnc.refs.utils.testDataset import TestDataSet

def get_head_body_tuples_unlbled_test():

    data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__))))))
    d = TestDataSet(data_path)

    h = []
    b = []
    for stance in d.stances:
        h.append(stance['Headline'])
        b.append(d.articles[stance['Body ID']])

    return h, b

def get_head_body_tuples(include_holdout=False):
    # file paths
    data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__))))))
    splits_dir = "%s/data/fnc-1/splits" % (path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__))))))
    dataset = DataSet(data_path)

    def get_stances(dataset, folds, holdout):
        # Creates the list with a dict {'headline': ..., 'body': ..., 'stance': ...} for each
        # stance in the data set (except for holdout)
        stances = []
        for stance in dataset.stances:
            if stance['Body ID'] in holdout and include_holdout == True:
                stances.append(stance)
            for fold in folds:  # TODO maybe just flatten folds beforehand
                if stance['Body ID'] in fold:
                    stances.append(stance)

        return stances

    # create new vocabulary
    folds, holdout = kfold_split(dataset, n_folds=10, base_dir=splits_dir)  # [[133,1334,65645,], [32323,...]] => body ids for each fold
    stances = get_stances(dataset, folds, holdout)

    print("Stances length: " + str(len(stances)))

    h = []
    b = []
    # create the final lists with all the headlines and bodies of the set except for holdout
    for stance in stances:
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    return h, b

def create_word_ngram_vocabulary(ngram_range=(1,1), max_features=100, lemmatize=False, term_freq=False, norm='l1', use_idf=False, include_holdout=False):
    """
    Creates, returns and saves a vocabulary for (Count-)Vectorizer over all training and test data (holdout excluded) to create BoW
    methods. The method simplifies using the pipeline and later tests with feature creation for a single headline and body.
    This method will cause bleeding, since it also includes the test set.

    :param filename: a filename for the vocabulary
    :param ngram_range: the ngram range for the Vectorizer. Default is (1, 1) => unigrams
    :param max_features: the length of the vocabulary
    :return: the vocabulary
    """
    # file paths
    features_dir = "%s/data/fnc-1/features" % (path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__))))))

    print("Calling create_word_ngram_vocabulary with ngram_range=("
          + str(ngram_range[0]) + ", " + str(ngram_range[1]) + "), max_features="
          + str(max_features) + ", lemmatize=" +  str(lemmatize) + ", term_freq=" + str(term_freq))

    def normalize_word(w):
        _wnl = nltk.WordNetLemmatizer()
        return _wnl.lemmatize(w).lower()

    def get_all_stopwords():
        stop_words_nltk = set(stopwords.words('english'))  # use set for faster "not in" check
        stop_words_sklearn = feature_extraction.text.ENGLISH_STOP_WORDS
        all_stop_words = stop_words_sklearn.union(stop_words_nltk)
        return all_stop_words

    def get_tokenized_lemmas_without_stopwords(s):
        all_stop_words = get_all_stopwords()
        return [normalize_word(t) for t in nltk.word_tokenize(s)
                         if t not in string.punctuation and t.lower() not in all_stop_words]


    def train_vocabulary(head_and_body):
        # trains a CountVectorizer on all of the data except for holdout data
        if lemmatize == False:
            vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english', max_features=max_features)
            if term_freq == True:
                vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words='english', max_features=max_features, use_idf=use_idf, norm=norm)
        else:
            vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=max_features,
                                         tokenizer=get_tokenized_lemmas_without_stopwords)
            if term_freq == True:
                vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features,
                                         tokenizer=get_tokenized_lemmas_without_stopwords, use_idf=use_idf, norm=norm)
        vectorizer.fit_transform(head_and_body)
        vocab = vectorizer.vocabulary_
        return vocab

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]
        return head_and_body


    # create filename for vocab
    vocab_file = "word_(" + str(ngram_range[0]) + "_" + str(ngram_range[1]) + ")-gram_" + str(max_features)
    if lemmatize == True:
        vocab_file += "_lemmatized"
    if term_freq == True:
        vocab_file += "_tf"
    if use_idf == True:
        vocab_file += "_idf"
    if include_holdout == True:
        vocab_file += "_holdout"
    vocab_file += "_" + norm + ".pickle"

    # if vocab already exists, just load and return it
    if (os.path.exists(features_dir + "/" + vocab_file)):
        with open(features_dir + "/" + vocab_file, 'rb') as handle:
            vocab = pickle.load(handle)
            print("Existing vocabulary found and load.")
            return vocab

    h, b = get_head_body_tuples(include_holdout=include_holdout)
    head_and_body = combine_head_and_body(h, b) # combine head and body
    vocab = train_vocabulary(head_and_body) # get vocabulary (features)

    # save the vocabulary as file
    with open(features_dir + "/" + vocab_file, 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("vocab length: " + str(len(vocab)))
    return vocab

def combine_head_and_body(headlines, bodies):
    head_and_body = [headline + " " + body for i, (headline, body) in
                     enumerate(zip(headlines, bodies))]

    return head_and_body

def word_ngrams_concat_count_no_bleeding(headlines, bodies, headlines_test, bodies_test,
                                         binary=False, max_features=200, ngram_range=(1, 1),
                                         analyzer='word', lowercase=True, max_df=1.0, min_df=1):
    """
    Simple bag of words feature extraction
    """

    def get_train_features():
        tf_vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english', max_features=max_features,
                                        binary=binary, analyzer=analyzer, lowercase=lowercase, max_df=max_df, min_df=min_df)
        tf_vectorizer.fit_transform(combine_head_and_body(headlines, bodies))
        vocab = tf_vectorizer.vocabulary_

        tf_vectorizer_head = CountVectorizer(vocabulary=vocab, stop_words='english', binary=binary,
                                             analyzer=analyzer, lowercase=lowercase)
        X_train_head = tf_vectorizer_head.fit_transform(headlines)

        tf_vectorizer_body = CountVectorizer(vocabulary=vocab, stop_words='english', binary=binary,
                                             analyzer=analyzer, lowercase=lowercase)
        X_train_body = tf_vectorizer_body.fit_transform(bodies)

        X_train = np.concatenate([X_train_head.toarray(), X_train_body.toarray()], axis=1)

        return X_train, vocab

    def get_test_features(vocab):
        tf_vectorizer_head = CountVectorizer(vocabulary=vocab, stop_words='english', binary=binary,
                                             analyzer=analyzer, lowercase=lowercase)
        X_test_head = tf_vectorizer_head.fit_transform(headlines_test)

        tf_vectorizer_body = CountVectorizer(vocabulary=vocab, stop_words='english', binary=binary,
                                             analyzer=analyzer, lowercase=lowercase)
        X_test_body = tf_vectorizer_body.fit_transform(bodies_test)

        X_test = np.concatenate([X_test_head.toarray(), X_test_body.toarray()], axis=1)
        return X_test


    X_train, vocab = get_train_features()
    X_test = get_test_features(vocab)

    return X_train, X_test

def word_ngrams_concat_tfidf_no_bleeding(headlines, bodies, headlines_test, bodies_test,
                                         binary=False, max_features=200, ngram_range=(1, 1),
                                         use_idf=False, smooth_idf=False, norm='l2', sublinear_tf=False,
                                         analyzer='word', lowercase=True, stop_words='english', max_df=1.0, min_df=1):
    """
    Takes parameters to fit a TfidfVectorizer on the training stances
    and transforms the test headlines and bodies seperately on it. At the end, both feature vectors
    get concatenated and returned. Finally X_train and X_test will be returned. No bleeding between test and
    train data.
    """

    def get_train_features():
        tf_vectorizer = TfidfVectorizer(ngram_range=ngram_range,
                                           stop_words=stop_words, max_features=max_features,
                                           use_idf=use_idf, smooth_idf=smooth_idf, norm=norm,
                                        binary=binary, sublinear_tf=sublinear_tf, analyzer=analyzer, lowercase=lowercase,
                                        max_df=max_df, min_df=min_df)
        tf_vectorizer.fit_transform(combine_head_and_body(headlines, bodies))
        vocab = tf_vectorizer.vocabulary_

        tf_vectorizer_head = TfidfVectorizer(vocabulary=vocab, use_idf=use_idf,
                                             smooth_idf=smooth_idf, norm=norm,
                                             stop_words=stop_words, binary=binary,
                                             sublinear_tf=sublinear_tf, analyzer=analyzer, lowercase=lowercase)
        X_train_head = tf_vectorizer_head.fit_transform(headlines)

        tf_vectorizer_body = TfidfVectorizer(vocabulary=vocab, use_idf=use_idf,
                                             smooth_idf=smooth_idf, norm=norm,
                                             stop_words=stop_words, binary=binary,
                                             sublinear_tf=sublinear_tf, analyzer=analyzer, lowercase=lowercase)
        X_train_body = tf_vectorizer_body.fit_transform(bodies)

        X_train = np.concatenate([X_train_head.toarray(), X_train_body.toarray()], axis=1)

        return X_train, vocab


    def get_test_features(vocab):
        tf_vectorizer_head = TfidfVectorizer(vocabulary=vocab, use_idf=use_idf,
                                             smooth_idf=smooth_idf, norm=norm,
                                             stop_words=stop_words, binary=binary,
                                             sublinear_tf=sublinear_tf, analyzer=analyzer, lowercase=lowercase)
        X_test_head = tf_vectorizer_head.fit_transform(headlines_test)

        tf_vectorizer_body = TfidfVectorizer(vocabulary=vocab, use_idf=use_idf,
                                             smooth_idf=smooth_idf, norm=norm,
                                             stop_words=stop_words, binary=binary,
                                             sublinear_tf=sublinear_tf, analyzer=analyzer, lowercase=lowercase)
        X_test_body = tf_vectorizer_body.fit_transform(bodies_test)

        X_test = np.concatenate([X_test_head.toarray(), X_test_body.toarray()], axis=1)
        return X_test


    X_train, vocab = get_train_features()
    X_test = get_test_features(vocab)

    return X_train, X_test

def word_ngrams_concat(headlines, bodies, max_features=200, ngram_range=(1, 1),
                       use_idf=False, norm='l2', lemmatize=False,
                       term_freq=True, include_holdout=False):
    """
    Takes parameters to fit a TfidfVectorizer on the training and test stance (optional holdout)
    and transforms the headlines and bodies seperately on it. At the end, both feature vectors
    get concatenated and returned
    """
    def get_features(vocab):
        if  term_freq == True:
            vectorizer_head = TfidfVectorizer(vocabulary=vocab, use_idf=use_idf,
                                             norm=norm, stop_words='english')
        else:
            vectorizer_head = CountVectorizer(vocabulary=vocab,
                                             stop_words='english')
        X_head = vectorizer_head.fit_transform(headlines)

        if term_freq == True:
            vectorizer_body = TfidfVectorizer(vocabulary=vocab, use_idf=use_idf,
                                             norm=norm, stop_words='english')
        else:
            vectorizer_body = CountVectorizer(vocabulary=vocab,
                                             stop_words='english')
        X_body = vectorizer_body.fit_transform(bodies)

        X = np.concatenate([X_head.toarray(), X_body.toarray()], axis=1)

        return X


    vocab = create_word_ngram_vocabulary(ngram_range=ngram_range, max_features=max_features,
                                         lemmatize=lemmatize, use_idf=use_idf, term_freq=term_freq, norm=norm,
                                         include_holdout=include_holdout)

    return get_features(vocab)