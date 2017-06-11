import math
import os
import os.path as path
from datetime import datetime
import nltk
import numpy as np
import regex as re
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from fnc.refs.feature_engineering_helper import word_ngrams
from fnc.refs.feature_engineering_helper import topic_models
from fnc.utils.doc2vec import avg_embedding_similarity
from fnc.utils.loadEmbeddings import LoadEmbeddings
from fnc.utils.stanford_parser import StanfordMethods
from fnc.utils.tf_idf_helpers import tf_idf_helpers

_wnl = nltk.WordNetLemmatizer()


def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


def gen_or_load_feats(feat_fn, headlines, bodies, feature_file, bodyId, feature, headId=""):
    if not os.path.isfile(feature_file):
        if 'stanford' in feature:
            feats = feat_fn(headlines, bodies, bodyId, headId)
        else:
            feats = feat_fn(headlines, bodies)
        np.save(feature_file, feats)

    return np.load(feature_file)

def gen_non_bleeding_feats(feat_fn, headlines, bodies, headlines_test, bodies_test, features_dir, feature,
                           fold):
    """
    Similar to gen_or_load_feats() it generates the non bleeding features and save them on the disk
    """
    feature_file = "%s/%s.%s.npy" % (features_dir, feature, fold)
    if not os.path.isfile(feature_file):
        print (str(datetime.now()) + ": Generating features for: " + feature + ", fold/holdout: " + str(fold))

        X_train, X_test = feat_fn(headlines, bodies, headlines_test, bodies_test)

        if (str(fold) != 'holdout'):
            np.save("%s/%s.%s.npy" % (features_dir, feature, fold), X_train)
            np.save("%s/%s.%s.test.npy" % (features_dir, feature, fold), X_test)
        else:
            np.save("%s/%s.%s.npy" % (features_dir, feature, 'holdout'), X_train)
            np.save("%s/%s.%s.test.npy" % (features_dir, feature, 'holdout'), X_test)


def word_overlap_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_lemmas(clean_headline)
        clean_body = get_tokenized_lemmas(clean_body)
        features = [
            len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
        X.append(features)
    return X

def refuting_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in _refuting_words]
        X.append(features)
    return X


def polarity_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    def calculate_polarity(text):
        tokens = get_tokenized_lemmas(text)
        return sum([t in _refuting_words for t in tokens]) % 2
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        features = []
        features.append(calculate_polarity(clean_headline))
        features.append(calculate_polarity(clean_body))
        X.append(features)
    return np.array(X)


def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
        if gram in text_body[:100]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features


def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features


def hand_features(headlines, bodies):

    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean(headline).split(" "):
            if headline_token in clean(body):
                bin_count += 1
            if headline_token in clean(body)[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(clean(headline).split(" ")):
            if headline_token in clean(body):
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph

        clean_body = clean(body)
        clean_headline = clean(headline)
        features = []
        features = append_chargrams(features, clean_headline, clean_body, 2)
        features = append_chargrams(features, clean_headline, clean_body, 8)
        features = append_chargrams(features, clean_headline, clean_body, 4)
        features = append_chargrams(features, clean_headline, clean_body, 16)
        features = append_ngrams(features, clean_headline, clean_body, 2)
        features = append_ngrams(features, clean_headline, clean_body, 3)
        features = append_ngrams(features, clean_headline, clean_body, 4)
        features = append_ngrams(features, clean_headline, clean_body, 5)
        features = append_ngrams(features, clean_headline, clean_body, 6)
        return features

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        X.append(binary_co_occurence(headline, body)
                 + binary_co_occurence_stops(headline, body)
                 + count_grams(headline, body))
    return X

###########################
# NEW FEATURES START HERE #
###########################
def NMF_cos_50(headlines, bodies):
    """
    Implements non negative matrix factorization. Calculates the cos distance between the resulting head and body vector.
    """
    return topic_models.NMF_topics(headlines, bodies, n_topics=50, include_holdout=False, include_unlbled_test=False)


def NMF_cos_300_holdout_unlbled_test(headlines, bodies):
    """
    Implements non negative matrix factorization. Calculates the cos distance between the resulting head and body vector.
    """
    return topic_models.NMF_topics(headlines, bodies, n_topics=300, include_holdout=True, include_unlbled_test=True)

def latent_dirichlet_allocation_25(headlines, bodies):
    """
    Sklearn LDA implementation based on the 5000 most important words (based on train+test+holdout+ unlabeled test data's term freq => bleeding).
    Returns feature vector of cosinus distances between the topic models of headline and bodies.

    Links:
        https://pypi.python.org/pypi/lda, bottom see suggestions like MALLET, hca
        https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
        https://www.quora.com/What-are-the-best-features-to-put-into-Latent-Dirichlet-Allocation-LDA-for-topic-modeling-of-short-text
    """
    return topic_models.latent_dirichlet_allocation_cos(headlines, bodies, n_topics=25, include_holdout=False,
                                                        use_idf=False, term_freq=True, incl_unlbled_test=False)

def latent_dirichlet_allocation_25_holdout_unlbled_test(headlines, bodies):
    """
    Sklearn LDA implementation based on the 5000 most important words (based on train+test+holdout+ unlabeled test data's term freq => bleeding).
    Returns feature vector of cosinus distances between the topic models of headline and bodies.

    Links:
        https://pypi.python.org/pypi/lda, bottom see suggestions like MALLET, hca
        https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
        https://www.quora.com/What-are-the-best-features-to-put-into-Latent-Dirichlet-Allocation-LDA-for-topic-modeling-of-short-text
    """
    return topic_models.latent_dirichlet_allocation_cos(headlines, bodies, n_topics=25, include_holdout=True,
                                                        use_idf=False, term_freq=True, incl_unlbled_test=True)

def latent_semantic_indexing_gensim_300_concat_holdout(headlines, bodies):
    """
    Takes all the data (holdout+test+train) and interpretes the headlines and bodies as different
    documents. Instead of combining them, they are appended. Then it tokenizes these ~50k headline-docs and ~50k body-docs,
    builds a Tfidf-Matrix out of them and creates a LSI-Model out of it. In the next step the headlines and
    bodies for the feature generation are also treated as different documents and merely appended. Also, they are tokenized and
    a Tfifd-Matrix is built. This matix is passed to the learned LSI-Model and a Matrix is being returned.
    In this matrix, each document is represented as a vector with length(topics) of (topic-id, distance of this doc to the topic).
    The probabilities are then taken as a feature vector for the document. The first half of the matrix represent the headline docs,
    the latter half represent the body docs. In the end, the feature vectors of the headlines get concatenated with its body feature vector.

    The differences to the latent_semantic_indexing_gensim_300_concat_OLD are:
        - holdout data is also used
        - a Tfidf matrix is built and used to create the LSI model and also to retrieve the features instead of just a corpus to build the LSI model and
            passing each headline and body separately into the LSI model to retrieve its features (does it make a difference, since dictionary already takes
            tfidf into account?)
        - the vectors are taken fully and not just the cosinus distance between them
    """
    return topic_models.latent_semantic_indexing_gensim_concat(headlines, bodies, n_topics=300, include_holdout=True,
                                                               include_unlbled_test=False)

def latent_semantic_indexing_gensim_300_concat_holdout_unlbled_test(headlines, bodies):
    """
    Takes all the data (holdout+test+train) and interpretes the headlines and bodies as different
    documents. Instead of combining them, they are appended. Then it tokenizes these ~50k headline-docs and ~50k body-docs,
    builds a Tfidf-Matrix out of them and creates a LSI-Model out of it. In the next step the headlines and
    bodies for the feature generation are also treated as different documents and merely appended. Also, they are tokenized and
    a Tfifd-Matrix is built. This matix is passed to the learned LSI-Model and a Matrix is being returned.
    In this matrix, each document is represented as a vector with length(topics) of (topic-id, distance of this doc to the topic).
    The probabilities are then taken as a feature vector for the document. The first half of the matrix represent the headline docs,
    the latter half represent the body docs. In the end, the feature vectors of the headlines get concatenated with its body feature vector.

    The differences to the latent_semantic_indexing_gensim_300_concat_OLD are:
        - holdout data is also used
        - a Tfidf matrix is built and used to create the LSI model and also to retrieve the features instead of just a corpus to build the LSI model and
            passing each headline and body separately into the LSI model to retrieve its features (does it make a difference, since dictionary already takes
            tfidf into account?)
        - the vectors are taken fully and not just the cosinus distance between them
    """
    return topic_models.latent_semantic_indexing_gensim_concat(headlines, bodies, n_topics=300, include_holdout=True,
                                                               include_unlbled_test=True)

def NMF_concat_300_holdout(headlines, bodies):
    """
    Implements non negative matrix factorization. Concatenates the resulting head and body vector.
    """
    return topic_models.NMF_topics(headlines, bodies, n_topics=300, include_holdout=True, include_unlbled_test=False,
                                   cosinus_dist=False)

def NMF_concat_300_holdout_unlbled_test(headlines, bodies):
    """
    Implements non negative matrix factorization. Concatenates the resulting head and body vector.
    """
    return topic_models.NMF_topics(headlines, bodies, n_topics=300, include_holdout=True, include_unlbled_test=True,
                                   cosinus_dist=False)

def word_unigrams_5000_concat_tf_l2_holdout(headlines, bodies):
    """
    Simple bag of words feature extraction with term freq of words as feature vectors, length 5000 head + 5000 body,
    concatenation of head and body, l2 norm and bleeding (BoW = train+test+holdout+unlabeled test set).
    """

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]
        return head_and_body

    def get_features(vocab):
        vectorizer_head = TfidfVectorizer(vocabulary=vocab, use_idf=True,
                                          norm="l2", stop_words='english')
        X_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, use_idf=True,
                                          norm="l2", stop_words='english')
        X_body = vectorizer_body.fit_transform(bodies)

        X = np.concatenate([X_head.toarray(), X_body.toarray()], axis=1)

        return X

    # get headlines and bodies of train, test and holdout set
    h, b = word_ngrams.get_head_body_tuples(include_holdout=True)

    # create the vocab out of the BoW
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', max_features=5000, use_idf=False,
                            norm='l2')
    tfidf.fit_transform(combine_head_and_body(h, b))
    vocab = tfidf.vocabulary_

    X = get_features(vocab)

    return X

def word_unigrams_5000_concat_tf_l2_holdout_unlbled_test(headlines, bodies):
    """
    Simple bag of words feature extraction with term freq of words as feature vectors, length 5000 head + 5000 body,
    concatenation of head and body, l2 norm and bleeding (BoW = train+test+holdout+unlabeled test set).
    """

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]
        return head_and_body

    def get_features(vocab):
        vectorizer_head = TfidfVectorizer(vocabulary=vocab, use_idf=True,
                                          norm="l2", stop_words='english')
        X_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, use_idf=True,
                                          norm="l2", stop_words='english')
        X_body = vectorizer_body.fit_transform(bodies)

        X = np.concatenate([X_head.toarray(), X_body.toarray()], axis=1)

        return X

    # get headlines and bodies of train, test and holdout set
    h, b = word_ngrams.get_head_body_tuples(include_holdout=True)

    # add the unlabeled test data words to the BoW of test+train+holdout data
    h_unlbled_test, b_unlbled_test = word_ngrams.get_head_body_tuples_unlbled_test()
    h.extend(h_unlbled_test)
    b.extend(b_unlbled_test)

    # create the vocab out of the BoW
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', max_features=5000, use_idf=True,
                            norm='l2')
    tfidf.fit_transform(combine_head_and_body(h, b))
    vocab = tfidf.vocabulary_

    X = get_features(vocab)

    return X


def stanford_based_verb_noun_sim(headlines, bodies, bodyIds, headIds, order_sentences=False, num_sents=99):
    def load_embeddings(headlines, bodies):
        # embedding parameters:
        embedding_size = 300
        vocab_size = 3000000
        embeddPath = "%s/data/embeddings/google_news/GoogleNews-vectors-negative300.bin.gz" % (
            path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
        embeddData = path.normpath("%s/data/" % (path.dirname(path.abspath(embeddPath))))
        binary_val = True
        embeddings = LoadEmbeddings(filepath=embeddPath, data_path=embeddData, vocab_size=vocab_size,
                                    embedding_size=embedding_size, binary_val=binary_val)
        #     print('Loaded embeddings: Vocab-Size: ' + str(vocab_size) + ' \n Embedding size: ' + str(embedding_size))
        return embedding_size, embeddings

    def clear_unwanted_chars(mystring):
        return str(mystring.encode('latin', errors='ignore').decode('latin'))

    def stanford_helper_order_sents(order_sentences, num_of_sents, body_id, clean_headline, clean_body,
                                    myStanfordmethods, mytf_tf_idf_helpers):
        # Order sentences by tf-idf-score:
        if order_sentences:
            body_id = "ranked_" + str(num_of_sents) + "_" + str(body_id)
            'Only rank sentences, if there is no entry in the StanfordPickle'
            if not myStanfordmethods.check_if_already_parsed(body_id):
                # print(body_id + " is not in stanford_pickle")
                ranked_sentences = mytf_tf_idf_helpers.order_by_tf_id_rank(clean_headline, clean_body, num_of_sents)
            else:
                'In this case the content of ranked sentences does not matter, since the Stanford stored information is used'
                # print(body_id + " is already in stanford_pickle _ skipping tf_idf_ranking")
                ranked_sentences = clean_body
        else:
            ranked_sentences = clean_body
            body_id = "unranked_" + str(body_id)

        return ranked_sentences, body_id

    myStanfordmethods = StanfordMethods()
    mytf_tf_idf_helpers = tf_idf_helpers()

    def calculate_word_sim(embeddings, headline, body, body_id, head_id):
        clean_headline = clear_unwanted_chars(headline)
        clean_body = clear_unwanted_chars(body)

        ranked_sentences, body_id = stanford_helper_order_sents(order_sentences, num_sents, body_id, clean_headline,
                                                                clean_body, myStanfordmethods, mytf_tf_idf_helpers)
        headline_nouns, headline_verbs, head_neg, head_sentiment, head_words_per_sentence = myStanfordmethods.getStanfordInfo(
            'headline', str(body_id), str(head_id), clean_headline, max_number_of_sentences=num_sents)
        body_nouns, body_verbs, body_neg, body_sentiment, body_words_per_sentence = myStanfordmethods.getStanfordInfo(
            'body', str(body_id), str(head_id), ranked_sentences, max_number_of_sentences=num_sents)

        try:
            noun_sim = avg_embedding_similarity(embeddings, embedding_size, ' '.join(headline_nouns),
                                                ' '.join(body_nouns))
        except Exception as e:
            # print(e)
            # print('Problem with nouns for dataset with headline ID: ' + str(body_id) + '\n Headline-text: ' + str(clean_headline))
            # print(body_nouns)
            noun_sim = -1
        if math.isnan(noun_sim):
            # print('NAN for nouns for dataset with headline ID: ' + str(body_id) + '\n Headline-text: ' + str(clean_headline) + '\n \n Body-text: ' + str(ranked_sentences) + ' \n \n Body-verbs: ' + str(body_verbs) + '\n \n Headline-verbs:' + str(headline_verbs))
            # print(body_nouns)
            noun_sim = -1

        try:
            verb_sim = avg_embedding_similarity(embeddings, embedding_size, ' '.join(headline_verbs),
                                                ' '.join(body_verbs))
        except Exception as e:
            # print(e)
            # print('Problem with verbs for dataset with headline ID: ' + str(body_id) + '\n Headline-text: ' + str(clean_headline))
            # print(body_verbs)
            verb_sim = -1

        if math.isnan(verb_sim):
            # print('NAN for verbs for dataset with headline for body ID: ' + str(body_id) + '\n Headline-text: ' + str(clean_headline) + '\n \n Body-text: ' + str(ranked_sentences) + ' \n \n Body-verbs: ' + str(body_verbs) + '\n \n Headline-verbs:' + str(headline_verbs))
            # print(body_verbs)
            verb_sim = -1

        features = []
        features.append(noun_sim)
        features.append(verb_sim)

        return features

    x = []
    embedding_size, embeddings = load_embeddings(headlines, bodies)
    for i, (headline, body, bodyIds, headIds) in tqdm(enumerate(zip(headlines, bodies, bodyIds, headIds))):
        x.append(calculate_word_sim(embeddings, headline, body, bodyIds, headIds))
    # save all information in file
    myStanfordmethods.store_pickle_file()
    return x


def stanford_based_verb_noun_sim_1sent(headlines, bodies, bodyIds, headIds, order_sentences=True, num_sents=1):
    return stanford_based_verb_noun_sim(headlines, bodies, bodyIds, headIds, order_sentences, num_sents)