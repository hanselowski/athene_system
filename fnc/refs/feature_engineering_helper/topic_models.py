from sklearn.externals import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import fnc.refs.feature_engineering_helper.word_ngrams as word_ngrams
from sklearn.metrics.pairwise import cosine_distances
from time import time
from gensim import corpora, models
import nltk
import os
import os.path as path
import pickle



def latent_dirichlet_allocation_cos(headlines, bodies, n_topics=25, include_holdout=False,
                                    use_idf=False, term_freq=True, incl_unlbled_test = False):
    """
    Sklearn LDA implementation based on the 5000 most important words (based on train+test data's term freq => bleeding).
    Returns feature vector of cosinus distances between the topic models of headline and bodies.

    Links:
        https://pypi.python.org/pypi/lda, bottom see suggestions like MALLET, hca
        https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
        https://www.quora.com/What-are-the-best-features-to-put-into-Latent-Dirichlet-Allocation-LDA-for-topic-modeling-of-short-text
    """

    # TODO check https://people.cs.umass.edu/~wallach/posters/bbow.pdf
    # TODO check with bigrams, too
    # TODO try to use embeddings glove / word2vec


    def print_top_words(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic #%d:" % topic_idx)
            print(", ".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                     enumerate(zip(headlines, bodies))]

        return head_and_body

    def get_features(vocab):
        vectorizer_head = TfidfVectorizer(vocabulary=vocab, use_idf=False, norm='l2')
        X_train_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, use_idf=False, norm='l2')
        X_train_body = vectorizer_body.fit_transform(bodies)

        # calculates n most important topics of the bodies. Each topic contains all words but ordered by importance. The
        # more important topic words a body contains of a certain topic, the higher its value for this topic
        lda_body = LatentDirichletAllocation(n_topics=n_topics, learning_method='online', random_state=0, n_jobs=3)

        print("latent_dirichlet_allocation_cos: fit and transform body")
        t0 = time()
        lda_body_matrix = lda_body.fit_transform(X_train_body)
        print("done in %0.3fs." % (time() - t0))

        print("latent_dirichlet_allocation_cos: transform head")
        # use the lda trained for body topcis on the headlines => if the headlines and bodies share topics
        # their vectors should be similar
        lda_head_matrix = lda_body.transform(X_train_head)

        #print_top_words(lda_body, vectorizer_body.get_feature_names(), 100)

        print('latent_dirichlet_allocation_cos: calculating cosine distance between head and body')
        # calculate cosine distance between the body and head
        X = []
        for i in range(len(lda_head_matrix)):
            X_head_vector = np.array(lda_head_matrix[i]).reshape((1, -1)) #1d array is deprecated
            X_body_vector = np.array(lda_body_matrix[i]).reshape((1, -1))
            cos_dist = cosine_distances(X_head_vector, X_body_vector).flatten()
            X.append(cos_dist.tolist())
        return X

    if incl_unlbled_test == True:
        h, b = word_ngrams.get_head_body_tuples(include_holdout=True)
        h_test, b_test = word_ngrams.get_head_body_tuples_unlbled_test()

        h.extend(h_test)
        b.extend(b_test)

        tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', max_features=5000, use_idf=use_idf,
                                norm='l2')
        tfidf.fit_transform(combine_head_and_body(h, b))
        vocab = tfidf.vocabulary_
    else:
        vocab = word_ngrams.create_word_ngram_vocabulary(ngram_range=(1, 1), max_features=5000, lemmatize=False, term_freq=term_freq,
                                         norm='l2', include_holdout=include_holdout, use_idf=use_idf)
    X = get_features(vocab)
    return X

def latent_dirichlet_allocation_gensim_cos(headlines, bodies, n_topics=50):
    """
    In cotrast to the implemented Sklearn's LDA method, this one loads ALL the words of train+test data,
    not just 5000 out of the Vectorizer => bleeding.

    Links:
        http://stackoverflow.com/questions/20349958/understanding-lda-implementation-using-gensim
        https://radimrehurek.com/gensim/models/ldamodel.html#gensim.models.ldamodel.LdaModel
        https://rare-technologies.com/multicore-lda-in-python-from-over-night-to-over-lunch/
        https://radimrehurek.com/gensim/tut1.html
    """

    def combine_and_tokenize_head_and_body(headlines, bodies):
        head_and_body = [nltk.word_tokenize(headline + " " + body) for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]

        return head_and_body

    def get_features(n_topics):
        features_dir = "%s/data/fnc-1/features" % (path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__))))))

        filename = "lda_gensim_cos_" + str(n_topics) + "topics"
        if (os.path.exists(features_dir + "/" + filename + ".pkl")):
            lda = models.LdaMulticore.load(features_dir + "/" + filename + ".pkl")
            dictionary = corpora.Dictionary.load(features_dir + "/" + filename + ".dict")
            print("latent_dirichlet_allocation_gensim_cos model found and load")
        else:
            print("Creating new latent_dirichlet_allocation_gensim_cos model")
            h, b = word_ngrams.get_head_body_tuples()
            head_and_body = combine_and_tokenize_head_and_body(h, b)
            dictionary = corpora.Dictionary(head_and_body)
            dictionary.save(features_dir + "/" + filename + ".dict")
            corpus = [dictionary.doc2bow(text) for text in head_and_body]
            print(dictionary)

            lda = models.LdaMulticore(corpus, id2word=dictionary, num_topics=n_topics, workers=1)
            lda.save(features_dir + "/" + filename + ".pkl")

        X = []
        for i in range(len(headlines)):
            X_head_vector = lda[dictionary.doc2bow(nltk.word_tokenize(headlines[i]))]
            X_body_vector = lda[dictionary.doc2bow(nltk.word_tokenize(bodies[i]))]

            # calculate zero padded vector for cosinus distance
            X_head_vector_filled = np.zeros(n_topics, dtype=np.double)
            for id, prob in X_head_vector:
                X_head_vector_filled[id] = prob

            X_body_vector_filled = np.zeros(n_topics, dtype=np.double)
            for id, prob in X_body_vector:
                X_body_vector_filled[id] = prob

            # reshape for sklearn
            X_head_vector_filled_reshaped = np.array(X_head_vector_filled).reshape((1, -1))  # 1d array is deprecated
            X_body_vector_filled_reshaped = np.array(X_body_vector_filled).reshape((1, -1))

            cos_dist = cosine_distances(X_head_vector_filled_reshaped, X_body_vector_filled_reshaped).flatten()
            X.append(cos_dist.tolist())

        return X

    n_topics = n_topics
    X = get_features(n_topics)

    return X

def latent_semantic_indexing_gensim_concat(headlines, bodies, n_topics=50, include_holdout=False, include_unlbled_test=False):
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

    def combine_and_tokenize_head_and_body(headlines, bodies, file_path=None):
        temp_list = []
        temp_list.extend(headlines)
        temp_list.extend(bodies)
        if file_path != None and (os.path.exists(file_path)):
            with open(file_path, 'rb') as handle:
                return pickle.load(handle)

        head_and_body_tokens = [nltk.word_tokenize(line) for line in temp_list]  # TODO remove stopwords?

        if file_path != None:
            with open(file_path, 'wb') as handle:
                pickle.dump(head_and_body_tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return head_and_body_tokens

    def get_features(n_topics):
        features_dir = "%s/data/fnc-1/features" % (path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__))))))

        filename = "lsi_gensim_concat_" + str(n_topics) + "topics"
        if include_holdout == True:
            filename += "_holdout"
        if include_unlbled_test == True:
            filename += "_unlbled_test"

        h, b = word_ngrams.get_head_body_tuples(include_holdout=include_holdout)

        if include_unlbled_test == True:
            h_unlbled_test, b_unlbled_test = word_ngrams.get_head_body_tuples_unlbled_test()
            h.extend(h_unlbled_test)
            b.extend(b_unlbled_test)

        head_and_body = combine_and_tokenize_head_and_body(h, b,
                                                           file_path=features_dir + "/" + filename + ".tokens")

        if (os.path.exists(features_dir + "/" + filename + ".dict")):
            print("latent_semantic_indexing_gensim_concat: dict found and load")
            dictionary = corpora.Dictionary.load(features_dir + "/" + filename + ".dict")
        else:
            print("latent_semantic_indexing_gensim_concat: create new dict")
            dictionary = corpora.Dictionary(head_and_body)
            dictionary.save(features_dir + "/" + filename + ".dict")

        if (os.path.exists(features_dir + "/" + filename + ".lsi")):
            print("latent_semantic_indexing_gensim_concat: found lsi model")
            lsi = models.LsiModel.load(features_dir + "/" + filename + ".lsi")
        else:
            print("latent_semantic_indexing_gensim_concat: build corpus and tfidf corpus")
            corpus = [dictionary.doc2bow(text) for text in head_and_body]
            tfidf = models.TfidfModel(corpus)  # https://stackoverflow.com/questions/6287411/lsi-using-gensim-in-python
            corpus_tfidf = tfidf[corpus]

            print("latent_semantic_indexing_gensim_concat: create new lsi model")
            lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics)
            lsi.save(features_dir + "/" + filename + ".lsi")

        # get tfidf corpus of head and body
        corpus_train = [dictionary.doc2bow(text) for text in combine_and_tokenize_head_and_body(headlines, bodies)]
        tfidf_train = models.TfidfModel(corpus_train)
        corpus_train_tfidf = tfidf_train[corpus_train]

        corpus_lsi = lsi[corpus_train_tfidf]

        X_head = []
        X_body = []
        i = 0
        for doc in corpus_lsi:
            if i < int(len(corpus_lsi) / 2):
                X_head_vector_filled = np.zeros(n_topics, dtype=np.float64)
                for id, prob in doc:
                    X_head_vector_filled[id] = prob
                X_head.append(X_head_vector_filled)
            else:
                X_body_vector_filled = np.zeros(n_topics, dtype=np.float64)
                for id, prob in doc:
                    X_body_vector_filled[id] = prob
                X_body.append(X_body_vector_filled)
            i += 1

        X = np.concatenate([X_head, X_body], axis=1)

        return X

    n_topics = n_topics
    X = get_features(n_topics)

    return X

def NMF_topics(headlines, bodies, n_topics=300, include_holdout=False, include_unlbled_test = False, cosinus_dist=True):
    """
        Implements non negative matrix factorization. Calculates the cos distance between the resulting head and body vector or
        just concatenates them.

        Links:
            http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-topics-extraction-with-nmf-lda-py
            https://pypi.python.org/pypi/lda on bottom see suggestions like MALLET, hca
            https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
            https://www.quora.com/What-are-the-best-features-to-put-into-Latent-Dirichlet-Allocation-LDA-for-topic-modeling-of-short-text
        """

    # TODO check https://people.cs.umass.edu/~wallach/posters/bbow.pdf
    # TODO use bigrams, too
    # TODO use topics extracted with glove / word2vec
    # TODO use wikipeadia corpus?!

    features_dir = "%s/data/fnc-1/features" % (
    path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__))))))

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]

        return head_and_body

    def get_all_data(head_and_body, filename):
        if not (os.path.exists(features_dir + "/" + filename + ".vocab")):
            vectorizer_all = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', use_idf=True, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            vocab = vectorizer_all.vocabulary_
            print("NMF_topics: complete vocabulary length=" + str(len(list(vocab.keys()))))

            with open(features_dir + "/" + filename + ".vocab", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return X_all, vocab
        else:
            with open(features_dir + "/" + filename + ".vocab", 'rb') as handle:
                vocab = pickle.load(handle)
            vectorizer_all = TfidfVectorizer(vocabulary=vocab, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            return X_all, vectorizer_all.vocabulary_

    def get_vocab(head_and_body, filename):
        if not (os.path.exists(features_dir + "/" + filename + ".vocab")):
            vectorizer_all = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', use_idf=True, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            vocab = vectorizer_all.vocabulary_
            print("NMF_topics: complete vocabulary length=" + str(len(X_all[0])))

            with open(features_dir + "/" + filename + ".vocab", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return vocab
        else:
            with open(features_dir + "/" + filename + ".vocab", 'rb') as handle:
                return pickle.load(handle)

    def get_features(head_and_body):
        filename = "NMF_topics" + str(n_topics) + "topics"

        if include_holdout == True:
            filename += "_holdout"

        if include_unlbled_test == True:
            filename += "unlbled_test"

        if not (os.path.exists(features_dir + "/" + filename + ".pkl")):
            X_all, vocab = get_all_data(head_and_body, filename)

            # calculates n most important topics of the bodies. Each topic contains all words but ordered by importance. The
            # more important topic words a body contains of a certain topic, the higher its value for this topic
            nfm = NMF(n_components=n_topics, random_state=1, alpha=.1)

            print("NMF_topics: fit and transform body")
            t0 = time()
            nfm.fit_transform(X_all)
            print("done in %0.3fs." % (time() - t0))

            with open(features_dir + "/" + filename + ".pkl", 'wb') as handle:
                joblib.dump(nfm, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            vocab = get_vocab(head_and_body, filename)
            with open(features_dir + "/" + filename + ".pkl", 'rb') as handle:
                nfm = joblib.load(handle)

        vectorizer_head = TfidfVectorizer(vocabulary=vocab, norm='l2')
        X_train_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, norm='l2')
        X_train_body = vectorizer_body.fit_transform(bodies)

        print("NMF_topics: transform head and body")
        # use the lda trained for body topcis on the headlines => if the headlines and bodies share topics
        # their vectors should be similar
        nfm_head_matrix = nfm.transform(X_train_head)
        nfm_body_matrix = nfm.transform(X_train_body)

        if cosinus_dist == False:
            return np.concatenate([nfm_head_matrix, nfm_body_matrix], axis=1)
        else:
            # calculate cosine distance between the body and head
            X = []
            for i in range(len(nfm_head_matrix)):
                X_head_vector = np.array(nfm_head_matrix[i]).reshape((1, -1))  # 1d array is deprecated
                X_body_vector = np.array(nfm_body_matrix[i]).reshape((1, -1))
                cos_dist = cosine_distances(X_head_vector, X_body_vector).flatten()
                X.append(cos_dist.tolist())
            return X


    h, b = word_ngrams.get_head_body_tuples(include_holdout=include_holdout)
    if include_unlbled_test == True:
        h_unlbled_test, b_unlbled_test = word_ngrams.get_head_body_tuples_unlbled_test()
        h.extend(h_unlbled_test)
        b.extend(b_unlbled_test)
    head_and_body = combine_head_and_body(h, b)

    X = get_features(head_and_body)

    return X