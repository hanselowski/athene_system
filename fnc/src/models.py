from gensim import corpora, models
from gensim.matutils import cossim
from fnc.utils.data_helpers import sent2stokens_wostop, text2sent
import os.path as path

from fnc.utils.loadEmbeddings import LoadEmbeddings
from fnc.utils.doc2vec import avg_embedding_similarity
from fnc.utils.util_funcs import print_results, create_lists, create_lists_distance_based, print_results_distance_based
from fnc.utils.word_mover_distance import computeAverageWMD

import logging
import warnings

class Model():
    def __init__(self, model_type, embeddPath):
        self.embeddPath = embeddPath
        if embeddPath != '':
            self.embeddData = path.normpath("%s/data/" % (path.dirname(path.abspath(embeddPath))))
        else:
            self.embeddData = ''
        self.model_type = model_type
        self.vocab_size = 3000000
        self.embedding_size = 300

        #logging.basicConfig(level="INFO", format='%(message)s')

        #used to surpress deprecated warning of sklearn:
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    def sentence_ranking(self, train_data, body_dict):
        '''
        Write code here
        Algorithm:
        
        Step 1:
        headline -> avg. word embeddings (Hr) [Headline representation]
        Split the document into sentences -> S1, S2... SN (avg. word embeddings S1r, S2r... SNr) [Document sentence representation]
        
        utils/doc2vec - Check the code for avg. word embeddings
        
        Step 2:
        Rank based on similarity (hr,S1r), (hr, S2r), (hr, SNr)
        
        Example similarity codes:
        utils/doc2vec - Cosine similarity 
        utils/word_mover_distance - Word mover distance   
        
        Step 3:
        Classify as 'Related or Unrelated' based on threshold
        
        Step4:
        Plot related vs unrelated
        '''
    # calculate average sentence vector and compare headline with each sentence, use highest similarity
    def doc2vec_similarity_max(self, train_data, body_dict, threshold):
        '''
        :param
        train_data : a list of training samples of type ['headline', 'bodyID', 'stance']
        body_dict : a dictionary of values containing {bodyID:'bodyText'}
        threshold : used distinguish between similar and not similar
        '''
        # Load embeddings
        logging.info('Load embeddings: Vocab-Size: ' + str(self.vocab_size) + ' Embedding size: ' + str(self.embedding_size))

        embeddings = LoadEmbeddings(filepath=self.embeddPath, data_path=self.embeddData, vocab_size=self.vocab_size, embedding_size=self.embedding_size)

        # Align body-text in workable format
        bodyText_list = list(body_dict.values())
        bodyIds_index = {k:index for index, k in enumerate(body_dict.keys())}

        unrelated, related, y_true, y_pred = [], [], [], []
        sentence_list = []

        for headline, bodyID, stance in train_data:
            logging.info("Headline: " + headline)
            score = 0
            bodyText = bodyText_list[bodyIds_index[bodyID]]
            sentence_list = text2sent(bodyText)
            # logging.info("Bodytext: " + bodyText)

            for sentence in sentence_list:
                #logging.info("Sentence: " + sentence)
                # compare both sentences - vectors not necessary, since this procedure works with text
                # note: avg_embeddings_similarity tokenizes and lemmatizes the sentences prior to calculation, so no pre-assessment is necessary (Sentence to tokens without stopwords)
                temp_score = avg_embedding_similarity(embeddings, self.embedding_size, headline, sentence)
                #logging.info("Similarity: " + str(temp_score))

                # store the highest similarity score
                score=max(score, temp_score)

            # asses headline - body as related or unrelated based on threshold, taken the highest similarity of sentences
            unrelated, related, y_true, y_pred = create_lists(score, stance, threshold, [unrelated, related, y_true, y_pred])


            # following lines just for manual cross-checks
            if score <= threshold:
                calculated_stance = "unrelated"
            else:
                calculated_stance = "related"

            logging.info("Best score for this headline - sentence similarity: " + str(score))
            logging.info("Real/calculated stance: " + stance + " / " + calculated_stance)
        # ToDo: Correctly write and evaluate the results
        print_results([unrelated, related, y_true, y_pred], self.model_type)

    # calculate word_mover_distance
    def word_mover_distance_similarity(self, train_data, body_dict, threshold, type):
        '''
        :param
        train_data : a list of training samples of type ['headline', 'bodyID', 'stance']
        body_dict : a dictionary of values containing {bodyID:'bodyText'}
        threshold : used distinguish between similar and not similar
        type: sentence|wholeText: compute distance per sentence or with whole body text
        '''
        # Load embeddings
        #logging.info('Load embeddings: Vocab-Size: ' + str(self.vocab_size) + ' Embedding size: ' + str(self.embedding_size))

        embeddings = LoadEmbeddings(filepath=self.embeddPath, data_path=self.embeddData,
                                     vocab_size=self.vocab_size, embedding_size=self.embedding_size)

        # Align body-text in workable format
        bodyText_list = list(body_dict.values())
        bodyIds_index = dict((k, index) for index, k in enumerate(list(body_dict.keys())))

        unrelated, related, y_true, y_pred = [], [], [], []
        sentence_list = []

        for headline, bodyID, stance in train_data:
            #logging.info("Headline: " + headline)

            distance = 99999
            bodyText = bodyText_list[bodyIds_index[bodyID]]
            sentence_list = text2sent(bodyText)
            #logging.info("Bodytext: " + bodyText)
            if type == "sentence":
                for sentence in sentence_list:
                    #logging.info("Sentence: " + sentence)
                    temp_distance = abs(computeAverageWMD(embeddings, headline, sentence))

                    # store the lowest distance
                    distance=min(distance, temp_distance)

                    #Note: Distance is not normallized!!
            elif type == "wholeText":
                distance = abs(computeAverageWMD(embeddings, headline, bodyText))

            unrelated, related, y_true, y_pred = create_lists_distance_based(distance, stance, threshold, [unrelated, related, y_true, y_pred])
            if distance <= threshold:
                calculated_stance = "related"
            else:
                calculated_stance = "unrelated"

            #logging.info("Best word_mover_distance for this headline - body combination: " + str(distance))
            #logging.info("Real/calculated stance: " + stance + " / " + calculated_stance)

        # ToDo: Correctly write and evaluate the results
        print_results_distance_based([unrelated, related, y_true, y_pred], self.model_type)

    # calculate average sentence vector and compare with whole body text
    def avg_embed(self, train_data, body_dict, threshold):

        embeddings = LoadEmbeddings(filepath=self.embeddPath, data_path=self.embeddData,
                                     vocab_size=self.vocab_size, embedding_size=self.embedding_size)

        
        bodyText_list = list(body_dict.values())
        bodyIds_index = {k:index for index, k in enumerate(body_dict.keys())}
        
        bodyText_w = [sent2stokens_wostop(text) for text in bodyText_list]

        unrelated, related, y_true, y_pred = [], [], [], []
                        
        for headline, bodyID, stance in train_data:        
            headline_w = sent2stokens_wostop(headline)
            body_w = bodyText_w[bodyIds_index[bodyID]]
            
            sim = avg_embedding_similarity(embeddings, self.embedding_size, ' '.join(headline_w), ' '.join(body_w))
     
            unrelated, related, y_true, y_pred = create_lists(sim, stance, threshold, 
                                                                   [unrelated, related, y_true, y_pred])
        
        print_results([unrelated, related, y_true, y_pred], self.model_type)
    
    def tfidf_sim(self, train_data, body_dict, threshold):
        '''
        :param 
        train_data : a list of training samples of type ['headline', 'bodyID', 'stance']
        body_dict : a dictionary of values containing {bodyID:'bodyText'}
        threshold : used distinguish between similar and not similar
        '''
        bodyText_list = list(body_dict.values())
        bodyIds_index = {k:index for index, k in enumerate(body_dict.keys())}
        
        bodyText_w = [sent2stokens_wostop(text) for text in bodyText_list]
        
        vocab = corpora.Dictionary(bodyText_w)
        corporaBody_bow = [vocab.doc2bow(text) for text in bodyText_w]
        tfidf_model = models.TfidfModel(corporaBody_bow)
        
        unrelated, related, y_true, y_pred = [], [], [], []
        for headline, bodyID, stance in train_data:        
            headline_bow = vocab.doc2bow(sent2stokens_wostop(headline))
            
            headlines_tfidf = tfidf_model[headline_bow]
            corporaBody_tfidf = tfidf_model[corporaBody_bow[bodyIds_index[bodyID]]]
            
            sim = cossim(headlines_tfidf, corporaBody_tfidf)
            unrelated, related, y_true, y_pred = create_lists(sim, stance, threshold, [unrelated, related, y_true, y_pred])
        
        print_results([unrelated, related, y_true, y_pred], self.model_type)      
        
        
    def sdm_sim(self, train_data, body_dict, threshold):
        
        '''
        :param 
        train_data : a list of training samples of type ['headline', 'bodyID', 'stance']
        body_dict : a dictionary of values containing {bodyID:'bodyText'}
        threshold : used distinguish between similar and not similar
        '''
        import retinasdk
        fullClient = retinasdk.FullClient("e8bf8de0-fe52-11e6-b22d-93a4ae922ff1", apiServer="http://api.cortical.io/rest", retinaName="en_associative")
        
        bodyText_list = body_dict.values()
        bodyIds_index = dict((k,index) for index, k in enumerate(body_dict.keys()))

        unrelated, related, y_true, y_pred = [], [], [], []
        cnt1 = 0
        cnt2 = 1
        for headline, bodyID, stance in train_data:        

            comp_with_stop_words = fullClient.compare('[{"text": "'+headline+'"}, {"text": "'+bodyText_list[bodyIds_index[bodyID]]+'"}]')
            sim = comp_with_stop_words.cosineSimilarity
#             sim = comp_with_stop_words.jaccardDistance
            
#             comp_without_stop_words = fullClient.compare('[{"text": "'+' '.join(sent2stokens_wostop(headline))+'"}, {"text": "'+' '.join(sent2stokens_wostop(bodyText_list[bodyIds_index[bodyID]]))+'"}]')
#             sim = comp_without_stop_words.cosineSimilarity
            
            unrelated, related, y_true, y_pred = create_lists(sim, stance, threshold, [unrelated, related, y_true, y_pred])
            
            # keep track of the processed examples
            if (cnt1 == 100):
                print(cnt2*100)
                cnt2 += 1
                cnt1 = 0
            cnt1 += 1

            
        print_results([unrelated, related, y_true, y_pred], self.model_type)   

    def model_train(self, train_data, body_dict):
        if self.model_type == 'ranking':
            self.sentence_ranking(train_data, body_dict)    
            
    def analyze_data(self, train_data, body_dict, threshold = 0.1):
        logging.info('Model-type: ' + self.model_type)
        if self.model_type == 'tf_idf':
            self.tfidf_sim(train_data, body_dict, threshold) 
        if self.model_type == 'ranking':
            self.sentence_ranking(train_data, body_dict)
        if self.model_type == 'avg_embed':
            self.avg_embed(train_data, body_dict, threshold)
        if self.model_type == 'doc2vec':
            self.doc2vec_similarity_max(train_data, body_dict, threshold)
            #self.doc2vec_similarity_max(train_data, body_dict, threshold = 0.67, embeddPath= embeddPath)
        if self.model_type == 'word_mover_sentence':
            self.word_mover_distance_similarity(train_data, body_dict, threshold, type="sentence")
            #self.word_mover_distance_similarity(train_data, body_dict, threshold = 1.2, embeddPath= embeddPath, type="sentence")
        if self.model_type == 'word_mover_wholeText':
            self.word_mover_distance_similarity(train_data, body_dict, threshold, type="wholeText")
            #self.word_mover_distance_similarity(train_data, body_dict, threshold = 1.32, embeddPath= embeddPath, type="wholeText")
        if self.model_type == 'sdm':
            self.sdm_sim(train_data, body_dict, threshold)
