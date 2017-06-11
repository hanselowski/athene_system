import os.path as path
import nltk
from fnc.utils.corpus_reader import CorpusReader
from gensim import corpora, models
from fnc.utils.data_helpers import sent2stokens_wostop, text2sent
from gensim.matutils import cossim

class tf_idf_helpers:
    def __init__(self):
        self.generate_tf_idf_corpora()
        self.tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()


    #Utility function to order sentences by tf/idf rank
    def generate_tf_idf_corpora(self):
        data_path = "%s/../data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        reader = CorpusReader(data_path)
        body_dict = reader.load_body("train_bodies.csv")
        bodyText_list = list(body_dict.values())
        bodyIds_index = {k:index for index, k in enumerate(body_dict.keys())}

        bodyText_w = [sent2stokens_wostop(text) for text in bodyText_list]

        self.vocab = corpora.Dictionary(bodyText_w)
        corporaBody_bow = [self.vocab.doc2bow(text) for text in bodyText_w]
        self.tfidf_model = models.TfidfModel(corporaBody_bow)


    def order_by_tf_id_rank(self, headline, sentences, number_of_sentences):
        headline_bow = self.vocab.doc2bow(sent2stokens_wostop(headline))
        headline_tfidf = self.tfidf_model[headline_bow]

        scored_sentences = []
        'Replace newlines with blank, since the punkt tokenizer does not recognize .[newline]'
        #sentences = sentences.replace('\n', ' ')

        for sentence in self.tokenizer.tokenize(sentences):
            sentence_tfidf = self.vocab.doc2bow(sent2stokens_wostop(sentence))
            sim = cossim(headline_tfidf, sentence_tfidf)
            #print(str(sim))
            scored_sentences.append([sentence, sim])

        sorted_sentences= sorted(scored_sentences, key=lambda scored_sentences: scored_sentences[1], reverse= True)
        '''
        for sentence in sorted_sentences:
        print(str(sentence))
        '''
        ' return sorted_sentences '

        sentences_string = ""
        current_sentence_number = 0
        for sentence in sorted_sentences:
            current_sentence_number += 1
            sentences_string += sentence[0] + ' '
            if current_sentence_number == number_of_sentences:
                break
        #print("Ranked: \n " + sentences_string)
        return sentences_string
