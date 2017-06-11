import sys, os, os.path as path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from nltk.corpus import stopwords
from fnc.utils.loadEmbeddings import LoadEmbeddings
from fnc.utils.data_helpers import sent2tokens_wostop
from scipy import spatial
import numpy as np
stoplist = set(stopwords.words('english'))

def avg_feature_vector(sent, model, num_features):
        #function to average all words vectors in a given paragraph
        words = sent2tokens_wostop(sent, stoplist)
        featureVec = np.zeros((num_features,), dtype="float32")
        nwords = len(words)

        for word in words:
            if word.isdigit():
                continue
            if model.isKnown(word):
                featureVec = np.add(featureVec, model.word2embedd(word))
            else:
                featureVec = np.add(featureVec, model.word2embedd(u"unknown"))

        if(nwords>0):
            featureVec = np.divide(featureVec, nwords)
        return featureVec

def avg_embedding_similarity(embeddings, embedding_size, sent1, sent2):
    #print("Calculating similarity for: " + sent1 + "\n and\n" + sent2)
    v1 = avg_feature_vector(sent1, model=embeddings, num_features=embedding_size)
    v2 = avg_feature_vector(sent2, model=embeddings, num_features=embedding_size)
    cosine_distance = spatial.distance.cosine(v1, v2)
    score =  1 - cosine_distance
    #print("Score = " + str(score))
    return score

if __name__ == "__main__":
    sent1 = "United States of America"
    sent2 = "USA"
    data_path = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/data/embeddings"
        
    embeddPath = os.path.normpath("%s/google_news/GoogleNews-vectors-negative300.bin.gz" % (data_path))
    embeddData = os.path.normpath("%s/google_news/data/" % (data_path))
    vocab_size = 3000000
    embedding_size = 300
    
    embeddings = LoadEmbeddings(filepath=embeddPath, data_path=embeddData, vocab_size=vocab_size, embedding_size=embedding_size)
    score = avg_embedding_similarity(embeddings, embedding_size, sent1, sent2)
    print(score)
