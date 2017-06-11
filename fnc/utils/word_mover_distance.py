import sys, os, os.path as path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sklearn.metrics import euclidean_distances
from nltk.corpus import stopwords
from fnc.utils.loadEmbeddings import LoadEmbeddings
from fnc.utils.data_helpers import sent2tokens_wostop

stoplist = set(stopwords.words('english'))

def sent2embedd(sentence, emb):
    list_emb=[]
    miss=[]
    words = sent2tokens_wostop(sentence, stoplist)
    for token in words:
        embedding = (emb.word2embedd(token) if emb.isKnown(token) 
                    else emb.word2embedd("unknown"))
        list_emb.append((token,embedding))
        if not emb.isKnown(token): 
            miss.append(token) 
    return list_emb

def wmdistance(sent1_embs, sent2_embs):
    wmd = 0.0
    for _,x in sent1_embs:
        min_dist = sys.float_info.max
        for _,y in sent2_embs:
            x = x.reshape(1, -1)
            y = y.reshape(1, -1)
            distance = euclidean_distances(x,y)
            if distance < min_dist:
                min_dist = distance
        wmd += min_dist
    return - float(wmd) / (len(sent1_embs) + len(sent2_embs))
    
# Note that this breaks the symmetry and is not a distance anymore:
# To overcome this, we compute the average of the score in both side: (weigthedWMD(a,b) + weightedWMD(b,a))/2
def weighted_wmdistance(sent1_embs, sent2_embs, idfs, mean):
    wmd = 0.0
    for token1, x in sent1_embs:
        min_dist = sys.float_info.max
        weight = idfs[token1] if token1 in idfs else mean
        for _, y in sent2_embs:
            print(x, x.shape())
            print(y, y.shape())
            score = weight * euclidean_distances(x,y) 
            exit(0)
            if score < min_dist:
                min_dist = score
        wmd += min_dist
    return - float(wmd) / (len(sent1_embs) + len(sent2_embs))

def computeWMD(embedd, sent1, sent2):
    sent1_embs = sent2embedd(sent1, embedd)
    sent2_embs = sent2embedd(sent2, embedd)
    return wmdistance(sent1_embs, sent2_embs)

def computeAverageWMD(embedd, sent1, sent2):
    sent1_embs = sent2embedd(sent1, embedd)
    sent2_embs = sent2embedd(sent2, embedd)
    return (wmdistance(sent1_embs, sent2_embs) + wmdistance(sent2_embs, sent1_embs))/2.0

def computeWeightedWMD(sent1, sent2, embedd, idfs, mean):
    sent1_embs = sent2embedd(sent1, embedd)
    sent2_embs = sent2embedd(sent2, embedd)
    return (weighted_wmdistance(sent1_embs, sent2_embs, idfs, mean) + weighted_wmdistance(sent2_embs, sent1_embs, idfs, mean)) / 2.

if __name__ == "__main__":
    sent1 = "Barak Obama"
    sent2 = "The President of United States of America"

    data_path = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/data/embeddings"
        
    embeddPath = os.path.normpath("%s/google_news/GoogleNews-vectors-negative300.bin.gz" % (data_path))
    embeddData = os.path.normpath("%s/google_news/data/" % (data_path))
    vocab_size = 3000000
    embedding_size = 300
    
    embeddings = LoadEmbeddings(filepath=embeddPath, data_path=embeddData, vocab_size=vocab_size, embedding_size=embedding_size)
    score = computeAverageWMD(embeddings, sent1, sent2)
    print(score)