import re, string
from nltk.tokenize import word_tokenize
from nltk import ngrams
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

PUNCT = tuple(string.punctuation)
stoplist = set(stopwords.words('english')).union(set(PUNCT))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def text_normalization(text):
    text = re.sub(u"[^A-Za-z0-9(),!?\'\`:/\-\.]", u" ", text)
    text = re.sub(u'&#8217;',u"'",text)
    text = re.sub(u'\u2019',u"'",text)
    text = re.sub(u'&#160;',u" ",text)
    text = re.sub(u'&#8211;',u"-",text)
    text = re.sub(u'\u000a',u"",text)
    text = re.sub(u'\u2014',u"-",text)
    text = re.sub(u'\n+$',u" ",text)
    text = re.sub(u'\s+',u" ",text)
    text = re.sub(u'^\s+',u"",text)
    text = re.sub(u'\s+$',u"",text)
    text = re.sub(u"\'s", u" \'s", text)
    text = re.sub(u"\'ve", u" \'ve", text)
    text = re.sub(u"n\'t", u" n\'t", text)
    text = re.sub(u"\'re", u" \'re", text)
    text = re.sub(u"\'d", u" \'d", text)
    text = re.sub(u"\'ll", u" \'ll", text)
    text = re.sub(u",", u" , ", text)
    text = re.sub(u"!", u" ! ", text)
    text = re.sub(u"\(", u" ( ", text)
    text = re.sub(u"\)", u" ) ", text)
    text = re.sub(u"\?", u" ? ", text)
    text = re.sub(u"\.{2,}",u" . ", text)
    text = re.sub(u":[^/]",u" : ", text)
    text = re.sub(u"\s{2,}",u" ", text)
    return text
        
def extract_ngrams(text, stemmer, N):
    '''
    Parameter Arguments:
    text: 'Ney York is a city. It has a huge population.'
    N: Length of the n-grams e.g. 1, 2
    
    return: a list of n-grams
    [('new', 'york'), ('york', 'is'), ('is', 'a'), ('a', 'city'), (city, '.'), 
    ('it', 'has'), ('has','a'), ('a', 'huge'), ('huge', 'population') , ('population', '.')]
    '''
    ngrams_list = []
    ngram_items = list(ngrams(sent2stokens(text, stemmer), N))
    for i, ngram in enumerate(ngram_items):
        ngram_str = ' '.join(ngram)
        ngrams_list.append(ngram_str)
    return ngrams_list

def sent2stokens(text, stemmer, language='english'):
    '''
    Sentence to stemmed tokens
    Parameter arguments:
    sent = a unicode string e.g. sent = '... The boys are playing'
    
    return:
    list of stemmed tokens
    ['the', 'boy', 'are', 'play', '.']
    '''
    return [lemmatizer.lemmatize(word) for word in word_tokenize(text.lower())]
    
def sent2stokens_wostop(text, language='english'):
    '''
    Sentence to lemmatized tokens without stopwords
    Parameter arguments:
    sent = a unicode string e.g. sent = '... The boys are playing'
    
    return:
    list of stemmed tokens without stop words
    ['boys', 'playing']
    '''
    return [lemmatizer.lemmatize(word) for word in word_tokenize(text.lower()) if word not in stoplist]    
    
def sent2tokens_wostop(text, language='english'):
    '''
    Sentence to tokens without stopwords
    Parameter arguments:
    sent = a unicode string e.g. sent = '... The boys are playing'
    
    return:
    list of stemmed tokens without stop words
    ['boys', 'playing']
    '''
    return [word for word in word_tokenize(text.lower()) if word not in stoplist]

def sent2tokens(text, language='english'):
    '''
    Sentence to stemmed tokens
    Parameter arguments:
    words = list of words e.g. sent = '... The boy is playing.'
    
    return:
    list of tokens
    ['the', 'boy', 'is', 'playing','.']
    '''
    return word_tokenize(text.lower())

def text2sent(text, language='english'):
    '''
    Converts a text to sentences
    :param text: Textstring containing several sentences
    :param language:
    :return: sentence_tokenized text
    '''
    return [sentence for sentence in sent_tokenize(text.lower(), language)]

def normalize_word(w):
    return lemmatizer.lemmatize(w).lower()

def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in word_tokenize(s)]

def get_stem(w):
    return stemmer.stem(w)


