import sys,os.path as path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
import pandas as pd
import numpy as np
from fnc.utils.data_helpers import text_normalization

class CorpusReader(object):
    def __init__(self, data_path):
        self.data_path = data_path

    def load_body(self, filename):
        '''
        Load body dataframe to a dictionary
        
        filename: Name of the csv file
        return: dictionary{ bodyId: bodyText}
        '''
        
        #FIELDNAMES = ['Body ID', 'Body']
        filepath = "%s/%s" % (self.data_path, filename)
        
        # Load the body data
        bodiesDF = pd.read_csv(filepath)
        bodyIds = bodiesDF['Body ID']
        bodyTexts = bodiesDF['articleBody']
        
        bodyTexts = np.array(bodyTexts)
        bodyIds = np.array(bodyIds)

        bodyTexts = [text_normalization(text) for text in bodyTexts]
               
        return dict(zip(bodyIds, bodyTexts))
    
    def load_dataset(self, filename):
        #FIELDNAMES = ['Headline', 'Body ID', 'Stance']
        filepath = "%s/%s" % (self.data_path, filename)
        stanceDF = pd.read_csv(filepath)
        headlines = stanceDF['Headline']
        bodyIds = stanceDF['Body ID']
        stance = stanceDF['Stance']
        
        headlines = [text_normalization(headline) for headline in headlines]
        
        data = []
        for i, headline in enumerate(headlines):
            data.append([headline, bodyIds[i], stance[i]])

        return data 
        