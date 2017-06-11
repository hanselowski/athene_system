from pycorenlp import StanfordCoreNLP
import os
import pickle
import nltk
import networkx as nx
#import matplotlib.pyplot as plt

tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
_data_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
_pickled_data_folder = os.path.join(_data_folder, 'pickled')
_stanford_pickle_database_file = 'stanparsed_fnc.pickle'

class StanfordMethods:
    def __init__(self):
        self.webparser = StanfordCoreNLP('http://localhost:9020')
        self.load_pickle_file()
        #To use this parser an instance has to be started in parallel:
        #Download Stanford CoreNLP from: https://stanfordnlp.github.io/CoreNLP/index.html
        #Extract anywhere and execute following command: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9020

    def webparse(self, text):
        return self.webparser.annotate(text, properties={
            'timeout': '500000',
            'annotators': 'tokenize,ssplit,truecase,pos,depparse,parse,sentiment',
            'outputFormat': 'json'
        })

    def load_pickle_file(self):
        try:
            self.known_ids = pickle.load(open(os.path.join(_pickled_data_folder, _stanford_pickle_database_file), 'rb'))
            print('loaded known_ids pickle')
            #print(self.known_ids)
        except:
            print('Stanford pickle does not exist')
            self.known_ids= {}


    def store_pickle_file(self):
        with open(os.path.join(_pickled_data_folder, _stanford_pickle_database_file), 'wb') as f:
            pickle.dump(self.known_ids, f, pickle.HIGHEST_PROTOCOL)


    def getStanfordInfo(self, type, body_id, head_id, text, max_number_of_sentences=99):
        '''
        reads info from file else calculates'
        :param type: either 'body' or 'headline'. type body will be stored, type headline will always be parsed, because it is not unique
        :param body_id: element id of which the stanford information is needed
        :param head_id
        :param text: text of which the stanfordinformation shall be extracted
        :param max_number_of_sentences: number of sentences which shall be parsed at maximum
        :return: [ [nouns], [verbs], [negations count, [root_dist]], [sentiment_value] ]
        '''
        if (type == 'body') and (body_id in self.known_ids):
            return self.known_ids[body_id]
        elif (type == 'headline') and (head_id+body_id in self.known_ids):
            return self.known_ids[str(head_id) + str(body_id)]
        else :
            try:
                result = self.extract_stanford_information(text, max_number_of_sentences)
                if type == 'body':
                    self.known_ids[body_id] = result
                elif type == 'headline':
                    self.known_ids[str(head_id) + str(body_id)] = result
                return result
            except Exception as e:
                self.store_pickle_file()
                print('problem with id: ' + str(body_id) + " type:" + type)
                print(text)
                print(e)
                raise e

    def extract_stanford_information(self, text, max_number_of_sentences=99):
        '''
        Stanford-parse the sentence
         :parameter: text, max_number of sentences to be parsed
         :return: [ [nouns], [verbs], [negations count, [root_dist]], [sentiment_value] ]
        '''
        #since the nlp parser might get some problems with long texts,
        #I decided to divide the text into sentences before parsing it with stanfordparser

        nouns = []
        verbs = []
        # sentiment_list = []
        sentiment_value_list = []
        negation_count = 0
        root_dist = []
        current_sentence = 0
        number_of_words = 0
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

        for raw_sentence in tokenizer.tokenize(text):
            try:
                tagged_text = dict(self.webparse(raw_sentence))
                # Normally only one sentence should be in a raw_Sentence
                #  - but the nltk PunktSentence Tokenizer might have missed a split
                for sentence in tagged_text['sentences']:
                    current_sentence += 1
                    # Extract nouns and verbs
                    sentiment_value_list.append(int(sentence['sentimentValue']))
                    for token in sentence['tokens']:
                        if 'NN' in token['pos']:
                            nouns.append(token['originalText'])
                        elif 'V' in token['pos']:
                            verbs.append(token['originalText'])

                        if token['originalText'] in _refuting_words:
                            negation_count += 1
                            root_dist.append(calculate_distance(sentence, find_root_node(sentence), token['index']))
                    # Count negations
                    '''
                    # This only works correct, when at least on sentence can be parsed per text
                    for dependency in sentence['basicDependencies']['dep']:
                        try:
                           # dep, dependent, dependentGloss, governor, governorGloss = dependency.values()

                            if dependency == 'neg':
                                negation_count += 1
                                #calculate distance to negated words
                                #find head token

                                #find negated token

                                print('Negated token: ' + sentence['tokens'][i-1]['originalText'])

                                distance = calculate_distance(tagged_text, 'not', 'I')
                            # sentiment_list.append(sentence['sentiment'])

                        # Skip sentence if problem occurs while parsing
                        except Exception as e:
                            print('Error parsing sentence: ' + raw_sentence)
                            print(e)
                            #raise e
                            continue
                    '''
                    number_of_words += 1
                #only parse number of given sentences per call
                if current_sentence == max_number_of_sentences:
                    break
            except Exception as e:
                print('Error parsing sentence: ' + raw_sentence)
                print(e)
                #raise e
                continue
        #ToDo: Think about good way to combine the distance
        if negation_count >= 1:
            negation = [negation_count, root_dist]
        else:
            negation = [-1, -1]

        #calculate average words per sentence
        words_per_sentence = number_of_words/current_sentence

        return nouns, verbs, negation, sentiment_value_list, words_per_sentence

    def check_if_already_parsed(self, id):
        return id in self.known_ids

def find_root_node(sentence):
    #tokens = tagged_text['sentences'][0]['tokens']
    for edge in sentence['basicDependencies']:
        if edge['governor'] == 0:
            #print(tokens[edge['dependent']-1]['originalText'])
            return edge['dependent']

# taken from http://stackoverflow.com/questions/32835291/how-to-find-the-shortest-dependency-path-between-two-words-in-python
# changed to compute distance from root to one token
def create_dependency_graph(sentence):
    # Load Stanford CoreNLP's dependency tree into a networkx graph
    edges = []
    dependencies = {}
    for edge in sentence['basicDependencies']:
        edges.append((edge['governor'], edge['dependent']))
        dependencies[(min(edge['governor'], edge['dependent']),
                      max(edge['governor'], edge['dependent']))] = edge

    graph = nx.Graph(edges)
    return graph


def calculate_distance(sentence, token1_index, token2_index):
    graph = create_dependency_graph(sentence)
    path = nx.shortest_path(graph, source=token1_index, target=token2_index)
    '''
    print('path: {0}'.format(path))
    print('shortest path: ' + str(len(path)))

    for token_id in path:
        token = tokens[token_id-1]
        token_text = token['originalText']
        print('Node {0}\ttoken_text: {1}'.format(token_id,token_text))
    '''
    return len(path)

