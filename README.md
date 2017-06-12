## Introduction

The project was developed for the Fake News Challenge One (FNC-1) http://www.fakenewschallenge.org/ by team Athene:
Andreas Hanselowski, Avinesh PVS, Benjamin Schiller and Felix Caspelherr

Ubiquitous Knowledge Processing (UKP) Lab, TU-Darmstadt, Germany


## Requirements

* Software dependencies
	* python >= 3.4 (tested with 3.4.0)



## Installation

1. Install required python packages.

        pip install -r requirements.txt
        
2. To reproduce the the results of our best submission to the FNC-1, please go to 
https://drive.google.com/drive/folders/0B0-muIdcdTp7cUhVdFFqRHpEcVk?usp=sharing  and download 
     the files features.zip and model.zip.
     
     * features.zip has to unzipped into folder athene_system/data/fnc-1/features
     * model.zip has to be unzipped into folder athene_system/data/fnc-1/mlp_models
        
3. Parts of the Natural Language Toolkit (NLTK) might need to be installed manually.

	    python3.4 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
	
4. Installation of the packages for anaconda3 (python 3)

	e.g.: python3.4 -m pip $i install nltk
	      python3.4 -m pip $i install -r requirements.txt
	      
5. Copy Word2Vec GoogleNews-vectors-negative300.bin.gz in folder athene_system/data/embeddings/google_news/ 
    (download link: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)

6. Download http://www.cis.upenn.edu/~ccb/ppdb/release-1.0/ppdb-1.0-xl-lexical.gz
        Extract it in folder athene_system/data/ppdb/
        
7. To use the Stanford-parser an instance has to be started in parallel:
        Download Stanford CoreNLP from: https://stanfordnlp.github.io/CoreNLP/index.html
        Extract anywhere and execute following command: 
            java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9020

## Additional notes

* In order to reproduce the classification results of the best submission at the day of the FNC-1, it is mandatory to use tensorflow v0.9.0 (ideally GPU version)
and the exact library versions stated in requirements.txt, including python 3.4.
	
## To Run

1. python pipeline.py --help for more details
    
        python pipeline.py --pipeline_type=train

    The classifier and corresponding features (Installation, step 2) will be loaded and a prediction 
    will be executed on the unlabled test stances. After the process has finished, the submission.csv can
    be found at athene_system/data/fnc-1/fnc_results
