## Requirements

* Software dependencies
	* python >= 3.4 (tested with 3.4.0)



## Installation

1. Install required python packages.

        pip install -r requirements.txt
        
2. Parts of the Natural Language Toolkit (NLTK) might need to be installed manually.

	    python -c "import nltk; nltk.download("stopwords"); nltk.download("punkt"); nltk.download("wordnet")"
	
3. Installation of the packages for anaconda3 (python 3)

	e.g.: python3.4 -m pip $i install nltk
	      python3.4 -m pip $i install -r requirements.txt
	      
4. Copy Word2Vec GoogleNews-vectors-negative300.bin.gz in folder fn_classifier/data/embeddings/google_news/ 
    (download link: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)

5. Download http://www.cis.upenn.edu/~ccb/ppdb/release-1.0/ppdb-1.0-xl-lexical.gz
        Extract it in folder fn_classifier/data/ppdb/
        
6. To use the Stanford-parser an instance has to be started in parallel:
        Download Stanford CoreNLP from: https://stanfordnlp.github.io/CoreNLP/index.html
        Extract anywhere and execute following command: 
            java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9020

## Additional notes

* In order to reproduce the classification results of the best submission at the day of the FNC-1, it is mandatory to use tensorflow v0.9.0 (ideally GPU version)
and the exact library versions stated in requirements.txt, including python 3.4.
	
## To Run

1. python pipeline.py --help for more details
    
        python pipeline.py --pipeline_type=train