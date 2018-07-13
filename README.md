

![2010-07-07_ukp_banner](https://user-images.githubusercontent.com/29311022/27184688-27629126-51e3-11e7-9a23-276628da2430.png)

![aiphes_logo - small](https://user-images.githubusercontent.com/29311022/27278631-2e19f99e-54e2-11e7-919c-f89ae0c90648.png)
![tud_weblogo](https://user-images.githubusercontent.com/29311022/27184769-65c6583a-51e3-11e7-90e0-12a4bdf292e2.png)



## Introduction

The repository was developed as a part of the Fake News Challenge Stage 1 (FNC-1 http://www.fakenewschallenge.org/) by team Athene:
[Andreas Hanselowski](mailto:hanselowski@aiphes.tu-darmstadt.de), [Avinesh PVS](mailto:avinesh@aiphes.tu-darmstadt.de), [Benjamin Schiller](mailto:schiller.benny@googlemail.com) and Felix Caspelherr.
In the project, we worked in close collaboration with [Debanjan Chaudhuri](mailto:deba.kgec@gmail.com). 

Our new paper in COLING 2018: [A Retrospective Analysis of the Fake News Challenge Stance Detection Task](https://arxiv.org/pdf/1806.05180.pdf)

Our [Blog Post](https://medium.com/@andre134679/team-athene-on-the-fake-news-challenge-28a5cf5e017b) on the Fake News Challenge.




Prof. Dr. Iryna Gurevych, AIPHES-Ubiquitous Knowledge Processing (UKP) Lab, TU-Darmstadt, Germany

## Requirements

* Software dependencies
	
		python >= 3.4.0 (tested with 3.4.0)
	
## Installation

1. Install required python packages.

		python3.4 -m pip install -r requirements.txt --upgrade
        
2. In order to reproduce the the results of our best submission to the FNC-1, please go to [Athene_FNC-1 Google Drive](https://drive.google.com/drive/folders/0B0-muIdcdTp7cUhVdFFqRHpEcVk?usp=sharing) and download the [features.zip](https://drive.google.com/open?id=0B0-muIdcdTp7UWVyU0duSDRUd3c) and [model.zip](https://drive.google.com/open?id=0B0-muIdcdTp7Sm42ZW1yUndyY1E) and unzip them in respective folders.

		unzip  features.zip athene_system/data/fnc-1/features
		unzip  model.zip athene_system/data/fnc-1/mlp_models
        
3. Parts of the Natural Language Toolkit (NLTK) might need to be installed manually.

		python3.4 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
	      
4. Copy Word2Vec [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) in folder athene_system/data/embeddings/google_news/ 

5. Download [Paraphrase Database: Lexical XL Paraphrases 1.0](http://www.cis.upenn.edu/~ccb/ppdb/release-1.0/ppdb-1.0-xl-lexical.gz) and extract it to the ppdb folder.
	
		gunzip ppdb-1.0-xl-lexical.gz athene_system/data/ppdb/
        
6. To use the Stanford-parser an instance has to be started in parallel: Download [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html), extract anywhere and execute following command: 

		wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
		java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9020

## Additional notes

* In order to reproduce the classification results of the best submission at the day of the FNC-1, it is mandatory to use tensorflow v0.9.0 (ideally GPU version) and the exact library versions stated in requirements.txt, including python 3.4.

* Setup tested on Anaconda3 (tensorflow 0.9 gpu version)*

		conda create -n env_python3.4 python=3.4 anaconda
	
		source activate env_python3.4
	
		env_python3.4/bin/python3.4 -m pip install -r requirements.txt --upgrade
	
		env_python3.4/bin/python3.4 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0rc0-cp34-cp34m-linux_x86_64.whl

	
## To Run

To run the pre trained model and test

	python pipeline.py -p ftest


For more details
		
	python pipeline.py --help         
        
        e.g.: python pipeline.py -p crossv holdout ftrain ftest
        
        * crossv: runs 10-fold cross validation on train / validation set and prints the results
        * holdout: trains classifier on train and validation set, tests it on holdout set and prints the results
        * ftrain: trains classifier on train/validation/holdout set and saves it to athene_systems/data/fnc-1/mlp_models
        * ftest: predicts stances of unlabeled test set based on the model (see Installation, step 2) 

 After _ftest_ was executed, the labeled stances will be saved to disk:
 
 	cat athene_system/data/fnc-1/fnc_results/submission.csv


## System description

A more detailed description of the system including the features, which have been used, can be found in the document: [system_description_athene.pdf](https://github.com/hanselowski/athene_system/edit/master/system_description_athene.pdf)
   
