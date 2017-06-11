WARNING: Contains lengthy instructions on how to run some code.

This repository contains the source code for my MSc Project: "For or Against? Assessing the evidence for news headline claims". The code is written in Python 2.7 and makes use of a number of external libraries, such as pandas, sklearn, gensim, munkres and others. To run the code from scratch, I suggest:

1. cloning the project in the normal way, i.e issuing the command: 
      
      **git clone https://github.com/willferreira/mscproject.git**, at the command prompt

2. creating a new folder, called *data*,  in the top directory of the project
3. copying the contents (folders and files) from this dropbox link to the new *data* folder: https://www.dropbox.com/sh/9t7fd7xfahb0e1v/AACtdXhZmaTU9QgxZ8jL5tyVa?dl=0
(or from this google link if dropbox doesn't work for you: https://drive.google.com/folderview?id=0BwPdBcatuO0vYTAxSnA1d09qdGM&usp=sharing)
4. installing the excellent anaconda distribution of Python 2.7 from continuum.io, available here: http://continuum.io/downloads 
5. creating a new Python virtual environment, by issuing the following command at the prompt:

      **conda create -n XXX anaconda python=2.7** 
   
   (replacing XXX with whatever you want to call the environment, e.g. mscproject_py27)
6. activating the new virtual environment issuing the following command at the prompt:

      **source activate XXX** 
      
   (replacing XXX with whatever you called your environment)
7. installing package: repoze.lru (provides a function memoize decorator) by issuing the following command at the 
   prompt (accept whatever package updates it proposes):

      **conda install repoze.lru**
      
8. installing package: gensim (provides a word2vec library) by issuing the following command at the prompt 
   (accept whatever package updates it proposes):

      **conda install gensim**
      
9. installing package: munkres 1.0.7 (provides an implementation of the Hungarian Algorithm, used for word alignment) by:
    1. downloading the package from https://pypi.python.org/pypi/munkres/
    2. unzipping the file somewhere
    3. cd munkres-1.0.7
    4. issuing the command: **python setup.py install**, at the prompt

You should now have all you need to run the code. The relevant scripts are in the project bin/ directory. From there you can run the following:

**python run_train_test.py**

    - trains the model on the EmergentLite training data-set, and then runs the trained model on the test data-set. 
      All the features are used in the model, namely: Q,BoW,AlgnW2V,AlgnPPDB,RootDist,NegAlgn,SVO. 
      The output should look something like this:
      
      Feature set: ['Q', 'BoW', 'AlgnW2V', 'AlgnPPDB', 'RootDist', 'NegAlgn', 'SVO']
      >> Training classifier <<
      >> Classifying test data <<
      
      Confusion matrix:
      =================
                 for  against  observing
      for        219        3         24
      against     15       64         12
      observing   75       11        101
      
      Measures:
      =========
      accuracy: 0.7328
      
      Per class:
                  accuracy  precision     recall         F1
      for        0.7767176  0.7087379  0.8902439  0.7891892
      against    0.9217557  0.8205128  0.7032967  0.7573964
      observing  0.7671756  0.7372263   0.540107  0.6234568
      
**python run_train_test.py -i**

      As above, but the features are added incrementally, and the intermediate results of 10-fold cv are displayed 
      during the cv phase of training. The final output shows the changes in accuracy, averaged over the cv folds, 
      and on the test set, as each new feature is added to the model. The (final) output should look something like:
      
      <lots of fold specific output>
      ...
      
      >> Training classifier <<
      >> Classifying test data <<
      
      Confusion matrix:
      =================
                 for  against  observing
      for        219        3         24
      against     15       64         12
      observing   75       11        101
      
      Measures:
      =========
      accuracy: 0.7328
      
      Per class:
                  accuracy  precision     recall         F1
      for        0.7767176  0.7087379  0.8902439  0.7891892
      against    0.9217557  0.8205128  0.7032967  0.7573964
      observing  0.7671756  0.7372263   0.540107  0.6234568
                  accuracy-cv  accuracy-test
      Q           0.519765          0.503817
      BoW         0.708224          0.698473
      W2V         0.708909          0.698473
      PPDB        0.711729          0.713740
      RootDep     0.731114          0.732824
      NegAlgn     0.732362          0.730916
      SVO         0.734407          0.732824
      
**python run_train_test.py -f <command-separated list of features>**

      Using the -f switch, the model can be run with any subset of the features, given as a comma-separated list, e.g.
      python run_train_test.py -f "Q,BoW,SVO".
      
**python run_train_test.py -i -f <command-separated list of features>**

      This case combines the above, so that a the incremental output for a given list of features is displayed.
      
**python run_train_test.py -a**

      Using the -a switch causes the script to run the ablation test. The final output should look something like this:
      
      <lots of feature specfic output>
      ...
      
                  accuracy-cv       accuracy-test
      -['Q']           1.848082       0.190840
      -['BoW']         1.664340       5.152672
      -['W2V']         0.049020      -0.190840
      -['PPDB']        0.466699       0.763359
      -['RootDep']     2.024615       2.480916
      -['NegAlgn']     0.335388       0.000000
      -['SVO']         0.204543       0.190840
      
**python run_train_test.py -a -f <command-separated list of features>**

      This case performs the ablation test for a given list of features is displayed.

To run the code for the MaxEntClassificationEDA classifier, do the following:

1. Follow the instructions to download an install EOP, which can be found here: https://github.com/hltfbk/EOP-1.2.3/wiki
2. Train the model with the English RTE-3 training data-set, and then test it with the EmergentLite test data-set: 

      1. cd into the following directory: <where you installed EOP>/Excitement-Open-Platform-1.2.3/target/EOP-1.2.3
      2. train the model: issue the following command at the prompt:
      
            java -Djava.ext.dirs=../EOP-1.2.3 eu.excitementproject.eop.util.runner.EOPRunner -train -trainFile ./eop-resources-1.2.3/data-set/English_dev.xml -config ./eop-resources-1.2.3/configuration-files/MaxEntClassificationEDA_Base+WN+VO+TP+TPPos_EN.xml
            
      3. test the model: issue the following command at the prompt:
      
            java -Djava.ext.dirs=../EOP-1.2.3 eu.excitementproject.eop.util.runner.EOPRunner -test -testFile <path to where mscproject was cloned>/mscproject/data/emergent/url-versions-2015-06-14-clean-test-rte.xml -config ./eop-resources-1.2.3/configuration-files/MaxEntClassificationEDA_Base+WN+VO+TP+TPPos_EN.xml -output <where you want the output to go>
            
3. Train the model with the EmergentLite training data-set, and then test it with the EmergentLite test data-set:

      1. cd into the following directory: <where you installed EOP>/Excitement-Open-Platform-1.2.3/target/EOP-1.2.3
      2. train the model: issue the following command at the prompt:
      
            java -Djava.ext.dirs=../EOP-1.2.3 eu.excitementproject.eop.util.runner.EOPRunner -train -trainFile <path to where mscproject was cloned>/mscproject/data/emergent/url-versions-2015-06-14-clean-train-rte.xml -config ./eop-resources-1.2.3/configuration-files/MaxEntClassificationEDA_Base+WN+VO+TP+TPPos_EN.xml
            
      3. test the model: same as step 3. above
      
In each case above, the output will consist of a number of files. The results files will be named: MaxEntClassificationEDA_Base+WN+VO+TP+TPPos_EN.xml_results.{txt or xml}; the contents are pretty self-explanatory.

The project comes complete with an ./output/eop/ directory containing pre-computed results:

      ./rte-clean-test/ - the results of training the model on English RTE-3, and testing it on EmergentLite (test data-set)
      
      ./rte-clean-test-fa/ - the results of training the model on English RTE-3, and testing it on EmergentLite (test data-set) with observing stance articles removed, i.e. only against the for and against stances
      
      ./emergent-clean-test/ - the results of training the model on EmergentLite (training data-set), and testing it on EmergentLite (test data-set)
      
      ./fold-X/ - the results of training the model on English RTE-3, and testing it on EmergentLite (test data-set fold X)
      
Running the following script:

**python run_eop_compare.py**

compares the (pre-computed) output of the MaxEntClassificationEDA classifier for the scenarios decribed above, and outputs accuracy results.

      






