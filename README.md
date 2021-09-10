# Stanford Question Answering Dataset #
![Alt text](https://i.imgur.com/vYGmOJw.png)

### What is this repository for? ###

* This repository implements two model trained on the SQuAD dataset, one for question answering, and one for information retrieval.


### Requirements ###

* Python 3
* Keras
* Tensorflow 
* Numpy
* Pickle 3
* Pandas

### Overview ###

* E2E.ipynb Contains the module for End to End information retrieval
* SQuAD_project.ipynb Contains the base SQuAD model 
* IR_SQuAD.ipynb Contains the module for information retrieval
* compute_answers.py A script that takes as input a dataset containing questions and contexts and returns the predicted answers.

### How do I run the model? ###

* ./python3 compute_answers.py PATH_TO_DATASET 

### Who do I talk to? ###

* Andrea Lavista - andrea.lavista@studio.unibo.it
* Daniele Domenichelli - daniele.domenichell2@studio.unibo.it
* Luca Rispoli - luca.rispoli@studio.unibo.it
