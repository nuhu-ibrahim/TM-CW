# Text Mining (COMP61332) Coursework 1: Question Classification
## Requirements
### Scikit-Learn
We used the scikit-learn library to perform estimation of the models' performance. Run the command below to install scikit-learn.
```
    pip install -U scikit-learn
```
## Folder Descriptions
### data folder
This folder contains all the training configuration files, pre-trained embeddings, data cleaning files, and trained models.
### data/model folder
This folder is located inside the data folder and is where all trained models are stored to be used on the test dataset.
### document folder
This folder contains the program documentations.
### src folder
This folder contains a question_classifier.py file, which is the main source code for the project.

## Important Files Description
### src/question_classifier.py
This file is the source code (executable python file) that is to be executed from the command prompt. Instructions can be found below on how to execute this file
### data/BiLSTM.yaml
This file is the configuration file for the BiLSTM classifier. All important criterias that is to be employed by the model during the training is to be manipulated from this file. This file must be of .yaml extension and must be a valid yaml file.
### data/BOW.yaml
This file is the configuration file for the BOW classifier. All important criterias that is to be employed by the model during the training is to be manipulated from this file. This file must be of .yaml extension and must be a valid yaml file.
### data/ensemble.yaml
This file is the configuration file for the several classifiers that are combined (i.e. ensemble). All important criterias that is to be employed by the models during training are to be manipulated from this file. This file must be of .yaml extension and must be a valid yaml file.
### document/README.md
This file is a documentation file that describes the entire project, where files are kepts, and how to execute it.

## Structure of the configuration files
Each config file has three major sub-structure, which includes:
    i.   file\_path: for configuring the location of the files the models depend on.
    ii.  preprocess: for configuring the types of pre-processings that should be performed on data.
    iii. model: for configuring the parameters that should be used by the model.
	iv. boost_rate: for ensemble model only, randomly select boost_rate * len(train_data) pieces of data to build the train data set for each single model in ensemble model.
	v. submodel: for ensemble model only, use to define the number and structure of submodels.
### file_path
Is identical to the structure described below. All the files referenced in this part are to be saved in the the data directory
```
        file_path:
            train_data_path: <Name of the text file that contains the train set>
            test_data_path: <Name of the text file that contains the validation (development) set>
            test_data_path: <Name of the text file that contains the test set>
            stopwords_path: <Name of the text file that contains the stopwords>
            pretrained_path: <Name of the text file that contains the pre-trained embedding>
```
### preprocess:
Is identical to the structure described below. All the values in this sub-section is either true or false.
```
        preprocess:
            lowercase: <true|false>
            remove_stop_words: <true|false>
            remove_special_characters: <true|false>
            replace_abbreviations: <true|false>
            remove_white_space: <true|false>
```
### model:
Is identical to the structure described below. Value either be "BiLSTM", "BOW" or "ensemble".
```
model: <BiLSTM|BOW|ensemble>
```
### parameters:
Is identical to the strucure described below:
```
        embedding: <pretrained|random>
        epoch: <NUM>
        embedding_dim: <NUM>
        hidden_dim: <NUM>
        lr: 0.09 [0 >= lr <= 1]
        freeze: <true|false>
```
### boost_rate:
Is identical to the strucure described below:
```		
        boost_rate: 0.09 [0 >= lr <= 1]
```
### submodel:
Is identical to the strucure described below:
```		submodel:
			model1:
				model_type: <BiLSTM|BOW>
        		embedding: <pretrained|random>
        		epoch: <NUM>
        		embedding_dim: <NUM>
        		hidden_dim: <NUM>
        		lr: 0.09 [0 >= lr <= 1]
        		freeze: <true|false>
			model2:
				model_type: <BiLSTM|BOW>
        		embedding: <pretrained|random>
        		epoch: <NUM>
        		embedding_dim: <NUM>
        		hidden_dim: <NUM>
        		lr: 0.09 [0 >= lr <= 1]
        		freeze: <true|false>
			model3:...
```
## Instruction for executing the program:
### Executing the BOW -- for the Randomly Initialized Word Embedding
#### Step 1 -- Configuration:
Ensure that the BOW.yaml configuration file has the value of embedding as "random" and the value of model as "BOW" as below, and adjust all parameters as you desired.
```
        embedding: random
        model: BOW
```
#### Step 2 -- Training the Model:
Execute the command below:

```
    python3 question_classifier.py --train --config BOW.yaml
```
#### Step 3 -- Testing the Model
Execute the command below:

``` 
    python3 question_classifier.py --test --config BOW.yaml
```

### Executing the BOW -- for the Fine-tuned Pre-trained Embedding
#### Step 1 -- Configuration:
Ensure that the BOW.yaml configuration file has the value of embedding as "trained", value of model as "BOW" and that the value of freeze is "false" as below, and adjust all parameters as desired:
```
        embedding: trained
        freeze: false
        model: BOW
```

#### Step 2 -- Training the Model:
Execute the command below:
```
    python3 question_classifier.py --train --config BOW.yaml
```
#### Step 3 -- Testing the Model
Execute the command below:
```
    python3 question_classifier.py --test --config BOW.yaml
```

### Executing the BOW -- for the Freezed Pre-trained Embedding Option
#### Step 1 -- Configuration:
Ensure that the BOW.yaml configuration file has the value of embedding as "trained", value of model as "BOW" and that the value of freeze is "true" as below, and adjust all parameters as desired:
```
    embedding: pretrained
    freeze: true
    model: BOW
```
#### Step 2 -- Training the Model:
Execute the command below:
```
    python3 question_classifier.py --train --config BOW.yaml
```
#### Step 3 -- Testing the Model
Execute the command below:
```
    python3 question_classifier.py --test --config BOW.yaml
```

### Executing the BiLSTM -- for the Randomly Initialized Word Embedding
#### Step 1 -- Configuration:
Ensure that the BiLSTM.yaml configuration file has the value of embedding as "random" i.e. "embedding: random", value of model as "BiLSTM" as below, and adjust all parameters as you desired.
```
    embedding: random
    model: BiLSTM
```
#### Step 2 -- Training the Model:
Execute the command below:
```
    python3 question_classifier.py --train --config BiLSTM.yaml
```
#### Step 3 -- Testing the Model
Execute the command below:
```
    python3 question_classifier.py --test --config BiLSTM.yaml
```

### Executing the BiLSTM -- for the Fine-tuned Pre-trained Embedding
#### Step 1 -- Configuration:
Ensure that the BiLSTM.yaml configuration file has the value of embedding as "trained", value of model as "BiLSTM" and that the value of freeze is "false" as below, and adjust all parameters as desired:
```
    embedding: pretrained
    freeze: false
    model: BiLSTM
```
#### Step 2 -- Training the Model:
Execute the command below:
```
    python3 question_classifier.py --train --config BiLSTM.yaml
```
#### Step 3 -- Testing the Model
Execute the command below:
```
    python3 question_classifier.py --test --config BiLSTM.yaml
```

### Executing the BiLSTM -- for the Freezed Pre-trained Embedding Option
#### Step 1 -- Configuration:
Ensure that the BiLSTM.yaml configuration file has the value of embedding as "trained", value of model as "BiLSTM" and that the value of freeze is "true" as below, and adjust all parameters as desired:
```
    embedding: pretrained
    freeze: true
    model: BiLSTM
```
#### Step 2 -- Training the Model:
Execute the command below:
```
    python3 question_classifier.py --train --config BiLSTM.yaml
```
#### Step 3 -- Testing the Model
Execute the command below:
```
    python3 question_classifier.py --test --config BiLSTM.yaml
```

### Executing the Ensemble
#### Step 1 -- Configuration:
Ensure that the ensemble.yaml configuration has correct inputs for each kinds of models of choice as described in this documentation above and ensure that the model attribute has value of "ensemble" as below:
```
    model: ensemble
```
For the purpose of this work, the ensemble is a combination of five different models that can be tweaked in any different way from the ensemble.yaml configuration file.

#### Step 2 -- Training the Model:
Execute the command below:
```
    python3 question_classifier.py --train --config ensemble.yaml
```
#### Step 3 -- Testing the Model
Execute the command below:
```
    python3 question_classifier.py --test --config ensemble.yaml
```