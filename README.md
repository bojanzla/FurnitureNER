furniture - NER
==============================

Furniture Named Entity Recognition

Project Organization
------------


    ├── README.md           <- The top-level README
    ├── Explanations.pdf    <- Guide through the code
    │
    ├── .env                <- Environment configuration (in .gitignore)
    │
    ├── .gitignore          <- Tells Git which files and folders to ignore
    |
    ├── data                <- Data folder (Data is not open, so this folder is in .gitignore
    │
    ├── models              <- Trained and serialized models
    │
    ├── notebooks           <- Jupyter notebooks.
    │
    ├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc
    |
    ├── src                 <- Source code of this project
    │   ├── __init__.py     <- Makes src a Python module
    │   │
    │   ├── config.py       <- Configuration file of the project
    │   │
    │   ├── app             <- Application folder
    │   │   └── main.py     <- Main application file, such as API or other script file
    │   │
    │   ├── label            <- Scripts to used to help embedd and tag the data
    │   │   ├── ada_embedder.py         <- text embedding with ADA-002
    │   │   └── bert_embedder.py        <- text embedding with BERT L4_512_A8 uncased 
    │   │
    │   ├── scrape           <- Scripts to used to scrape webpages
    │   │   └── crawler.py         <- Webpage scrapper
    |   |
    │   ├── train            <- Scripts to train models
    │   │   ├── train.py         <- Classical ML with embeding
    │   │   └── train_tf.py      <- BERT fine-tuning with Tensorflow
    │   │
    │   ├── utils             <- Scripts to support other methods        
    │   │   └── functions.py    <- Shared functions depo
    │   │
    │   └── visuals         <- Scripts to create visualizations
    │       └── visualize.py    <- General visualization methods
    │
    ├── requirements.txt      <- requirements.txt for venv
    ├── requirements_tf.txt   <- requirements_tf.txt for venv_tf 
    └── 

--------

Setup
------------
First we need to create two virtual environments in our project. This is due to highly 
recommended isolation of Tensorflow. Create them using <b> requirements.txt </b> and
<b> requirements_tf.txt </b> 

------------
Quick try
------------

For trying out the solution first activate <b> venv_tf </b> and then go to src/app/ and run main.py.  <br /> 
You will see the following prompt

 ```
  Please enter URL for furniture product extraction:
 ```

after typing in the desired url that you want to get your products from you'll be prompted again with
 ```
  Do you want to scrape full domain? (y/n): 
 ```
where with answer 'y' you whole domain gets scrapped and with 'n' only the current page. <br />
If everything goes ok you'll get a list of furniture products :)

### Important :
#### In this stage the app will work only with url containing <b> '/product/' </b> ! 

Expalanation of the building steps and logic is given in <b> explanation.md </b> file

## Models

Trained model files (.keras) are not included in this repository due to GitHub file size limits. 
To reproduce them, run the training scripts.

The following models were trained:
- bert_L-2_H-256_A-4
- bert_L-2_H-768_A-12
- bert_L-4_H-128_A-2
- bert_L-8_H-128_A-2
- bert_embedder

