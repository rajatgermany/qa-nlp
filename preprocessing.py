import pandas as pd
import numpy as np
import pickle
from textblob import TextBlob
import torch

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

import numpy as np
from numpy import dot
from numpy.linalg import norm

import nltk
nltk.download('punkt')


from InferSent.models import InferSent

# Load Infersent Model
V = 1
MODEL_PATH = './encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))
W2V_PATH = './Glove/glove.840B.300d.txt'
infersent.set_w2v_path(W2V_PATH)

print("infersent is loaded")

def feature_value(sentences, sentences_embeddings, question_embedding, metric):
    result = []
    for i in range(0,len(sentences)):
        question_embedding = [question_embedding]
        sentence_embedding = [sentences_embeddings[i]]

        if metric == 'cosine_similarity':
            metric = cosine_similarity(question_embedding, sentence_embedding)
            
        if metric == 'euclidean':
            metric = euclidean_distances(question_embedding, sentence_embedding)  

        result.append(metric[0][0])  
    return result


def pad(data, max_length):
    mean = sum(data)/len(data)
    length_of_data = len(data)
    pad_number =  22 - length_of_data
    data = data + [mean]*pad_number
    return data


def combine_features(sentences, cosine_similarity, euclidean):
    """
    Pads the euclidean and cosine values for particualr instance and generates resultant dataframe
    for modelling , it has eculidean distance between question and all sentnces and cosine similarity
    between between question and all sentences as well and last feature is the index of the answer in the sentnces
    """
    euclidean = pad(euclidean, 12)
    cosine_similarity = pad(cosine_similarity, 12)
    features = euclidean + cosine_similarity 
    return features

def preprocess(document, question):
    blob = TextBlob(document)
    sentences = [item.raw for item in blob.sentences]
    infersent.build_vocab(sentences, tokenize=True)

    question_embedding = infersent.encode([question], tokenize=True)[0]
    sentences_embeddings = []

    for i in range(0,len(sentences)):
        sent_embedding = infersent.encode([sentences[i]], tokenize=True)[0]
        sentences_embeddings.append(sent_embedding)
    
    cosine_similarity = feature_value(sentences, sentences_embeddings , question_embedding, 'cosine_similarity') 
    euclidean = feature_value(sentences, sentences_embeddings , question_embedding, 'euclidean') 

    festures = combine_features(sentences, cosine_similarity, euclidean)
    dataframe = pd.DataFrame([festures])
    return dataframe, sentences
   