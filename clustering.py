# libraries for dataset preparation, feature engineering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import functions, collect_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import time
from sklearn.multiclass import OneVsRestClassifier

import pickle
import os

TRAIN_DATA = "./data/News/Train_Full"
TRAINING_FILE = './data/result/wnottokenize_rmstopword_clustering.csv'

TEST_DATA = "./data/News/Test_Full"

TEST_FILE = './data/test.csv'

MODEL_FILE = './data/model.pkl'
RESULT_PREDICT = './data/result.csv'
TRUE_LABEL = './data/true_label.csv'

def calc_tfidf_category(category, top = 10):
    data = collect_data.readData(TRAIN_DATA, TRAINING_FILE)
    tmp = data[data['label']==category]
      
    return get_top(tmp["text"], top)

def get_top(data, top = 10):
    tfidf_vect = TfidfVectorizer(analyzer='word',
                                 token_pattern=r'[a-zA-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ0-9_]+',
                                 lowercase = True,
                                 ngram_range = (1,4),
                                 stop_words= collect_data.get_stop_word(),
                                 max_features=10000)
    count_train = tfidf_vect.fit(data)
    bag_of_words = tfidf_vect.transform(data)
    feature_names = np.array(count_train.get_feature_names())
    max_val = bag_of_words.max(axis=0).toarray().ravel()
    
    #sort weights from smallest to biggest and extract their indices 
    sort_by_tfidf = max_val.argsort()
    return feature_names[sort_by_tfidf[-top:]]

#3.b
def top_words_category():
    for index, category in enumerate(collect_data.LABELS):
        print(calc_tfidf_category(category, 100).tolist())


def calc_tfidf_new(category, top = 10):
    data = collect_data.readData(TRAIN_DATA, TRAINING_FILE)
    tmp = data[data['label']==category]
    top_words = {}
    for index, text in enumerate(tmp['text'].tolist()):
        top_words[index] = get_top([text], top).tolist()

    return top_words

#3.a 
def top_word_new():
    for index, category in enumerate(collect_data.LABELS):
        print(calc_tfidf_new(category))


def clustering_word():
    data = collect_data.readData(TRAIN_DATA, TRAINING_FILE)
    tfidf_vect = TfidfVectorizer(analyzer='word',
                                 token_pattern=r'[a-zA-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ0-9_]+',
                                 lowercase = True,
                                 ngram_range = (1,4),
                                 stop_words= collect_data.get_stop_word(),
                                 max_features=10000)
    count_train = tfidf_vect.fit(data["text"])
    # bag_of_words = tfidf_vect.transform(data)
    # feature_names = np.array(count_train.get_feature_names())
    print(count_train.get_feature_names(), len(count_train.get_feature_names()))

        
if __name__ == '__main__':
    # top_words_category()
    # top_word_new()
    #classifiers_app1()
    # multi_classifires()
    #train_tunning()
    clustering_word()