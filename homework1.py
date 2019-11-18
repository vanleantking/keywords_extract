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
TRAINING_FILE = './data/result/stopword_token_clustering.csv'

TEST_DATA = "./data/News/Test_Full"

TEST_FILE = './data/test.csv'

MODEL_FILE = './data/model.pkl'
RESULT_PREDICT = './data/result.csv'
TRUE_LABEL = './data/true_label.csv'

def calc_tfidf_category(category, top = 10):
    data = collect_data.readData(TRAIN_DATA, TRAINING_FILE, 1500)
    tmp = data[data['label']==category]
      
    return get_top(tmp["text"], top)

def get_top(data, top = 10):
    tfidf_vect = TfidfVectorizer(analyzer='word',
                                 token_pattern=r'[a-zA-Z0-9_]+',
                                 lowercase = True,
                                 max_df=0.05,
                                 stop_words= collect_data.get_stop_word(),
                                 max_features=5000)
    count_train = tfidf_vect.fit(data)
    bag_of_words = tfidf_vect.transform(data)
    feature_names = np.array(count_train.get_feature_names())
    print(feature_names)
    return
    max_val = bag_of_words.max(axis=0).toarray().ravel()
    
    #sort weights from smallest to biggest and extract their indices 
    sort_by_tfidf = max_val.argsort()
    return feature_names[sort_by_tfidf[-top:]]

#3.b
def top_words_category():
    for index, category in enumerate(collect_data.LABELS):
        print(calc_tfidf_category(category, 100).tolist())


def calc_tfidf_new(category, top = 10):
    data = collect_data.readData(TRAIN_DATA, TRAINING_FILE, 1500)
    tmp = data[data['label']==category]
    top_words = {}
    for index, text in enumerate(tmp['text'].tolist()):
        top_words[index] = get_top([text], top).tolist()

    return top_words

#3.a 
def top_word_new():
    for index, category in enumerate(collect_data.LABELS):
        print(calc_tfidf_new(category))
        
       
def model_one_with_all(df):
    classifier_params = {}
    for index, category in enumerate(collect_data.LABELS):
        classifier = df.copy() #initial another pandas
        classifier.loc[classifier['label'] != category, 'label'] = 'Khac'
        size_polipatics_society = classifier[classifier['label'] == category].shape[0]
        size_others = classifier[classifier['label'] == 'Khac'].shape[0]
        print('Number of politics-society documents: %s' %size_polipatics_society)
        print('Number of other documents: %s' %size_others)
        train_y = classifier['label']
        train_x = classifier['text']
        
        # split the dataset into training and test datasets 
        
        #print(train_x[165], train_y[165])
        # label encode the target variable, encode labels to 0 or 1
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        
        # word level tf-idf
        tfidf_vect = TfidfVectorizer(analyzer='word',
                                     token_pattern=r'\w{1,}',
                                     stop_words = collect_data.get_stop_word(),
                                     max_features=5000)
        tfidf_vect.fit(train_x)
        xtrain_tfidf =  tfidf_vect.transform(train_x)
        
        # Getting transformed training and testing dataset
        print('Number of training documents: %s' %str(xtrain_tfidf.shape[0]))
        print('Number of features of each document: %s' %str(xtrain_tfidf.shape[1]))
        print('xtrain_tfidf shape: %s' %str(xtrain_tfidf.shape))
        print('train_y shape: %s' %str(train_y.shape))
        
        ### START CODE HERE ###
        train_y = np.expand_dims(train_y, axis=0)
        
        # for convenience in this exercise, we also use toarray() to convert
        # sparse to dense matrix 
        xtrain_tfidf =  xtrain_tfidf.T.toarray()
        ### END CODE HERE ###
        
        # New shape 
        print('xtrain_tfidf shape: %s' %str(xtrain_tfidf.shape))
        print('train_y shape: %s' %str(train_y.shape))
        
        # return c  
        d = functions.model_one_vs_all(xtrain_tfidf,
                            train_y, 
                            num_iterations = 3000,
                            learning_rate = .5,
                            print_cost = True)

        classifier_params[category] = {"w" : d['w'], 'b': d['b']}
    
        
    return classifier_params


def predict_one_vs_all(classifier_params, X):
    scores = np.zeros((4, X.shape[1]))
    for index, categoty in enumerate(collect_data.LABELS):
        logistic = classifier_params[categoty]
        scores[index, :] = functions.predictmulti(logistic['w'], logistic['b'], X)[0]
    pred_X = np.argmax(scores, axis=0)
    le = preprocessing.LabelEncoder()
    le.fit(["Thethao", "Doisong", "Khoahoc", "Kinhdoanh"])
    labels = le.inverse_transform(pred_X)
    return labels, pred_X, scores
        
def multi_classifires():
    df = collect_data.readData(TRAIN_DATA, TRAINING_FILE, 1500)  
    losses = []
    auc = []
    
    for category in collect_data.LABELS:
        classifier = df.copy()
        classifier.loc[classifier['label'] != category, 'label'] = 'Khac'
        train_x, test_x, train_y, test_y = model_selection.train_test_split(
            classifier['text'],
            classifier['label'])
        
        tfidf_vect = TfidfVectorizer(analyzer='word',
                                 token_pattern=r'\w{1,}',
                                 stop_words= collect_data.get_stop_word(),
                                 max_features=5000)
        tfidf_vect.fit(classifier['text'])
        xtrain_tfidf = tfidf_vect.transform(train_x)
        xtest_tfidf = tfidf_vect.transform(test_x)
        
        logistic_classifier = LogisticRegression(multi_class='ovr', solver='sag', C=10)

        cv_loss = np.mean(cross_val_score(logistic_classifier,
                                          xtrain_tfidf,
                                          train_y,
                                          cv=5,
                                          scoring='neg_log_loss'))
        losses.append(cv_loss)
        print('CV Log_loss score for class {} is {}'.format(category, cv_loss))
    
        cv_score = np.mean(cross_val_score(logistic_classifier,
                                           xtrain_tfidf,
                                           train_y,
                                           cv=5,
                                           scoring='accuracy'))
        print('CV Accuracy score for class {} is {}'.format(category, cv_score))
        
        logistic_classifier.fit(xtrain_tfidf, train_y)
        y_pred = logistic_classifier.predict(xtest_tfidf)
        y_pred_prob = logistic_classifier.predict_proba(xtest_tfidf)[:, 1]
        auc_score = metrics.roc_auc_score(test_y, y_pred_prob)
        auc.append(auc_score)
        print("CV ROC_AUC score {}\n".format(auc_score))
        
        print(confusion_matrix(test_y, y_pred))
        print(classification_report(test_y, y_pred))
    print('Total average CV Log_loss score is {}'.format(np.mean(losses)))
    print('Total average CV ROC_AUC score is {}'.format(np.mean(auc)))
    
def train_tunning():
    df = collect_data.readData(TRAIN_DATA, TRAINING_FILE, 1500)
    train_x, test_x, train_y, test_y = model_selection.train_test_split(
                df['text'],
                df['label'])
    if os.path.isfile('./data/model_train'):
        vec = open("./data/model_train", 'rb') # rb= read in bytes
        grid3 = pickle.load(vec)
        vec.close()
    else:
        start_time=time.time()    
        pipe = make_pipeline(TfidfVectorizer(analyzer='word',
                                     token_pattern=r'\w{1,}',
                                     stop_words= collect_data.get_stop_word()),
                             OneVsRestClassifier(LogisticRegression()))
        param_grid = {'tfidfvectorizer__max_features': [5000, 10000],
                      'onevsrestclassifier__estimator__solver': ['liblinear', 'sag'],
                     } 
        grid = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy')
        
        grid3 = grid.fit(train_x, train_y)
        
        end_time=time.time()
        print("total time",end_time-start_time)
        
        save_classifier = open("./data/model_train", 'wb') #wb= write in bytes. 
        pickle.dump(grid3, save_classifier) #use pickle to dump the grid3 we trained, as 'Tfidf_LogR.pickle' in wb format
        save_classifier.close()
    
    print(grid3.best_estimator_.named_steps['onevsrestclassifier'])
    print(grid3.best_estimator_.named_steps['tfidfvectorizer'])
    
    grid3.best_params_
    grid3.best_score_
    predicted_y_test = grid3.predict(test_x)
    
    X_test_list = test_x.tolist()
    predicted_y_test_list = predicted_y_test.tolist()
    
    save = pd.DataFrame(np.column_stack([X_test_list, predicted_y_test_list]))
    save.to_csv("./data/result_trained.csv", sep=',', encoding='utf-16',
              header=True, index=False)
    
if __name__ == '__main__':
    top_words_category()
    # top_word_new()
    #classifiers_app1()
    # multi_classifires()
    #train_tunning()
    

