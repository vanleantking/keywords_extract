import os
import pandas as pd
import re
from nltk.tokenize import sent_tokenize
from pyvi import ViTokenizer
from underthesea import word_tokenize
#import unicodedata

#from unidecode import unidecode

CATEGORIES = ["The thao", "Doi song", "Khoa hoc", "Kinh doanh"]
LABELS = ["Thethao", "Doisong", "Khoahoc", "Kinhdoanh"]
STOP_WORDS_PATH = './data/vnmstopwords.txt'

def readData(data_path, file_name, Limit = None):
    if os.path.isfile(file_name):
        df = pd.read_csv(file_name, encoding="utf-16")
        return df
    texts = []
    #for data_dir in TRAIN_DATA:
    for index, category in enumerate(CATEGORIES):
        path = os.path.join(data_path, category)
        if os.path.isdir(path):
            counter = 1
            for fname in sorted(os.listdir(path)):
                fpath = os.path.join(path, fname)
                d = {}
                with open(fpath, 'r', encoding="utf-16") as data_file:
                    d['text'] = process_data(data_file.read().strip())
                    d['label'] = LABELS[index]
                    texts.append(d)
                
                if Limit != None and counter >= Limit:
                    break
                counter += 1
    df = pd.DataFrame.from_dict(texts)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(file_name, sep=',', encoding='utf-16',
              header=True, columns=['text', 'label'], index=False)
    return df

def get_stop_word():
    stopwords = []
    with open(STOP_WORDS_PATH,"r", encoding="utf-8") as f:
        stopwords = [line.strip() for line in f]
    return stopwords
    
def process_data(news):
    #### remove digits ####
    output = re.sub(r'\d+', '', news)
    # output = word_tokenize(output, format="text")

    # remove stopwords
    stopwords = change_format_stopwords()
    # stopwords = get_stop_word()
    for stw in stopwords:
        match_string = r'\b' + stw + r'\b'
        regex = re.compile(match_string, re.S)
        output = regex.sub(lambda m: m.group().replace(stw," ",1), output)

    output = re.sub('[^a-zA-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ_ ]', ' ', output)
    output = ' '.join(output.split())

    #news = ''.join([i for i in news if not i.isdigit()])
    #filter_digit = filter(lambda x: x.isalpha(), news)
    #result = ''.join(filter_digit)
    return output.strip().lower()


# return list of news
def train_w2vec(data_path, file_name, Limit = 1500):
    if os.path.isfile(file_name):
        df = pd.read_csv(file_name, encoding="utf-16")
        convert_to_list = df['text'].tolist()
        y = [eval(s) for s in convert_to_list]
        df['text'] = pd.Series(y)
        return df
    texts = []
    #for data_dir in TRAIN_DATA:
    for index, category in enumerate(CATEGORIES):
        path = os.path.join(data_path, category)
        if os.path.isdir(path):
            counter = 1
            for fname in sorted(os.listdir(path)):
                fpath = os.path.join(path, fname)
                d = {}
                with open(fpath, 'r', encoding="utf-16") as data_file:
                    d['text'] = list(tokenize(data_file.read().strip()))
                    d['label'] = LABELS[index]
                    texts.append(d)
                
                if counter >= Limit:
                    break
                counter += 1
    df = pd.DataFrame.from_dict(texts)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(file_name, sep=',', encoding='utf-16',
              header=True, columns=['text', 'label'], index=False)
    return df

def change_format_stopwords():
    stopwords = get_stop_word()
    result = ['_'.join(word.split()) for word in stopwords]
    return result
    
# preprocessing eacg news and change to list of words
def tokenize(e_news):
    stopwords = change_format_stopwords()
    # tokenizer = ViTokenizer.tokenize(e_news)
    tokenizer = word_tokenize(e_news, format="text")
    words_arrays = re.findall(r"[\w']+|[.,!?;\/+]", tokenizer.strip().lower())

    # remove stopwords
    sentence = [word for word in words_arrays \
                        if word not in stopwords]

    # remove punctations
    sentence = [c for c in sentence if c not in \
                        ('!', '.' ,':', '/', '\\', '-', '+', '_', '(', ')', '*', '&', '#', ';', '?', '>', '<', '%', \
                            '{', '}', '=', ',', ']', '[', '`', '\'')]
    # remove number
    sentence = [word for word in sentence \
                        if not re.search(r'[0-9]+', word)]

    
    return sentence


def list_tokenize(e_news):
    stopwords = change_format_stopwords()
    tokenizer = word_tokenize(e_news, format="text")
    # tokenizer = ViTokenizer.tokenize(e_news)
    sent_text = sent_tokenize(tokenizer)
    sents = []
    for sentence in sent_text:
        words_arrays = re.findall(r"[\w']+|[.,!?;\/+]", sentence.strip().lower())

        # remove stopwords
        sentence = [word for word in words_arrays \
                            if word not in stopwords]

        # remove punctations
        sentence = [c for c in sentence if c not in \
                            ('!', '.' ,':', '/', '\\', '-', '+', '_', '(', ')', '*', '&', '#', ';', '?', '>', '<', '%', \
                                '{', '}', '=', ',', ']', '[', '`', '\'')]
        # remove number
        sentence = [word for word in sentence \
                            if not re.search(r'[0-9]+', word)]
        sents.append(sentence)
    return sents
        
