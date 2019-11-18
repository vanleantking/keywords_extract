import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils import functions, collect_data

TRAIN_DATA = "./data/News/Train_Full"
TRAINING_FILE = './data/result/rmstopword_clustering.csv'

TEST_DATA = "./data/News/Test_Full"

TEST_FILE = './data/test.csv'

MODEL_FILE = './data/model.pkl'
RESULT_PREDICT = './data/result.csv'
TRUE_LABEL = './data/true_label.csv'


def clustering_word():
    data = collect_data.readData(TRAIN_DATA, TRAINING_FILE)
    return data["text"]
    # print(data.head())
    # tfidf_vect = TfidfVectorizer(analyzer='word',
    #                              token_pattern=r'[a-zA-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ0-9_]+',
    #                              lowercase = True,
    #                              ngram_range = (1,4),
    #                              min_df = 5,
    #                              stop_words= collect_data.get_stop_word(),
    #                              max_features=10000)
    # count_train = tfidf_vect.fit(data["text"])

    # bag_of_words = tfidf_vect.transform(data)
    # feature_names = np.array(count_train.get_feature_names())
    # print(count_train.get_feature_names(), len(count_train.get_feature_names()))

def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1, 1)
    
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
    print(sse)
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    f.show()
    print(f, ax)

def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=3000, replace=False)
    
    pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())
    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items,:].todense()))
    
    
    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
    
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot')
    
    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot')

def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
    


df = clustering_word()
tfidf = TfidfVectorizer(
    min_df = 5,
    max_df = 0.95,
    #ngram_range = (1,5),
    max_features = 10000,
    stop_words= collect_data.change_format_stopwords())
tfidf.fit(df)
word_transform = tfidf.transform(df)   
find_optimal_clusters(word_transform, 12)
clusters = MiniBatchKMeans(n_clusters=4, init_size=1024, batch_size=2048, random_state=20).fit_predict(word_transform)
plot_tsne_pca(word_transform, clusters)
get_top_keywords(word_transform, clusters, tfidf.get_feature_names(), 500)

# if __name__ == '__main__':
#     # top_words_category()
#     # top_word_new()
#     #classifiers_app1()
#     # multi_classifires()
#     #train_tunning()
#     df = clustering_word()
#     tfidf = TfidfVectorizer(
# 	    min_df = 5,
# 	    max_df = 0.95,
# 	    max_features = 10000,
# 	    stop_words= collect_data.get_stop_word())
#     tfidf.fit(df)
#     word_transform = tfidf.transform(df)
#     find_optimal_clusters(word_transform, 20)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # x = np.arange(0, 999, 0.1)
    # y1 = -.75
    # y2 = .75
    # ax.fill_between(x, y1, y2, color='lawngreen', alpha='.6')
    # ax.scatter(df.A, df.B)
    # ax.plot(df.A, df.B)
    # ax.axhline(y=0, color='black')
    # ax.set_xticks(np.arange(0, 999))
    # ax.set_ylim([-4, 4])
    # ax.set_xlim([0, df.A.max() + 1])
    # plt.show()