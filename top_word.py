from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from utils import functions, collect_data
import numpy as np

def get_document_filenames(document_path='/home/tool/document_text'):
    return [os.path.join(document_path, each)
            for each in os.listdir(document_path)]


def create_vectorizer():
    # Arguments here are tweaked for working with a particular data set.
    # All that's really needed is the input argument.
    return TfidfVectorizer(input='filename', max_features=200,
                           token_pattern='(?u)\\b[a-zA-Z]\\w{2,}\\b',
                           max_df=0.05,
                           stop_words='english',
                           ngram_range=(1, 3))


def display_scores(vectorizer, tfidf_result):
    # http://stackoverflow.com/questions/16078015/
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for item in sorted_scores:
        print("{0:50} Score: {1}".format(item[0], item[1]))

def get_top(data, top = 10):
    tfidf_vect = TfidfVectorizer(analyzer='word',
                                 token_pattern=r'[a-zA-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ0-9_]+',
                                 lowercase = True,
                                 # max_df=0.05,
                                 encoding='utf-16',
                                 stop_words= collect_data.get_stop_word(),
                                 max_features=5000)
    count_train = tfidf_vect.fit(data)
    bag_of_words = tfidf_vect.transform(data)
    feature_names = np.array(count_train.get_feature_names())
    print(feature_names)
    # return
    max_val = bag_of_words.max(axis=0).toarray().ravel()
    
    #sort weights from smallest to biggest and extract their indices 
    sort_by_tfidf = max_val.argsort()
    # print(sort_by_tfidf[-top:], bag_of_words.max(axis=0), bag_of_words.max(axis=0).toarray(), bag_of_words.max(axis=0).toarray().ravel())
    return feature_names[sort_by_tfidf[-top:]]

def get_top_n_words(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    
    get_top_n_words(["I love Python", "Python is a language programming", "Hello world", "I love the world"]) -> 
    [('python', 2),
     ('world', 2),
     ('love', 2),
     ('hello', 1),
     ('is', 1),
     ('programming', 1),
     ('the', 1),
     ('language', 1)]
    """
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


if __name__ == '__main__':
    corpus = """trinidad công_bố danh_sách cầu_thủ dự world_cup_dwight_yorke vẫn sẽ
    là đầu_tàu của đt trinidad amp tobago tại world_cup sắp tới hlv leo beenhakker
    vừa gây bất_ngờ cho người hâm_mộ khi công_bố danh_sách cầu_thủ trinidad amp tobago
    sẽ tham_dự vck_world_cup sắp tới cầu_thủ được lựa_chọn lần này đều là những cầu_thủ
    đang có phong_độ tốt của bóng_đá trinidad amp tobago trong đó cựu tiền_đạo của m _u
    dwight_yorke đang thi cho clb sydney australia sẽ là đội_trưởng ngoài dwight_yorke
    một loạt những cầu_thủ khác đang thi_đấu tại anh như stern_john coventry chris_birchall
    port_vale shaka_hislop west_ham ian_cox gillingham cũng đã có tên trong danh_sách đến
    đức vào mùa hè tới mặc_dù đã lựa_chọn được cầu_thủ tốt nhất nhưng hlv beenhakker sẽ phải
    rút danh_sách này xuống còn theo đúng quy_định của fifa và thời_hạn để ông làm_việc này
    là trước ngày ngoài cầu_thủ được dự_định sẽ đưa đến đức hlv beenhakker cũng đã quyết_định
    triệu_tập thêm cầu_thủ dự_bị và các cầu_thủ này sẽ được lựa_chọn nếu một trong số cầu_thủ
    chính_thức bất_ngờ bị chấn_thương danh_sách cầu_thủ của trinidad amp tobago thủ_môn
    kelvin_jack dundee shaka_hislop west_ham clayton_ince coventry_city hậu_vệ dennis_lawrence
    wrexham cyd_gray san_juan_jabloteh marvin_andrews rangers brent_sancho gillingham ian_cox
    gillingham atiba_charles w_connection avery_john new_england_revolution tiền_vệ
    silvio_spann unattached chris_birchall port_vale aurtis_whitley san_juan_jabloteh
    anthony_rougier united_petrotrin anthony_wolfe san_juan_jabloteh densill_theobald
    falkirk carlos_edwards luton dwight_yorke sydney_fc russell_latapy falkirk tiền_đạo
    stern_john coventry kenwyne_jones southampton collin_samuel dundee jason_scotland
    st_johnstone cornell_glen la_galaxy dự_bị brent_rahim jabloteh anton_pierre defence_force
    anthony_warner fulham nigel_henry kiruna_ff ricky_shakes swindon hector_sam port_vale
    scott_sealy kansas_wizards"""
    print(get_top([corpus]))
    # print(len(corpus))

    print(get_top_n_words([corpus]))

    # vectorizer = create_vectorizer([corpus])
    vectorizer = TfidfVectorizer(analyzer='word',
                           token_pattern=r'[a-zA-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ0-9_]+',
                           # max_df=0.05,
                           # stop_words='english',
                           encoding='utf-16',
                         stop_words= collect_data.get_stop_word(),
                            max_features=5000)
    tfidf_result = vectorizer.fit_transform([corpus])
    display_scores(vectorizer, tfidf_result)