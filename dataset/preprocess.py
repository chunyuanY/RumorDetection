import itertools
import re
from collections import Counter
import gensim
import numpy as np
import scipy.sparse as sp
import pickle
import jieba
jieba.set_dictionary('dict.txt.big')


w2v_dim = 300

dic = {
    'non-rumor': 0,   # Non-rumor   NR
    'false': 1,   # false rumor    FR
    'unverified': 2,  # unverified tweet  UR
    'true': 3,    # debunk rumor  TR
}

def clean_str_cut(string, task):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    if task != "weibo":
        string = re.sub(r"[^A-Za-z0-9(),!?#@\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)

    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    words = list(jieba.cut(string.strip().lower())) if task == "weibo" else string.strip().lower().split()
    return words


def build_symmetric_adjacency_matrix(edges, shape):
    def normalize_adj(mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

    adj = sp.coo_matrix((edges[:, 2], (edges[:, 0], edges[:, 1])), shape=shape, dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj.tocoo()


def read_corpus(root_path, file_name):
    X_tids = []
    X_uids = []

    with open(root_path + file_name +".train", 'r', encoding='utf-8') as input:
        X_train_tid, X_train_content, y_train = [], [], []
        for line in input.readlines():
            tid, content, label = line.strip().split("\t")
            X_tids.append(tid)
            X_train_tid.append(tid)
            X_train_content.append(clean_str_cut(content, file_name))
            y_train.append(dic[label])

    with open(root_path + file_name +".dev", 'r', encoding='utf-8') as input:
        X_dev_tid, X_dev_content, y_dev = [], [], []
        for line in input.readlines():
            tid, content, label = line.strip().split("\t")
            X_tids.append(tid)
            X_dev_tid.append(tid)
            X_dev_content.append(clean_str_cut(content, file_name))
            y_dev.append(dic[label])

    with open(root_path + file_name +".test", 'r', encoding='utf-8') as input:
        X_test_tid, X_test_content, y_test = [], [], []
        for line in input.readlines():
            tid, content, label = line.strip().split("\t")
            X_tids.append(tid)
            X_test_tid.append(tid)
            X_test_content.append(clean_str_cut(content, file_name))
            y_test.append(dic[label])

    with open(root_path + file_name +"_graph.txt", 'r', encoding='utf-8') as input:
        relation = []
        for line in input.readlines():
            tmp = line.strip().split()
            src = tmp[0]
            X_uids.append(src)

            for dst_ids_ws in tmp[1:]:
                dst, w = dst_ids_ws.split(":")
                X_uids.append(dst)
                relation.append([src, dst, w])

    X_id = list(set(X_tids + X_uids))
    num_node = len(X_id)
    print(num_node)
    X_id_dic = {id:i for i, id in enumerate(X_id)}

    relation = np.array([[X_id_dic[tup[0]], X_id_dic[tup[1]], tup[2]] for tup in relation])
    relation = build_symmetric_adjacency_matrix(relation, shape=(num_node, num_node))

    X_train_tid = np.array([X_id_dic[tid] for tid in X_train_tid])
    X_dev_tid = np.array([X_id_dic[tid] for tid in X_dev_tid])
    X_test_tid = np.array([X_id_dic[tid] for tid in X_test_tid])

    return X_train_tid, X_train_content, y_train, \
           X_dev_tid, X_dev_content, y_dev, \
           X_test_tid, X_test_content, y_test, \
           relation


# def train_dev_test_split(root_path, file_name):
#     num_node, relation, X_tid, X_content, y = read_corpus(root_path, file_name)
#     relation = build_symmetric_adjacency_matrix(relation, shape=(num_node, num_node))
#
#     X_content_idx = np.arange(len(X_content))
#     X_idx, X_dev_idx, y, y_dev = train_test_split(X_content_idx, y, test_size=0.1, random_state=0, stratify=y)  #
#
#     X_dev_tid = X_tid[X_dev_idx].tolist()
#     X_dev = X_content[X_dev_idx].tolist()
#
#     X_tid = X_tid[X_idx]
#     X_content = X_content[X_idx]
#
#     X_content_idx = np.arange(len(X_content))
#     X_train_idx, X_test_idx, y_train, y_test = train_test_split(X_content_idx, y, test_size=0.25, random_state=0, stratify=y) #
#
#     X_train_tid = X_tid[X_train_idx].tolist()
#     X_test_tid = X_tid[X_test_idx].tolist()
#
#     X_train = X_content[X_train_idx].tolist()
#     X_test = X_content[X_test_idx].tolist()
#
#     return X_train_tid, X_train, y_train.tolist(), \
#            X_dev_tid, X_dev, y_dev.tolist(), \
#            X_test_tid, X_test, y_test.tolist(), \
#            relation


def vocab_to_word2vec(fname, vocab):
    """
    Load word2vec from Mikolov
    """
    word_vecs = {}
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    count_missing = 0
    for word in vocab:
        if model.__contains__(word):
            word_vecs[word] = model[word]
        else:
            #add unknown words by generating random word vectors
            count_missing += 1
            word_vecs[word] = np.random.uniform(-0.25, 0.25, w2v_dim)

    print(str(len(word_vecs) - count_missing)+" words found in word2vec.")
    print(str(count_missing)+" words not found, generated by random.")
    return word_vecs


def build_vocab_word2vec(sentences, w2v_path='numberbatch-en.txt'):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    vocabulary_inv = []
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv += [x[0] for x in word_counts.most_common() if x[1] >= 2]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    print("embedding_weights generation.......")
    word2vec = vocab_to_word2vec(w2v_path, vocabulary)     #
    embedding_weights = build_word_embedding_weights(word2vec, vocabulary_inv)
    return vocabulary, embedding_weights


def pad_sequence(X, max_len=50):
    X_pad = []
    for doc in X:
        if len(doc) >= max_len:
            doc = doc[:max_len]
        else:
            doc = [0] * (max_len - len(doc)) + doc
        X_pad.append(doc)
    return X_pad


def build_word_embedding_weights(word_vecs, vocabulary_inv):
    """
    Get the word embedding matrix, of size(vocabulary_size, word_vector_size)
    ith row is the embedding of ith word in vocabulary
    """
    vocab_size = len(vocabulary_inv)
    embedding_weights = np.zeros(shape=(vocab_size+1, w2v_dim), dtype='float32')
    #initialize the first row
    embedding_weights[0] = np.zeros(shape=(w2v_dim,) )

    for idx in range(1, vocab_size):
        embedding_weights[idx] = word_vecs[vocabulary_inv[idx]]
    print("Embedding matrix of size "+str(np.shape(embedding_weights)))
    return embedding_weights


def build_input_data(X, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = [[vocabulary[word] for word in sentence if word in vocabulary] for sentence in X]
    x = pad_sequence(x)
    return x


def w2v_feature_extract(root_path, filename, w2v_path):
    X_train_tid, X_train, y_train, \
    X_dev_tid, X_dev, y_dev, \
    X_test_tid, X_test, y_test, relation = read_corpus(root_path, filename)

    print("text word2vec generation.......")
    vocabulary, word_embeddings = build_vocab_word2vec(X_train + X_dev + X_test, w2v_path=w2v_path)
    pickle.dump(vocabulary, open(root_path + "/vocab.pkl", 'wb'))
    print("Vocabulary size: "+str(len(vocabulary)))

    print("build input data.......")
    X_train = build_input_data(X_train, vocabulary)
    X_dev = build_input_data(X_dev, vocabulary)
    X_test = build_input_data(X_test, vocabulary)

    pickle.dump([X_train_tid, X_train, y_train, word_embeddings, relation], open(root_path+"/train.pkl", 'wb') )
    pickle.dump([X_dev_tid, X_dev, y_dev], open(root_path+"/dev.pkl", 'wb') )
    pickle.dump([X_test_tid, X_test, y_test], open(root_path+"/test.pkl", 'wb') )


if __name__ == "__main__":
    w2v_feature_extract('./twitter15/', "twitter15", "twitter_w2v.bin")
    w2v_feature_extract('./twitter16/', "twitter16", "twitter_w2v.bin")
    w2v_feature_extract('./weibo/', "weibo", "weibo_w2v.bin")



