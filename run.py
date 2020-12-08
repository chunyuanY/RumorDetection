import os
import pickle
import torch
from sklearn.metrics import classification_report
from model.GLAN import GLAN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_dataset(task):
    X_train_tid, X_train_source, X_train_replies, y_train, word_embeddings, graph = pickle.load(open("dataset/"+task+"/train.pkl", 'rb'))
    X_dev_tid, X_dev_source, X_dev_replies, y_dev = pickle.load(open("dataset/"+task+"/dev.pkl", 'rb'))
    X_test_tid, X_test_source, X_test_replies, y_test = pickle.load(open("dataset/"+task+"/test.pkl", 'rb'))
    config['embedding_weights'] = word_embeddings
    print("#nodes: ", graph.num_nodes)
    return X_train_tid, X_train_source, X_train_replies, y_train, \
           X_dev_tid, X_dev_source, X_dev_replies, y_dev, \
           X_test_tid, X_test_source, X_test_replies, y_test, graph


def train_and_test(model, task):
    model_suffix = model.__name__.lower().strip("text")
    config['save_path'] = 'checkpoint/weights.best.' + task + "." + model_suffix

    X_train_tid, X_train_source, X_train_replies, y_train, \
    X_dev_tid, X_dev_source, X_dev_replies, y_dev, \
    X_test_tid, X_test_source, X_test_replies, y_test, graph = load_dataset(task)

    nn = model(config, graph)
    # nn.fit(X_train_tid, X_train_source, X_train_replies, y_train,
    #        X_dev_tid, X_dev_source, X_dev_replies, y_dev)

    print("================================")
    nn.load_state_dict(torch.load(config['save_path']))
    y_pred = nn.predict(X_test_tid, X_test_source, X_test_replies)
    print(classification_report(y_test, y_pred, target_names=config['target_names'], digits=3))


config = {
    'lr':1e-3,
    'reg':0,
    'batch_size':16,
    'nb_filters':100,
    'kernel_sizes':[3, 4, 5],
    'dropout':0.5,
    'maxlen':50,
    'epochs':30,
    'num_classes':4,
    'target_names':['NR', 'FR', 'UR', 'TR']
}


if __name__ == '__main__':
    task = 'twitter15'
    # task = 'twitter16'
    # task = 'weibo'
    print("task: ", task)

    if task == 'weibo':
        config['num_classes'] = 2
        config['batch_size'] = 64
        config['reg'] = 1e-5
        config['target_names'] = ['NR', 'FR']

    model = GLAN
    train_and_test(model, task)

