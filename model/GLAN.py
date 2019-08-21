import torch
import torch.nn as nn
import torch.nn.init as init
from model.GAT import GAT
from model.TransformerBlock import TransformerBlock
from .NeuralNetwork import NeuralNetwork


class GLAN(NeuralNetwork):

    def __init__(self, config, adj):
        super(GLAN, self).__init__()
        self.config = config
        self.uV = adj.shape[0]
        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape
        maxlen = config['maxlen']
        dropout_rate = config['dropout']

        self.mh_attention = TransformerBlock(input_size=300)
        self.word_embedding = nn.Embedding(V, D, padding_idx=0, _weight=torch.from_numpy(embedding_weights))

        self.relation_embedding = GAT(nfeat=300, uV=self.uV, adj=adj)

        self.convs = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(600, 300)
        self.fc2 = nn.Linear(300, config['num_classes'])

        self.init_weight()
        print(self)

    def init_weight(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)


    def forward(self, X_tid, X_text):
        X_text = self.word_embedding(X_text) # (N*C, W, D)
        X_text = self.mh_attention(X_text, X_text, X_text)
        X_text = X_text.permute(0, 2, 1)

        rembedding = self.relation_embedding(X_tid)

        conv_block = [rembedding]
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(X_text))
            pool = max_pooling(act)
            pool = torch.squeeze(pool)
            conv_block.append(pool)
        conv_feature = torch.cat(conv_block, dim=1)
        features = self.dropout(conv_feature)

        a1 = self.relu(self.fc1(features))
        d1 = self.dropout(a1)

        output = self.fc2(d1)
        return output



# weibo
#         precision    recall  f1-score   support
#
#     NR    0.94867   0.94329   0.94597       529
#     FR    0.94275   0.94818   0.94545       521
#
# accuracy                        0.94571      1050
# macro avg    0.94571   0.94573   0.94571      1050
# weighted avg    0.94573   0.94571   0.94572      1050



# twitter 15
#         precision    recall  f1-score   support
#
#     NR    0.90805   0.94048   0.92398        84
#     FR    0.91667   0.91667   0.91667        84
#     UR    0.84706   0.85714   0.85207        84
#     TR    0.95000   0.90476   0.92683        84
#
# accuracy                        0.90476       336
# macro avg    0.90544   0.90476   0.90489       336
# weighted avg    0.90544   0.90476   0.90489       336



# twitter 16
#         precision    recall  f1-score   support
#
#     NR    0.95349   0.89130   0.92135        46
#     FR    0.81132   0.93478   0.86869        46
#     UR    0.90000   0.80000   0.84706        45
#     TR    0.95833   0.97872   0.96842        47
#
# accuracy                        0.90217       184
# macro avg    0.90579   0.90120   0.90138       184
# weighted avg    0.90610   0.90217   0.90204       184

