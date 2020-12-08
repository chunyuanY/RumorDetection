import torch
import torch.nn as nn
import torch.nn.init as init
from .TransformerBlock import MultiheadAttention
from .NeuralNetwork import NeuralNetwork
import torch.nn.functional as F
from .GAT import GATConv
import torch_geometric.utils as utils


class Attention(nn.Module):

    def __init__(self, in_features, hidden_size):
        super(Attention, self).__init__()
        self.linear1 = nn.Linear(in_features*2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, K, V, mask = None):
        '''
        :param K: (batch_size, d)
        :param V: (batch_size, hist_len, d)
        :return: (batch_size, d)
        '''
        K = K.unsqueeze(dim=1).expand(V.size())
        fusion = torch.cat([K, V], dim=-1)

        fc1 = self.activation(self.linear1(fusion))
        score = self.linear2(fc1)

        if mask is not None:
            mask = mask.unsqueeze(dim=-1)
            score = score.masked_fill(mask, -2 ** 32 + 1)

        alpha = F.softmax(score, dim=1)
        alpha = self.dropout(alpha)
        att = (alpha * V).sum(dim=1)
        return att


class GLAN(NeuralNetwork):

    def __init__(self, config, graph):
        super(GLAN, self).__init__()
        self.config = config
        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape
        maxlen = config['maxlen']
        dropout_rate = config['dropout']
        alpha = 0.4

        self.graph = graph
        self.word_embedding = nn.Embedding(V, D, padding_idx=0, _weight=torch.from_numpy(embedding_weights))
        self.user_tweet_embedding = nn.Embedding(graph.num_nodes, 300, padding_idx=0)

        self.mh_attention = MultiheadAttention(input_size=300, output_size=300)
        self.linear_fuse = nn.Linear(600, 1)

        self.gnn1 = GATConv(300, out_channels=8, dropout=dropout_rate, heads=8, negative_slope=alpha)
        self.gnn2 = GATConv(64, 300, dropout=dropout_rate, concat=False, negative_slope=alpha)

        self.attention = Attention(300, 300)
        self.convs_source = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.convs_replies = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

        self.fc_out = nn.Sequential(
            nn.Linear(600, 300),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(300, config['num_classes'])
        )
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.xavier_normal_(self.user_tweet_embedding.weight)
        init.xavier_normal_(self.linear_fuse.weight)
        for name, param in self.fc_out.named_parameters():
            if name.__contains__("weight"):
                init.xavier_normal_(param)

    def source_encoder(self, X_text):
        X_text = X_text.permute(0, 2, 1)

        conv_block = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs_source, self.max_poolings)):
            act = self.relu(Conv(X_text))
            pool = max_pooling(act)
            pool = torch.squeeze(pool)
            conv_block.append(pool)
        conv_feature = torch.cat(conv_block, dim=1)
        features = self.dropout(conv_feature)
        return features

    def replies_encoder(self, X_replies):
        bsz, src_len, num_words, dim = X_replies.size()
        X_replies = X_replies.view(bsz*src_len, num_words, dim)
        X_replies = X_replies.permute(0, 2, 1)

        conv_block = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs_replies, self.max_poolings)):
            act = self.relu(Conv(X_replies))
            pool = max_pooling(act)
            pool = torch.squeeze(pool)
            conv_block.append(pool)
        conv_feature = torch.cat(conv_block, dim=1)
        features = self.dropout(conv_feature)
        features = features.view(bsz, src_len, features.size(-1))
        return features


    def local_attention_network(self, X_source, X_replies, mask):
        X_srouce_feat = self.source_encoder(X_source)

        X_replies_feat = self.replies_encoder(X_replies)
        X_replies_feat = self.mh_attention(X_replies_feat, X_replies_feat, X_replies_feat)

        X_att = self.attention(X_srouce_feat, X_replies_feat, mask=mask)
        X_fuse = torch.cat([X_srouce_feat, X_att], dim=-1)

        alpha = torch.sigmoid(self.linear_fuse(X_fuse))
        X_local = alpha * X_srouce_feat + (1 - alpha) * X_att
        return X_local

    def global_graph_encoding(self, X_tid):
        node_init_feat = self.user_tweet_embedding.weight
        node_init_feat = self.dropout(node_init_feat)
        edge_index = self.graph.edge_index.cuda()
        edge_weight = self.graph.edge_weight.cuda()

        edge_index, edge_weight = utils.dropout_adj(edge_index, edge_weight, training=self.training)

        node_rep1 = self.gnn1(node_init_feat, edge_index, edge_weight)
        node_rep1 = self.dropout(node_rep1)

        graph_output = self.gnn2(node_rep1, edge_index, edge_weight)
        return graph_output[X_tid]

    def forward(self, X_tid, X_source, X_replies):
        mask = ((X_replies != 0).sum(dim=-1) == 0)
        X_source = self.word_embedding(X_source) # (N*C, W, D)
        X_replies = self.word_embedding(X_replies)

        X_local = self.local_attention_network(X_source, X_replies, mask)
        X_global = self.global_graph_encoding(X_tid)

        X_feat = torch.cat([X_local, X_global], dim=-1)
        X_feat = self.dropout(X_feat)

        output = self.fc_out(X_feat)
        return output
