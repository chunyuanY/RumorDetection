import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class SpecialSpmmFunction(torch.autograd.Function):
    """
    Special function for only sparse region backpropataion layer.
    """
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]

        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data)

        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()


    def forward(self, input, adj):
        N = input.size()[0]
        edge = torch.LongTensor(adj.nonzero())

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1)).cuda())
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):

    def __init__(self, nfeat, uV, adj, hidden=16, nb_heads=8, n_output=300, dropout=0.5, alpha=0.3):
        """Sparse version of GAT."""
        super(GAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.uV = uV
        self.adj = adj
        self.user_tweet_embedding = nn.Embedding(self.uV, 300, padding_idx=0)
        init.xavier_uniform_(self.user_tweet_embedding.weight)

        self.attentions = nn.ModuleList([SpGraphAttentionLayer(in_features = nfeat,
                                                        out_features= hidden,
                                                        dropout=dropout,
                                                        alpha=alpha,
                                                        concat=True) for _ in range(nb_heads)])
        
        self.out_att = SpGraphAttentionLayer(hidden * nb_heads,
                                              n_output,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, X_tid):
        X = self.user_tweet_embedding(torch.arange(0, self.uV).long().cuda())
        X = self.dropout(X)

        X = torch.cat([att(X, self.adj) for att in self.attentions], dim=1)
        X = self.dropout(X)

        X = F.elu(self.out_att(X, self.adj))
        X_ = X[X_tid]
        return X_

