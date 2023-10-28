import torch
import torch.nn as nn
import torch.nn.functional as F



class TextBILSTM_attention(nn.Module):

    def __init__(self, char_embediding_dim, hidden_dim, out_embedding_dim, num_layers, dropout=0.5):
        super(TextBILSTM_attention, self).__init__()
        # attention layer
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        # 双层lstm
        self.lstm_net = nn.LSTM(char_embediding_dim, hidden_dim,
                                num_layers=num_layers, dropout=dropout,
                                bidirectional=True)
        # FC层
        self.fc_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_embedding_dim)
        )

    def attention_net_with_w(self, lstm_out, lstm_hidden):
        '''
        :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
        :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
        :return: [batch_size, n_hidden]
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

    def forward(self, input):

        # input : [len_seq, batch_size, embedding_dim]
        input = input.permute(1, 0, 2)
        output, (final_hidden_state, final_cell_state) = self.lstm_net(input)
        # output : [batch_size, len_seq, n_hidden * 2]
        output = output.permute(1, 0, 2)
        # final_hidden_state : [batch_size, num_layers * num_directions, n_hidden]
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        atten_out = self.attention_net_with_w(output, final_hidden_state)
        return self.fc_out(atten_out)




class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        return torch.max_pool1d(x, kernel_size=x.shape[-1])


class TextRCNN(nn.Module):
    def __init__(self, D_bert, hidden_size, dropout=0.5):
        super(TextRCNN, self).__init__()
        self.lstm = nn.LSTM(input_size=D_bert, hidden_size=hidden_size,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.globalmaxpool = GlobalMaxPool1d()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(D_bert + 2 * hidden_size, hidden_size)


    def forward(self, x):
        # x: [batch,seq_max_length,D_bert]
        last_hidden_state, (c, h) = self.lstm(x)  # last_hidden_state: [batch,seq_max_length, hidden_size * num_bidirectional]
        out = torch.cat((x, last_hidden_state),
                        2)  # out: [batch,seq_max_length,D_bert + hidden_size * num_bidirectional]
        # print(out.shape)
        out = F.relu(out)
        out = out.permute(0, 2, 1)  # out: [batch,D_bert + hidden_size * num_bidirectional,seq_max_length]
        out = self.globalmaxpool(out).squeeze(-1)  # out: [batch,D_bert + hidden_size * num_bidirectional]
        out = self.fc(self.dropout(out))
        # print("out=",out.shape)
        return out

import collections
from torch.nn import BatchNorm1d
from torch_geometric.nn import RGATConv, FiLMConv
from utils import topk, filter_adjacency
from torch_scatter import scatter_max, scatter_mean

class RGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, dropout=0.5, n_layers=1, n_heads=1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.convs.append(RGATConv(in_channels=in_channels, out_channels=hidden_channels, num_relations=num_relations,
                                   heads=n_heads))
        for i in range(n_layers - 2):
            self.convs.append(RGATConv(hidden_channels, hidden_channels, num_relations,
                                       heads=n_heads))
        self.convs.append(RGATConv(hidden_channels, hidden_channels, num_relations,
                                   heads=n_heads))
        self.fc1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            # x = F.dropout(x, p=0.2, training=self.training)
            # x = self.lin1(x)
        avg_node_features = torch.mean(x, dim=0)
        max_node_features = torch.max(x, dim=0).values
        graph_vector = torch.cat((avg_node_features, max_node_features), dim=-1)
        graph_vector = torch.tanh(graph_vector)
        graph_vector = F.relu(self.fc1(self.dropout(graph_vector)))
        graph_vector = self.fc2(self.dropout(graph_vector))
        # print("graph_vector=",graph_vector)
        # print("log_=",F.log_softmax(graph_vector,dim=-1))
        return F.sigmoid(graph_vector)
        # return F.log_softmax(graph_vector,dim=-1)



class GNNFilm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,num_relations, n_layers, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(FiLMConv(in_channels, hidden_channels, num_relations))
        for _ in range(n_layers - 1):
            self.convs.append(FiLMConv(hidden_channels, hidden_channels, num_relations))
        self.norms = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.norms.append(BatchNorm1d(hidden_channels))
        self.lin_l = nn.Sequential(collections.OrderedDict([
            ('lin1', nn.Linear(hidden_channels, int(hidden_channels // 4), bias=True)),
            ('lrelu', nn.LeakyReLU(0.2)),
            ('lin2', nn.Linear(int(hidden_channels // 4), out_channels, bias=True))]))

    def forward(self, x, edge_index, edge_type):
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x, edge_index, edge_type))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_l(x)
        return x



"""SAGPool网络相关层函数
"""
class GraphConvolution(nn.Module):
    """图卷积层
    """

    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积层
            SAGPool中使用图卷积层计算每个图中每个节点的score
            Inputs:
            -------
            input_dim: int, 输入特征维度
            output_dim: int, 输出特征维度
            use_bias: boolean, 是否使用偏置
        """

        super(GraphConvolution, self).__init__()

        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))

        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.__init_parameters()

    def __init_parameters(self):
        """初始化权重和偏置
        """

        nn.init.kaiming_normal_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

        return

    def forward(self, adjacency, X):
        """图卷积层前馈
            Inputs:
            -------
            adjacency: tensor in shape [num_nodes, num_nodes], 邻接矩阵
            X: tensor in shape [num_nodes, input_dim], 节点特征
            Output:
            -------
            output: tensor in shape [num_nodes, output_dim], 输出
        """

        support = torch.mm(X, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias

        return output


# ----------------------------------------------------------------------------
# 读出操作

def global_max_pool(X, graph_indicator):
    """全局最大值池化
        计算图中所有节点特征的全局最大值作为图特征
        Inputs:
        -------
        X: tensor, 所有图所有节点(可能是池化后)的特征
        graph_indicator: tensor, 指明每个节点所属的图
        Output:
        -------
        max_pool: tensor, 全局最大值池化后的图特征
    """

    num_graphs = graph_indicator.max().item() + 1
    max_pool = scatter_max(X, graph_indicator, dim=0, dim_size=num_graphs)[0]

    return max_pool


def global_avg_pool(X, graph_indicator):
    """全局平均值池化
        计算图中所有节点特征的全局平均值作为图特征
        Inputs:
        -------
        X: tensor, 所有图所有节点(可能是池化后)的特征
        graph_indicator: tensor, 指明每个节点所属的图
        Output:
        -------
        avg_pool: tensor, 全局平均值池化后的图特征
    """

    num_graphs = graph_indicator.max().item() + 1
    avg_pool = scatter_mean(X, graph_indicator, dim=0, dim_size=num_graphs)

    return avg_pool


class Readout(nn.Module):
    """图读出操作
    """

    def forward(self, X, graph_indicator):
        """图读出操作前馈
            拼接每个图的全局最大值特征和全局平局值特征作为图特征
            Inputs:
            -------
            X: tensor, 所有图所有节点(可能是池化后)的特征
            graph_indicator: tensor, 指明每个节点所属的图
            Output:
            -------
            readout: tensor, 读出操作获得的图特征
            """

        readout = torch.cat([
            global_avg_pool(X, graph_indicator),
            global_max_pool(X, graph_indicator)
        ], dim=1)

        return readout


# ----------------------------------------------------------------------------
# 自注意力机制池化层


class SelfAttentionPooling(nn.Module):
    """自注意力机制池化层
    """

    def __init__(self, input_dim, keep_ratio):
        """自注意力机制池化层
            使用GCN计算每个图中的每个节点的score作为重要性,
            筛选每个图中topk个重要的节点, 获取重要节点的邻接矩阵,
            使用重要节点特征和邻接矩阵用于后续操作
            Inputs:
            -------
            input_dim: int, 输入的节点特征数量
            keep_ratio: float, 每个图中topk的节点占所有节点的比例
        """

        super(SelfAttentionPooling, self).__init__()

        self.keep_ratio = keep_ratio
        self.act = nn.Tanh()
        self.gcn = GraphConvolution(input_dim, 1)

        return

    def forward(self, X, adjacency, graph_batch):
        """自注意力机制池化层前馈
            Inputs:
            -------
            X: tensor, 节点特征
            adjacency: tensor, 输入节点构成的邻接矩阵
            graph_nbatch: tensor, 指明每个节点所属的图
        """

        # 计算每个图中每个节点的重要性
        node_score = self.gcn(adjacency, X)
        node_score = self.act(node_score)

        # 获得每个途中topk和重要节点
        mask = topk(node_score, graph_batch, self.keep_ratio)
        # 获得重要节点特征, 指明重要节点所属的图, 生成由重要节点构成的邻接矩阵
        mask_X = X[mask] * node_score.view(-1, 1)[mask]
        mask_graph_batch = graph_batch[mask]
        mask_adjacency = filter_adjacency(adjacency, mask)

        return mask_X, mask_adjacency, mask_graph_batch







import torch
from torch import Tensor
from torch.nn import Parameter as Param
from torch_sparse import SparseTensor, matmul
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
import math
from typing import Any



def uniform(size: int, value: Any):
    if isinstance(value, Tensor):
        bound = 1.0 / math.sqrt(size)
        value.data.uniform_(-bound, bound)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            uniform(size, v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            uniform(size, v)



class GatedGraphConv(MessagePassing):
    r"""The gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{h}_i^{(0)} &= \mathbf{x}_i \, \Vert \, \mathbf{0}

        \mathbf{m}_i^{(l+1)} &= \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot
        \mathbf{\Theta} \cdot \mathbf{h}_j^{(l)}

        \mathbf{h}_i^{(l+1)} &= \textrm{GRU} (\mathbf{m}_i^{(l+1)},
        \mathbf{h}_i^{(l)})

    up to representation :math:`\mathbf{h}_i^{(L)}`.
    The number of input channels of :math:`\mathbf{x}_i` needs to be less or
    equal than :obj:`out_channels`.
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1`)

    Args:
        out_channels (int): Size of each output sample.
        num_layers (int): The sequence length :math:`L`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`

    """
    def __init__(self, in_channels:int, out_channels: int, num_layers: int, aggr: str = 'add',
                 bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        # self.weight = Param(Tensor(num_layers, out_channels, out_channels))
        self.gat = GATv2Conv(out_channels, out_channels)
        self.rnn = nn.GRUCell(out_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        # uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            y = torch.cat([x, zero], dim=1)
        else:
            y = x
        for i in range(self.num_layers):
            # m = torch.matmul(y, self.weight[i])
            # # propagate_type: (x: Tensor, edge_weight: OptTensor)
            # m = self.propagate(edge_index, x=m, edge_weight=edge_weight,
            #                    size=None)
            m = self.gat(x=x, edge_index=edge_index)
            y = self.rnn(m, y)
        return y

    # def message(self, x_j: Tensor, edge_weight: OptTensor):
    #     return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
    #
    # def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
    #     return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'num_layers={self.num_layers})')

