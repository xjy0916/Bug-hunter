import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, FiLMConv,GATConv,global_mean_pool, GCNConv
from transformers import BertModel

def graph_construction(features, role, seq_length, graph_edge, ifcuda):
    speaker_type_map = [[0, 1], [2, 3]]  # 四种关系0,0->0; 0,1->1; 1,0->2; 1,1->3
    assert len(graph_edge) == features.size(0)
    node_features, edge_index, edge_type, graph_batch = [], [], [], []

    batch_size = features.size(0)
    #  将一个batch中的所有图组合成一个大图
    node_sum = 0
    for i in range(batch_size):
        node_features.append(features[i, :seq_length[i], :])

        for src_node, tgt_node in graph_edge[i]:
            edge_index.append(torch.tensor([src_node + node_sum, tgt_node + node_sum]))
            src_role = role[i][src_node]
            tgt_role = role[i][tgt_node]
            edge_type.append(speaker_type_map[src_role][tgt_role])

        node_sum += seq_length[i]
        graph_batch.append(torch.tensor([i] * seq_length[i]))

    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.stack(edge_index).t().contiguous()
    edge_type = torch.tensor(edge_type)
    graph_batch = torch.cat(graph_batch)

    if ifcuda:
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
        edge_type = edge_type.cuda()
        graph_batch = graph_batch.cuda()

    return node_features, edge_index, edge_type, graph_batch



class GraphNetwork(torch.nn.Module):
    def __init__(self, feature_dim, graph_class_num, num_relations, hidden_size=64, dropout=0.5, ifcuda=True):
        super(GraphNetwork, self).__init__()

        self.conv1 = FiLMConv(feature_dim, hidden_size, num_relations=num_relations)
        self.conv2 = GATConv(feature_dim, hidden_size)

        self.linear = nn.Linear(2 * (2 * hidden_size), hidden_size)
        self.graph_smax_fc = nn.Linear(hidden_size, graph_class_num)
        self.dropout = nn.Dropout(dropout)
        self.ifcuda = ifcuda

    def forward(self, x, edge_index, edge_type, graph_batch):
        out1 = F.relu(self.conv1(x, edge_index, edge_type))
        out2 = F.relu(self.conv2(x, edge_index))
        features = torch.cat((out1, out2), dim=-1)

        # max_feature -> batch * (feature_dim + hidden_size)
        max_features = global_max_pool(features, graph_batch)
        # sum_features -> batch * (feature_dim + hidden_size)
        mean_features = global_mean_pool(features, graph_batch)
        hidden = F.relu(self.linear(self.dropout(torch.cat([max_features, mean_features], dim=-1))))
        hidden = self.graph_smax_fc(self.dropout(hidden))
        graph_log_prob = F.log_softmax(hidden, -1)

        return graph_log_prob


class BugHunter(nn.Module):

    def __init__(self, pretrained_model, D_bert, filter_sizes, filter_num, D_cnn, D_graph, n_speakers,
                 graph_class_num=2, dropout=0.5, ifcuda=True):
        super(BugHunter, self).__init__()
        self.ifcuda = ifcuda
        self.pretrained_bert = BertModel.from_pretrained(pretrained_model)
        self.cnn_encoder = nn.ModuleList([nn.Conv1d(D_bert, filter_num, size) for size in filter_sizes])
        self.fc_cnn = nn.Linear(len(filter_sizes) * filter_num, D_cnn)

        n_relations = n_speakers ** 2
        self.graph_net = GraphNetwork(D_cnn, graph_class_num, n_relations, D_graph, dropout, self.ifcuda)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids, attention_mask, role, seq_length, graph_edge):
        # 1. utterance embedding
        b_, s_, w_ = input_ids.size()
        # -> (batch*sen_num) * word_num
        i_ids = input_ids.view(-1, w_)
        t_ids = token_type_ids.view(-1, w_)
        a_ids = attention_mask.view(-1, w_)

        # word_output = (batch*sen_num) * word_num * D_bert
        word_output = self.pretrained_bert(input_ids=i_ids, token_type_ids=t_ids, attention_mask=a_ids)[0]
        # 对padding的向量进行mask
        mask = a_ids.unsqueeze(-1).expand_as(word_output)  # mask -> (batch*sen_num) * word_num * D_bert
        word_output = word_output.masked_fill(mask == 0, 0.)

        # -> (batch*sen_num) * dim * word_num
        word_output = word_output.transpose(-2, -1).contiguous()


        convoluted = [F.relu(conv(word_output)) for conv in self.cnn_encoder]
        pooled = [F.max_pool1d(c, c.size(-1)).squeeze(-1) for c in convoluted]
        concated = torch.cat(pooled, -1)
        features = F.relu(self.fc_cnn(self.dropout(concated)))  # (num_utt * batch, dim) -> (num_utt * batch, dim)
        features = features.view(b_, s_, -1)  # (num_utt * batch, D_cnn) -> (batch, num_utt, D_cnn)


        # 2. graph construction
        node_features, edge_index, edge_type, graph_batch = \
            graph_construction(features, role, seq_length, graph_edge, self.ifcuda)

        # 3. graph embedding and classification
        graph_log_prob = self.graph_net(node_features, edge_index, edge_type, graph_batch)
        return graph_log_prob
