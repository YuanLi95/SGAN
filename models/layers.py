import torch
import torch.nn as nn
import torch.nn.functional as F
# import  torch.functional as F
import math
from layers.dynamic_rnn import DynamicLSTM
from  models.transformer import MultiHeadAttention
from torch.autograd import Variable
import numpy as np
import networkx as nx
import  time


class GraphAttentionLayer(nn.Module):
    def __init__(self, opt,in_features, hidden_features, dropout=0.5,n_heads=2, alpha=0.2, concat=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.hidden_dim = opt.hidden_dim
        self.use_edge_weight = opt.use_edge_weight
        self.alpha = alpha
        self.concat = concat
        self.edge_embedding_dim = opt.dependency_edge_dim
        if self.use_edge_weight=="yes":
            self.edge_hidden_dim = opt.hidden_dim
            self.edge_mapping = nn.Linear(in_features=self.edge_embedding_dim, out_features=self.edge_hidden_dim,
                                          bias=True)
            self.edge_weight_liner = nn.Linear(in_features=self.edge_hidden_dim,out_features=1,bias=True
                                               )

        self.layer_normal = nn.LayerNorm(in_features)
        self.attention_gate = Attention_Gate(in_features)
        self.W = nn.Parameter(torch.zeros(size=(in_features, self.hidden_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.concat_mapping = nn.Linear(in_features=2 * self.hidden_dim,out_features = self.hidden_dim,bias=False)

        self.a = nn.Parameter(torch.zeros(size=(self.hidden_dim*2 , 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.tanh = nn.Tanh()
        # self.relu = nn.ReLU()


    def forward(self, input, adj,edge_embedding,flage=True):

        h = torch.matmul(input, self.W)
        batch_size = h.size()[0]
        token_lenth = h.size()[1]
        # h.repeat_interleave(repeats=token_lenth, dim=2).view(batch_size, token_lenth * token_lenth, -1)    [bacth_szie,token_len*token_len,embedding_dem]

        a_input = torch.cat([h.repeat_interleave(repeats=token_lenth,dim=2).view(batch_size,token_lenth * token_lenth, -1), h.repeat_interleave(token_lenth, dim=0).view(batch_size,token_lenth * token_lenth, -1)], dim=2).view(batch_size,token_lenth,-1, 2 * self.hidden_dim)  # 这里让每两个节点的向量都连接在一起遍历一次得到 bacth* N * N * (2 * out_features)大小的矩阵

        e = self.leakyrelu(torch.matmul(a_input, self.a))
        if self.use_edge_weight == "yes":
            edge_embedding = self.edge_mapping(edge_embedding)
            edge_weight = torch.relu(edge_embedding)
            edge_weight = self.edge_weight_liner(edge_embedding)
            edge_weight = torch.sigmoid(edge_weight)
            edge_weight = edge_weight.reshape(batch_size,token_lenth,token_lenth,-1)
            # edge_weight = edge_weight.squeeze(3)
            # e_final = e.squeeze(3)
            # print(edge_weight[0])
            # print(e[0])
            e_final= e.mul(edge_weight).squeeze(3)
            # print(e_final[0])

            # zero_vec = -9e15 * torch.ones_like(edge_weight)
            # attention_edge = torch.where(adj > 0, edge_weight, zero_vec)

            zero_vec_text = -9e15 * torch.ones_like(e_final)
            attention_text = torch.where(adj > 0, e_final, zero_vec_text)
            # print(attention.shape)

            attention = torch.softmax(attention_text, dim=2)  # 这里是一个非线性变换，将有权重的变得更趋近于1，没权重的为0
            # attention_edge = torch.softmax(attention_edge, dim=2)


        else:
            e_final = e.squeeze(3)
        # e = self.relu(torch.matmul(a_input, self.a).squeeze(3))
        # print(e.shape)


            zero_vec_text = -9e15 * torch.ones_like(e_final)
            attention = torch.where(adj > 0, e_final, zero_vec_text)
            attention = torch.softmax(attention, dim=2)
            # print(attention.shape)



        attention = F.dropout(attention, self.dropout, training=self.training)
        # if flage ==False:   #False为测试模式
        #     print(adj)
        #     time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        #     numpy_attention = attention.cpu().numpy()
        #     np.save("./attention_numpy/numpy_attention_{0}.npy".format(time_str), numpy_attention)
        #     exit()
        h_prime = torch.matmul(attention,h)
        # h_prime = torch.norm(h_prime)
        h_prime = torch.add(h,h_prime)
        #用gate 更新
        # h_prime = self.attention_gate(h_prime,h)
        # exit()


        # h_prime = self.layer_normal(h_prime+residual)

        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime


class Attention_Gate(nn.Module):
    def __init__(self, in_features,dropout=0.3):
        super(Attention_Gate, self).__init__()
        self.hidden_dim = in_features

        self.liner_context_weight = nn.Linear(self.hidden_dim*2,1)
        self.liner_answer_weight = nn.Linear(self.hidden_dim*2,1)
    def forward(self, self_vector, generation_vector):
        batch,seq,hidden_dim = self_vector.shape
        self_vector = self_vector.reshape(-1,self.hidden_dim)
        generation_vector = generation_vector.reshape(-1, self.hidden_dim)
        common_vector = torch.cat([self_vector,generation_vector],dim=-1)

        g_m = torch.sigmoid(self.liner_context_weight(common_vector))
        g_a = torch.sigmoid(self.liner_answer_weight(common_vector))
        out_vector = g_m*self_vector+g_a*generation_vector
        out_vector = out_vector.reshape(batch,seq,hidden_dim)

        return out_vector


class GraphDotProductLayer(nn.Module):
    def __init__(self,opt, in_features, out_features, dropout, alpha,n_heads, concat=False):
        super(GraphDotProductLayer, self).__init__()
        self.concat = concat
        self.out_features = out_features
        self.in_features=in_features
        self.W = nn.Parameter(torch.zeros(size=(in_features,out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.layer_normal = nn.LayerNorm(in_features)
        # self.relu = nn.ReLU()
        self.mutiheadattention = MultiHeadAttention(in_features*2,in_features*2,out_features,num_heads=10,dropout=dropout)

    def forward(self, input, adj,aspect_double_idx):
        # h = torch.matmul(input, self.W)
        batch_size = input.size()[0]
        token_lenth = input.size()[1]
        aspect_hidden_state = []
        residual = input
        for i in range(batch_size):
            aspect_begin_idx = aspect_double_idx[i, 0]
            aspect_end_idx = aspect_double_idx[i, 1]
            aspect_hidden_state.append((input[i,aspect_begin_idx,:]+input[i,aspect_end_idx,:])/2)
        aspect_hidden_state = torch.stack(aspect_hidden_state)
        # print(aspect_hidden_state.shape)
        aspect_hidden_state = aspect_hidden_state.repeat_interleave(token_lenth,dim=0).view(batch_size,token_lenth,-1)
        transformer_input = torch.cat([input,aspect_hidden_state],dim=2)
        # print(transformer_input.shape)

        output,attention = self.mutiheadattention(transformer_input,transformer_input,transformer_input)
        # e =attention
        e = self.leakyrelu(attention)
        # e =

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        # print(value)
        h_prime = torch.bmm(attention,output)
        h_prime = self.layer_normal(residual+h_prime)
        # print(h_prime)
        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime


# class Confusion_Attention(nn.Module):
#     def __init__(self,opt):
#         self.conv_to_gat = nn.ModuleList([nn.Linear(opt.hidden_dim, opt.hidden_dim, bias=True) for i in range(3)])
#
#         self.gat_to_conv = nn.ModuleList([nn.Linear(opt.hidden_dim, opt.hidden_dim, bias=True) for i in range(3)])


class GAT(nn.Module):
    def __init__(self,opt, n_feat, n_hid, dropout, alpha, n_heads):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        # 定义multi-head的图注意力层
        self.attentions = [GraphAttentionLayer(opt,n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        # self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj,edge_embedding,aspect_double_idx):
        # edge_embedding 含有dependency type的embedding  shape [batch_size, seq*seq embedding_dim]
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = torch.mean(torch.stack([att(x, adj,edge_embedding) for att in self.attentions]),dim=0)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        # x = F.relu(self.out_att(x, adj))  # 输出并激活

        return x  # log_softmax速度变快，保持



# class GAT_EDGE(nn.Module):
#     def __init__(self,opt, n_feat, n_hid, dropout, alpha, n_heads):
#         """Dense version of GAT
#         n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
#         从不同的子空间进行抽取特征。
#         """
#         super(GAT_EDGE, self).__init__()
#         self.dropout = dropout
#
#         # 定义multi-head的图注意力层
#         self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
#                            range(n_heads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
#         self.
#         # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
#         # self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)
#
#     def forward(self, x, adj,aspect_double_idx):
#         x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
#         x = torch.mean(torch.stack([att(x, adj) for att in self.attentions]),dim=0)  # 将每个head得到的表示进行拼接
#
#         x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
#         # x = F.relu(self.out_att(x, adj))  # 输出并激活
#
#         return x  # log_softmax速度变快，保持






