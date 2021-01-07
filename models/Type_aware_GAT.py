import torch
import torch.nn as nn
import torch.nn.functional as F
# import  torch.functional as F
import math
from layers.dynamic_rnn import DynamicLSTM

from torch.autograd import Variable
import numpy as np
import networkx as nx
import  time
from .layers import GraphDotProductLayer,Attention_Gate
import copy

class Trans_Gat_Layer(nn.Module):
    def __init__(self, opt,in_features, hidden_features, dropout,n_heads,edge_embed, alpha=0.2, concat=False):
        super(Trans_Gat_Layer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.hidden_dim = opt.hidden_dim
        self.use_edge_weight = opt.use_edge_weight
        self.alpha = alpha
        self.concat = concat
        self.edge_embed_dropout = nn.Dropout(opt.edge_embed_dropout)
        # self.edge_voc_number = edge_embed.shape[0]
        # self.edge_voc_dim = edge_embed.shape[1])
        self.edge_embed = edge_embed
        self.edge_embedding_dim = opt.dependency_edge_dim
        self.edge_hidden_dim = opt.edge_hidden_dim
        self.edge_mapping = nn.Linear(in_features=self.edge_embedding_dim, out_features=self.hidden_dim,
                                      bias=True)
        self.edge_weight_liner = nn.Linear(in_features=self.hidden_dim,out_features=1,bias=True
                                           )

        self.layer_normal = nn.LayerNorm(in_features)
        self.attention_gate = Attention_Gate(in_features)
        self.W = nn.Parameter(torch.zeros(size=(in_features, self.hidden_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)


        self.edge_to = nn.Linear(in_features=self.hidden_dim,out_features=self.hidden_dim,bias=False
                                           )

        self.a = nn.Parameter(torch.zeros(size=(self.hidden_dim*2 , 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.tanh = nn.Tanh()
        # self.relu = nn.ReLU()



    def forward(self, input, adj,dependency_type_matrix,flage=True):

        h = torch.matmul(input, self.W)
        batch_size = h.size()[0]
        token_lenth = h.size()[1]

        # adj = Doubly_normalization(adj)

        # h.repeat_interleave(repeats=token_lenth, dim=2).view(batch_size, token_lenth * token_lenth, -1)    [bacth_szie,token_len*token_len,embedding_dem]
        dependency_type_matrix = dependency_type_matrix.reshape(batch_size, -1).long()
        edge = self.edge_embed(dependency_type_matrix)
        # print(edge)

        # text = self.batch_nor(text)
        edge_embedding = self.edge_embed_dropout(edge)
        a_input = torch.cat([h.repeat_interleave(repeats=token_lenth,dim=2).view(batch_size,token_lenth * token_lenth, -1), h.repeat_interleave(token_lenth, dim=0).view(batch_size,token_lenth * token_lenth, -1)], dim=2).view(batch_size,token_lenth,-1, 2 * self.hidden_dim)  # 这里让每两个节点的向量都连接在一起遍历一次得到 bacth* N * N * (2 * out_features)大小的矩阵
        # input_cat = a_input.clone()
        # a_input = a_input.transpose(2,3)
        e_text = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze()

        zero_vec_text = -9e15 * torch.ones_like(e_text)
        attention_text = torch.where(adj > 0, e_text, zero_vec_text)
        attention_text= torch.softmax(attention_text, dim=-1)
        h_text = torch.matmul(attention_text,h)



        edge_mapping = self.edge_mapping(edge_embedding)
        edge_mapping = torch.relu(edge_mapping)

        # edge_mapping_2 = edge_mapping.reshape(batch_size,token_lenth,token_lenth,-1).squeeze()
        edge_weight = self.edge_weight_liner(edge_mapping)
        edge_weight = self.leakyrelu(edge_weight)
        edge_weight = edge_weight.reshape(batch_size,token_lenth,token_lenth,-1).squeeze()

        zero_vec_edge = -9e15 * torch.ones_like(edge_weight)
        attention_edge = torch.where(adj > 0, edge_weight, zero_vec_text)
        attention_edge = torch.softmax(attention_edge, dim=-1)
        h_edge = self.edge_to(input)
        h_edge = torch.matmul(attention_edge, h_edge)


        # concat_embedding = torch.cat((h_edge,h_text),dim=-1)
        # concat_t = self.text_sig(concat_embedding)
        # sig_t = torch.sigmoid(concat_t)
        #
        # concat_e = self.edge_sig(concat_embedding)
        # sig_e = torch.sigmoid(concat_e)
        #
        #
        #
        # h_final = sig_t*h_text+sig_e*h_edge
        # attention = sig_t*attention_text+sig_e*attention_edge
        #
        #
        # # attention = F.dropout(attention, self.dropout, training=self.training)
        # #
        # # h_prime = torch.matmul(attention,h)
        # h_final = torch.relu(h_final)

        # h_prime = torch.norm(h_prime)
        # h_prime = torch.add(h,h_prime)
        #用gate 更新
        # h_prime = self.attention_gate(h_prime,h)
        # exit()


        # h_prime = self.layer_normal(h_prime+residual)
        return h_text,h_edge,attention_text,attention_edge



class Muti_GAT(nn.Module):
    def __init__(self,opt, in_features, hidden_features, dropout, alpha, n_heads,edge_embed):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(Muti_GAT, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(opt.hidden_dim, eps=1e-12)
        # self.W_z = nn.Linear(in_features=hidden_features,out_features=1,bias=True)
        # self.U_z = nn.Linear(in_features=hidden_features,out_features=1,bias=True)
        # self.W_r = nn.Linear(in_features=hidden_features, out_features=1, bias=True)
        # self.U_r = nn.Linear(in_features=hidden_features, out_features=1, bias=True)
        # self.W = nn.Linear(in_features=hidden_features,out_features=hidden_features,bias=True)
        # self.U = nn.Linear(in_features=hidden_features,out_features=hidden_features,bias=True)
        #
        self.text_sig = nn.Linear(in_features=2 * opt.hidden_dim, out_features=1, bias=True)

        self.edge_sig = nn.Linear(in_features=2 * opt.hidden_dim, out_features=1, bias=True)
        self.concat_to_finaly = nn.Linear(in_features=2 * opt.hidden_dim, out_features=opt.hidden_dim, bias=True)
        self.sig = nn.Sigmoid()
        self.dropout = dropout
        self.n_heads = n_heads

        # 定义multi-head的图注意力层
        self.attentions = nn.ModuleList([Trans_Gat_Layer(opt,in_features, hidden_features, dropout=dropout,n_heads=n_heads, alpha=alpha,edge_embed = edge_embed, concat=True) for _ in
                           range(n_heads)])
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        # self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)

    def forward(self,x, adj_type,edge_matrix,aspect_double_idx):
        # edge_embedding 含有dependency type的embedding  shape [batch_size, seq*seq embedding_dim]

        h_text = []
        attention_text= []
        h_edge = []
        attention_edge = []
        for i in range(self.n_heads):
            h_text_i, h_edge_i, attention_text_i, attention_edge_i, = self.attentions[i](x,adj_type,edge_matrix,aspect_double_idx)
            h_text.append(h_text_i)
            h_edge.append(h_edge_i)
            attention_text.append(attention_text_i)
            attention_edge.append(attention_edge_i)
        h_text = torch.mean(torch.stack(h_text),dim=0)
        h_edge = torch.mean(torch.stack(h_edge),dim=0)
        attention_text = torch.mean(torch.stack(attention_text), dim=0)
        attention_edge= torch.mean(torch.stack(attention_edge), dim=0)
        attention = torch.mean(torch.stack([attention_text,attention_edge]),dim=0)

        "gate 版本 "
        concat_hidden = torch.cat((h_edge,h_text),dim=-1)
        concat_t = self.text_sig(concat_hidden)
        sig_t = torch.sigmoid(concat_t)

        concat_e = self.edge_sig(concat_hidden)
        sig_e = torch.sigmoid(concat_e)



        h_prime = sig_t*h_text+(1-sig_t)*h_edge
        attention = sig_t*attention_text+sig_e*attention_edge
        "concate 版本"
        # h_prime = torch.cat((h_text,h_edge),dim=-1)
        # h_prime = self.concat_to_finaly(h_prime)


        """
        gate Gat 
        """


        # Z = self.sig(self.W_z(h_prime)+self.U_z(x))
        # R = self.sig(self.W_r(h_prime)+self.U_r(x))
        # H_v = self.W(h_prime)+torch.mul(R,self.U(x))
        # h_prime = (1-Z)*h_prime+Z*self.layer_norm(H_v)
        h_prime = torch.relu(h_prime)

        out_gat = F.dropout(h_prime, self.dropout, training=True)  # dropout，防止过拟合
        # x = F.relu(self.out_att(x, adj))  # 输出并激活

        return h_prime,attention  # log_softmax速度变快，保持


class GAT_GRU(nn.Module):
    def __init__(self,opt, in_features, hidden_features,):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(GAT_GRU, self).__init__()
        # elf.text_gru = DynamicLSTM(
        #     opt.embed_dim, opt.hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True, rnn_type="GRU")
        self.sig = torch.nn.Sigmoid()
        self.drop = torch.nn.Dropout(0.3)
        self.layer_norm = torch.nn.LayerNorm(opt.hidden_dim, eps=1e-12)
        self.W_z = nn.Linear(in_features=hidden_features,out_features=1,bias=True)
        self.U_z = nn.Linear(in_features=hidden_features,out_features=1,bias=True)
        self.W_r = nn.Linear(in_features=hidden_features, out_features=1, bias=True)
        self.U_r = nn.Linear(in_features=hidden_features, out_features=1, bias=True)
        self.W = nn.Linear(in_features=hidden_features,out_features=hidden_features,bias=True)
        self.U = nn.Linear(in_features=hidden_features,out_features=hidden_features,bias=True)

        self.text_sig = nn.Linear(in_features=2 * opt.hidden_dim, out_features=1, bias=True)

        self.edge_sig = nn.Linear(in_features=2 * opt.hidden_dim, out_features=1, bias=True)


    def forward(self,x,h_prime):
        # edge_embedding 含有dependency type的embedding  shape [batch_size, seq*seq embedding_dim]

        Z = self.sig(self.W_z(h_prime)+self.U_z(x))
        R = self.sig(self.W_r(h_prime)+self.U_r(x))
        H_v = torch.tanh(self.W(h_prime)+self.U(torch.mul(R,self.U(x))))
        h_prime = (1-Z)*h_prime+Z*H_v
        h_prime = self.drop(h_prime)

        return h_prime  # log_softmax速度变快，保持


def Doubly_normalization(adj_old):
    """

    :param adj_old:  batch*seq*seq(no padding)
    :return: adj  batch*seq*seq(no padding)
    :function:  双正则化
    """
    E_ij = torch.nn.functional.normalize(adj_old,p=1,dim=2)

    # print(torch.transpose(torch.div(E_ij, torch.sum(E_ij, dim=1).squeeze()), dim0=1, dim1=2).shape)?
    adj_new = torch.matmul(E_ij,torch.transpose(torch.div(E_ij, torch.sum(E_ij, dim=1).unsqueeze(1).repeat([1, adj_old.shape[1], 1])), dim0=1,dim1=2))

    return adj_new

def unpack_and_pack(adj,seq_lenth,seq_list):
    """
    :param adj: 邻接矩阵(padding)  bath*seq_lenth*seq_lenth
    :param seq_lenth: max of seq_list
    :param seq_list:  text_index 的真正长度
    :return: 双正则
    """
    # exit()
    unpack_matrix = []
    for i in range(len(seq_list)):
        unpad_vector = Doubly_normalization(adj[i,:seq_list[i],:seq_list[i]].unsqueeze(dim=0))
        unpack_matrix.append(F.pad(unpad_vector,(0,seq_lenth-seq_list[i],0,seq_lenth-seq_list[i]),mode='constant'))
    pack_matrix =torch.stack(unpack_matrix,dim=0).squeeze()
    return  pack_matrix




class Type_aware_GAT(nn.Module):
    def __init__(self, embedding_matrix,dependency_matrix,position_matrix, opt):
        super(Type_aware_GAT, self).__init__()
        self.opt = opt
        self.number = 2
        self.hop =opt.hop
        self.lamb = 0.1

        self.embed = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float),padding_idx=0)
            # print(dependency_matrix)
        self.edge_embed = nn.Embedding.from_pretrained(
            torch.tensor(dependency_matrix, dtype=torch.float), freeze=False, padding_idx=0)
        self.position_embed = nn.Embedding.from_pretrained(
            torch.tensor(position_matrix, dtype=torch.float), freeze=False, padding_idx=0)
        self.hid_dim = opt.hidden_dim

        self.text_lstm = DynamicLSTM(
            opt.embed_dim, opt.hidden_dim//2, num_layers=1, batch_first=True, bidirectional=True,rnn_type="LSTM")
        # self.text_lstmasp = DynamicLSTM(
        #     opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.text_embed_dropout = nn.Dropout(opt.text_embed_dropout)
        self.edge_embed_dropout = nn.Dropout(opt.edge_embed_dropout)

        self.lcf = opt.lcf
        self.SRD = opt.SRD
        self.biff_layer_number = opt.biff_layer_number
        # self.gat_list=nn.ModuleList([
        #     GraphAttentionLayer(opt,opt.hidden_dim, opt.hidden_dim) for i in range(self.number)]
        # )
        # self.conv_list = nn.ModuleList([nn.Conv1d(opt.hidden_dim, opt.hidden_dim, 3, padding=1) for i in range(self.number)])
        # self.conv_norm = nn.LayerNorm(opt.hidden_dim)
        # self.conv_liner = nn.ModuleList([nn.Linear(opt.hidden_dim, opt.hidden_dim) for i in range(self.number)])

        self.gat_norm = nn.LayerNorm(opt.hidden_dim)
        self.gat_liner = nn.ModuleList([nn.Linear(opt.hidden_dim, opt.hidden_dim) for i in range(self.number)])

        # self.conv2 = nn.Conv1d(opt.hidden_dim, opt.hidden_dim, 3, padding=1)
        self.gat_to_conv_liner = nn.Linear(opt.hidden_dim,opt.hidden_dim,bias=False)
        self.conv_to_gat_liner = nn.Linear(opt.hidden_dim, opt.hidden_dim,bias=False)
        if opt.use_scaled_dot == False:

            # if opt.use_edge_embedding == False:
                GraphLayer= Muti_GAT
                print("111111111")
        else:
            GraphLayer = Muuti_GAT

        self.gat_layer1 = nn.ModuleList(GraphLayer(opt,opt.hidden_dim, opt.hidden_dim,dropout =opt.graph_dropout,alpha=opt.GAT_alpha,n_heads=int(opt.hidden_dim/150),edge_embed=self.edge_embed) for i in range(1))
        self.gat_layer2 = nn.ModuleList(GraphLayer(opt,opt.hidden_dim, opt.hidden_dim,dropout =opt.graph_dropout,alpha=opt.GAT_alpha,n_heads=int(opt.hidden_dim/150),edge_embed=self.edge_embed) for i in range(1))
        self.gat_layer3 = nn.ModuleList(GraphLayer(opt,opt.hidden_dim, opt.hidden_dim,dropout =opt.graph_dropout,alpha=opt.GAT_alpha,n_heads=int(opt.hidden_dim/150),edge_embed=self.edge_embed) for i in range(1))
        # self.gat_layer4 = nn.ModuleList(GraphLayer(opt,opt.hidden_dim, opt.hidden_dim,dropout =opt.graph_dropout,alpha=opt.GAT_alpha,n_heads=int(opt.hidden_dim/150),edge_embed=self.edge_embed) for i in range(1))
        # self.gat_layer5 = nn.ModuleList(GraphLayer(opt,opt.hidden_dim, opt.hidden_dim,dropout =opt.graph_dropout,alpha=opt.GAT_alpha,n_heads=int(opt.hidden_dim/150),edge_embed=self.edge_embed) for i in range(1))

        self.gat_gru = GAT_GRU(opt, opt.hidden_dim, opt.hidden_dim)

        #
        #
        self.gat_ave_pool = torch.nn.AdaptiveAvgPool2d([1,opt.hidden_dim])
        self.gat_max_pool = torch.nn.AdaptiveMaxPool2d([1,opt.hidden_dim])




        # self.poool = nn.MaxPool1d(9)
        # self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)
        # self.fc1 = nn.Linear(opt.hidden_dim*2, opt.polarities_dim)
        self.fc3 = nn.Linear(opt.hidden_dim * 3, opt.polarities_dim)

        # self.fc3 = nn.Linear(opt.hidden_dim*5, opt.polarities_dim)

        self.liner_dropout = nn.Dropout(opt.liner_dropout)

        self.aspect_liner = nn.ModuleList( nn.Linear(in_features= opt.hidden_dim, out_features=opt.hidden_dim) for i in range(self.hop))
        self.concat_liner = nn.ModuleList(nn.Linear(in_features= opt.hidden_dim*2, out_features=1) for i in range(self.hop))
        self.layer_norm1 = torch.nn.LayerNorm(opt.hidden_dim, eps=1e-12)
        self.layer_norm2 = torch.nn.LayerNorm(opt.hidden_dim, eps=1e-12)
        self.layer_norm3 = torch.nn.LayerNorm(opt.hidden_dim, eps=1e-12)
        self.relu = nn.ReLU(inplace=True)
        # self.layer_norm1 = torch.nn.LayerNorm(opt.hidden_dim*2, eps=1e-12)

        # self.layer_norm2 = torch.nn.LayerNorm(opt.hidden_dim*3, eps=1e-12)
        "Biff layer"
        self.conv_to_gat = nn.ModuleList([nn.Linear(opt.hidden_dim,opt.hidden_dim,bias=True) for i in range(self.biff_layer_number)])
        # self.conv_to_gat = nn.Linear(opt.hidden_dim,opt.hidden_dim,bias=False)
        self.gat_to_conv = nn.ModuleList([nn.Linear(opt.hidden_dim,opt.hidden_dim,bias=True) for i in range(self.biff_layer_number)])
        # self.con_pool = nn.AvgPool2d(opt.hidden_dim,())



    # def speech_weight(self, x, aspect_double_idx, text_len, aspect_len, seq_len,speech_list):
    #     batch_size = x.shape[0]
    #     print(seq_len)
    #     tol_len = x.shape[1]  # sl+cl
    #     aspect_double_idx = aspect_double_idx.cpu().numpy()
    #     text_len = text_len.cpu().numpy()
    #     aspect_len = aspect_len.cpu().numpy()
    #     weight = [[] for i in range(batch_size)]
    #     for i in range(batch_size):
    #         # weight for text
    #         context_len = text_len[i] - aspect_len[i]
    #         for j in range(aspect_double_idx[i, 0]):
    #             weight[i].append((speech_list[j]))
    #         for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
    #             weight[i].append(0)
    #         for j in range(aspect_double_idx[i, 1] + 1, text_len[i]):
    #             weight[i].append((speech_list[j]))
    #         for j in range(text_len[i], seq_len):
    #             weight[i].append(0)
    #         # # weight for concept_mod
    #         # for j in range(seq_len, seq_len + concept_mod_len[i]):
    #         #     weight[i].append(1)
    #         # for j in range(seq_len + concept_mod_len[i], tol_len):
    #         #     weight[i].append(0)
    #     weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
    #     return weight * x  # 根据词性获得不同权重

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len, seq_len,adj):
        all_weights = []
        # for ii in range(len(x)):
        batch_size = x.shape[0]
        tol_len = x.shape[1]  # sl+cl
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        # concept_mod_len = concept_mod_len.cpu().numpy()

        weight = [[] for i in range(batch_size)]

        for i in range(batch_size):
            # weight for text
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append(
                    1 - (aspect_double_idx[i, 0] - j) / context_len)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i, 1] + 1, text_len[i]):
                weight[i].append(
                    1 - (j - aspect_double_idx[i, 1]) / context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        # print((weight*x).shape)
        # print((x).shape)
        # print((weight).shape)
        # all_weights.append(weight * x[ii])

        return weight * x  # 根据上下文的位置关系获得不同权重的

    def syntactic_distance_position_weight(self, x, aspect_double_idx, text_len, aspect_len, seq_len, obj):
        """
        Calculate syntactic relation distance
        :param x:
        :param aspect_double_idx:
        :param text_len:
        :param aspect_len:
        :param seq_len:
        :param obj:
        :return:hidden
        根据语法关系距离获得权重

       """

        def aspect_short_path(G, target, context_len):
            """"
            """
            d = nx.shortest_path_length(G, target=target)
            distance_list = []
            for node in G.nodes():
                try:
                    distance_list.append(min(d[node],context_len))
                except KeyError:
                    distance_list.append(context_len)
            return distance_list

        """
        Context dynamic mask (CDM)
        :return seq_len*1
        """

        def Context_mask_distance(SRD_list, context_len):
            SRD_distacne = [1.0] * len(SRD_list)
            for i in range(len(SRD_list)):
                if SRD_list[i] > self.SRD:
                    SRD_distacne[i] = 0.0
            return SRD_distacne

        """
        Context dynamic weighting(CDW)
        """

        def Context_dynamic_weighting(SRD_list, context_len):
            SRD_distacne = [0.0] * len(SRD_list)
            for i in range(len(SRD_list)):
                # if SRD_list[i] > self.SRD:
                    SRD_distacne[i] = (1.0 - ((SRD_list[i]) / context_len))
            return SRD_distacne

        batch_size = x.shape[0]
        tol_len = x.shape[1]  # sl+cl
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        obj = obj.cpu().numpy()

        for i in range(batch_size):
            syntactic_matrix = obj[i]
            aspect_begin_idx = aspect_double_idx[i,0]
            aspect_end_idx = aspect_double_idx[i,1]

            G = nx.from_numpy_matrix(syntactic_matrix)  # 邻接矩阵转换为图

            if self.opt.context_length_type == "context":
                context_len = text_len[i]
            else:
                context_len = self.opt.max_syntactic_distance
            distance_aspect_begin = np.array(aspect_short_path(G, aspect_begin_idx, context_len))  # 其他节点相对began节点的距离
            distance_aspect_end = np.array(aspect_short_path(G, aspect_end_idx, context_len))
            distance_aspect = np.array((distance_aspect_begin + distance_aspect_end) / 2)
            if self.lcf == "cdm":
                syntactic_positon_weight = Context_mask_distance(distance_aspect, context_len)
            if self.lcf == "cdw":
                syntactic_positon_weight = Context_dynamic_weighting(distance_aspect, context_len)
            weight[i] = syntactic_positon_weight
            weight[i][aspect_begin_idx ] = 0
            weight[i][aspect_end_idx] =0


        weight = torch.tensor(weight, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return weight * x

    def    mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i, 1] + 1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask * x  # 只保留aspect word 的dependency

    def kmax_pooling(self, x, dim, k):
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

    def get_state(self, bsz):
        if True:
            return Variable(torch.rand(bsz, self.hid_dim)).cuda()
        else:
            return Variable(torch.zeros(bsz, self.hid_dim))

    def forward(self, inputs):
        [text_indices,aspect_indices,left_indices,adj,readj,unadj,dependency_type_matrix,dependency_type_matrix_re,dependency_type_matrix_undir,speech_list],flage = inputs

        batch_size = text_indices.shape[0]
        seq_len = text_indices.shape[1]
        text_len = torch.sum(text_indices != 0, dim=-1)

        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat(
            [left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)  # 获得了 aspect的 两个左右下标

        text = self.embed(text_indices)

        #转换为二维 flatten
        # dependency_type_matrix = dependency_type_matrix.reshape(batch_size,-1).long()
        # edge = self.edge_embed(dependency_type_matrix)

        # print(edge)

        # text = self.batch_nor(text)
        # text = self.text_embed_dropout(text)
        # edge = self.edge_embed_dropout(edge)

        text_out, _ = self.text_lstm(text, text_len)


        batch_size = text_out.shape[0]
        seq_len = text_out.shape[1]
        hidden_size = text_out.shape[2] // 2
        x = text_out

        # Z = self.sig(self.W_z(h_prime)+self.U_z(x))
        # R = self.sig(self.W_r(h_prime)+self.U_r(x))
        # H_v = self.W(h_prime)+torch.mul(R,self.U(x))
        # h_prime = (1-Z)*h_prime+Z*self.layer_norm(H_v)
        x_graph = []
        # for i  in self.opt.Syn_Layer:
        for index,(adj_type,edge_matrix) in enumerate([(unadj,dependency_type_matrix_undir)]):
            x_position_out = self.syntactic_distance_position_weight(x, aspect_double_idx, text_len, aspect_len,
                                                                     seq_len, adj_type)


            x1,e_final = self.gat_layer1[index](x_position_out ,adj_type,edge_matrix,aspect_double_idx)
            # x1 = self.syntactic_distance_position_weight(x1, aspect_double_idx, text_len, aspect_len,
            #                                                          seq_len, adj_type)
            x1 = self.gat_gru(x_position_out,x1)

            x2,e_final = self.gat_layer2[index](x1,adj_type,edge_matrix,aspect_double_idx)
            # x2 = self.syntactic_distance_position_weight(x2, aspect_double_idx, text_len, aspect_len,
            #                                                          seq_len, adj_type)
            x2 = self.gat_gru(x2, x1)
            x3,e_final = self.gat_layer3[index](x2,adj_type,edge_matrix,aspect_double_idx)
            # # x3 = self.syntactic_distance_position_weight(x3, aspect_double_idx, text_len, aspect_len,
            # #                                              seq_len, adj_type)
            # # e_final = unpack_and_pack(e_final,seq_lenth=seq_len,seq_list=text_len)
            x3 = self.gat_gru(x3, x2)
            # x4, e_final = self.gat_layer4[index](x3, e_final, edge_matrix, aspect_double_idx)
            # x4 =self.gat_gru(x4,x3)
            # x_graph = x3
        x_graph = x3
        # x_graph= torch.mean(x_graph,dim=0)



        """  
         Biff layer  
        """

        #
        # x_biff_conv = torch.clone(x_conv)
        # x_biff_garph = torch.clone(x_graph)
        # # conv_to_gat
        # # x_biff_conv_old = x_biff_conv
        # # x_biff_garph_old = x_biff_garph
        # for i in range(self.biff_layer_number):
        #
        #     conv_to_gat_weigth = torch.relu(torch.softmax(torch.relu(torch.matmul(self.conv_to_gat[i](x_biff_conv),
        #                                                     x_biff_garph.transpose(1,2))),dim=2))
        #     # print(conv_to_gat_weigth.shape)
        #     gat_to_conv_weigth = torch.relu(torch.softmax(torch.relu(torch.matmul(self.gat_to_conv[i](x_biff_garph,),
        #                                                                           x_biff_conv.transpose(1,2))),dim=2))
        #
        #     x_biff_conv_old = torch.clone(x_biff_conv)
        #     x_biff_garph_old = torch.clone(x_biff_garph)
        #     x_biff_conv = torch.matmul(conv_to_gat_weigth,x_biff_garph_old)
        #     x_biff_garph = torch.matmul(gat_to_conv_weigth,x_biff_conv_old)
        #     x_biff_conv = self.conv_norm(x_biff_conv+x_biff_conv_old)
        #     x_biff_garph = self.gat_norm(x_biff_garph+x_biff_garph_old)
        #     # x_biff_conv = self.conv_liner[i](x_biff_conv)
        #     # x_biff_conv = torch.relu(x_biff_conv)
        #     # x_biff_garph = self.gat_liner[i](x_biff_garph)
        #     # x_biff_garph =torch.relu(x_biff_garph)
        # x_graph = self.gat_norm(x_biff_garph + x_graph)
        # x_conv = self.conv_norm(x_biff_conv + x_conv)
        #
        #



        # print(graph_mask.shape)
        graph_mask = self.mask(x_graph, aspect_double_idx)
        graph_max_pool = self.gat_max_pool(graph_mask).squeeze()
        #no mask
        before_aspect = graph_max_pool
        # text_out_s__mask =  self.syntactic_distance_position_weight(text_out, aspect_double_idx, text_len, aspect_len,
        #                                                              seq_len, adj_type)
        for i in range(self.hop):
            # alpha_mat = torch.matmul(graph_mask, text_out.transpose(1, 2))
            #
            # if i == self.hop - 1:
            #
            #     alpha = torch.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
            #     a1 = torch.matmul(alpha, text_out).squeeze(1)  # batch_size x hidden_dim
            # else:
            #     alpha_text = torch.softmax(alpha_mat, dim=2)
            #     a1 = torch.matmul(alpha_text, text_out).squeeze(1)
            #     graph_mask = self.lamb * self.layer_norm2(torch.sigmoid(a1)) + graph_mask
            aspect_embdding = before_aspect.unsqueeze(1).repeat([1, seq_len, 1])
            concat_final = torch.cat((text_out,aspect_embdding),dim=2)

            alpha = self.concat_liner[i](concat_final)
            alpha = torch.sigmoid(alpha).transpose(1,2)
            alpha_text = torch.softmax(alpha, dim=1)
            a1 = torch.matmul(alpha_text, text_out).squeeze(1)
            before_aspect = before_aspect + self.layer_norm1(a1)

                # calculate hidden state attention
        # text_speet_state = self.syntactic_distance_position_weight(text_out, aspect_double_idx, text_len, aspect_len, seq_len,adj)
        # text_out_mask = self.mask(text_speet_state , aspect_double_idx)
        #no position
        # text_out_mask = text_out
        text_out_mask =self.mask(text_out, aspect_double_idx)

        # no mask

        # #对照试验
        hop=1
        for i in range(hop):
            alpha_mat_text = torch.matmul(text_out_mask, text_out.transpose(1, 2))
            if i == hop - 1:
                alpha_text = torch.softmax(alpha_mat_text.sum(1, keepdim=True), dim=2)
                a3 = torch.matmul(alpha_text, text_out).squeeze(1)
            else:
                # alpha_text = torch.softmax(alpha_mat_text, dim=2)
                alpha_text = torch.softmax(alpha_mat_text, dim=2)
                a3 = torch.matmul(alpha_text, text_out).squeeze(1)

        #         text_out_mask = lambdaa * self.layer_norm2(torch.sigmoid(a3)) + text_out_mask
        # # # text_liner_list = [self.text_liner1, self.text_liner2, self.text_liner3,self.text_liner4,self.text_liner5]
        # x_position_out = self.mask(x_position_out,aspect_double_idx)
        # # x_position_out_liner = self.text_out_liner(x_position_out)
        # # x_position_out_liner =  self.text_liner_drop(x_position_out_liner)
        # for i in range(hop):
        #     alpha_mat_text = torch.matmul(x_position_out, text_out.transpose(1, 2))
        #     if i == hop - 1:
        #         alpha_text = torch.softmax(alpha_mat_text.sum(1, keepdim=True), dim=2)
        #         a2 = torch.matmul(alpha_text, text_out).squeeze(1)
            # else:
                # text_liner = text_liner_list[i]
                # alpha_text = torch.softmax(alpha_mat_text, dim=2)
                # alpha_text = torch.softmax(alpha_mat_text,dim=2)
                # a2 = torch.matmul(alpha_text, text_out).squeeze(1)
                # text_out_mask = lambdaa * self.layer_norm2(torch.sigmoid(a2)) + text_out_mask
                # text_out_mask =  self.layer_norm2(torch.sigmoid(text_liner(a2))) + text_out_mask


        #  CNN attention
        # no mask



        # x_conv = self.mask(x_conv, aspect_double_idx)
        # x_conv_max_pool = self.conv_max_pool(x_conv).squeeze()
        #
        # for i in range(hop):
        #     alpha_mat_x_conv = torch.matmul(x_conv, text_out.transpose(1, 2))
        #     if i == hop - 1:
        #         alpha_x_conv = torch.softmax(alpha_mat_x_conv.sum(1, keepdim=True), dim=2)
        #         # if flage == False:  # False为测试模式
        #         #     time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        #         #     conv_attention = alpha_x_conv.cpu().numpy()
        #         #     np.save("./attention_numpy/conv_attention_{0}.npy".format(time_str), conv_attention)
        #         a3 = torch.matmul(alpha_x_conv, x_conv).squeeze(1)  # batch_size x hidden_dim

                # out_conv = torch.cat([conv_averge_pooling,a3],1)

            # else:
            #     # conv_liner = conv_liner_list[i]
            #     alpha_x_conv = torch.softmax(alpha_mat_x_conv, dim=2)
            #     a3 = torch.matmul(alpha_x_conv, text_out).squeeze(1)  #
            #     # x_conv = lambdaa * self.layer_norm3(torch.sigmoid(a3)) + x_conv

        # fnout = torch.cat((a1,graph_max_pool,a3,x_conv_max_pool), 1)
        fnout = torch.cat((before_aspect,graph_max_pool,a3), -1)

        if self.opt.use_lstm_attention:
            # output = self.fc(fnout)
            fnout = self.liner_dropout(fnout)
            output = self.fc3(fnout)
        else:
            output = self.fc(fnout)
        # print(fnout)
        return output

