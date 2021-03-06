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
from .layers import GraphDotProductLayer,GAT
import copy

class Biff_Lcf_GAT(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(Biff_Lcf_GAT, self).__init__()
        self.opt = opt
        # self.number = 2
        if opt.use_bert ==False:
            self.embed = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float))
        self.hid_dim = opt.hidden_dim

        self.text_lstm = DynamicLSTM(
            opt.embed_dim, opt.hidden_dim//2, num_layers=1, batch_first=True, bidirectional=True,rnn_type="LSTM")
        # self.text_lstmasp = DynamicLSTM(
        #     opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        self.lcf = opt.lcf
        self.SRD = opt.SRD
        self.biff_layer_number = opt.biff_layer_number
        # self.gat_list=nn.ModuleList([
        #     GraphAttentionLayer(opt,opt.hidden_dim, opt.hidden_dim) for i in range(self.number)]
        # )
        self.conv_list = nn.ModuleList([nn.Conv1d(opt.hidden_dim, opt.hidden_dim, 3, padding=1) for i in range(self.biff_layer_number)])
        self.conv_norm = nn.LayerNorm(opt.hidden_dim)
        self.conv_liner = nn.ModuleList([nn.Linear(opt.hidden_dim, opt.hidden_dim) for i in range(self.biff_layer_number)])

        self.gat_norm = nn.LayerNorm(opt.hidden_dim)
        self.gat_liner = nn.ModuleList([nn.Linear(opt.hidden_dim, opt.hidden_dim) for i in range(self.biff_layer_number)])

        # self.conv2 = nn.Conv1d(opt.hidden_dim, opt.hidden_dim, 3, padding=1)
        self.gat_to_conv_liner = nn.Linear(opt.hidden_dim,opt.hidden_dim,bias=False)
        self.conv_to_gat_liner = nn.Linear(opt.hidden_dim, opt.hidden_dim,bias=False)
        if opt.use_scaled_dot == False:
            GraphLayer = GAT
            print("111111111")

        else:
            GraphLayer = GraphDotProductLayer
        self.gat1 = GraphLayer(opt,opt.hidden_dim, opt.hidden_dim,dropout =opt.graph_dropout,alpha=opt.GAT_alpha,n_heads=int(opt.hidden_dim/100))
        self.gat2 = GraphLayer(opt,opt.hidden_dim, opt.hidden_dim,dropout =opt.graph_dropout,alpha=opt.GAT_alpha,n_heads=int(opt.hidden_dim/100))
        self.gat3 = GraphLayer(opt,opt.hidden_dim, opt.hidden_dim,dropout =opt.graph_dropout,alpha=opt.GAT_alpha,n_heads=int(opt.hidden_dim/100))
        self.gat_list = nn.ModuleList([GraphLayer(opt,opt.hidden_dim, opt.hidden_dim,dropout =opt.graph_dropout,alpha=opt.GAT_alpha,n_heads=int(opt.hidden_dim/100)) for i in range(self.biff_layer_number)])
        self.conv_list = nn.ModuleList([nn.Conv1d(opt.hidden_dim, opt.hidden_dim, 5, padding=2) for i in range(self.biff_layer_number)])
        self.gat_undir_1 = GraphLayer(opt, opt.hidden_dim, opt.hidden_dim, dropout=opt.graph_dropout, alpha=opt.GAT_alpha,
                               n_heads=int(opt.hidden_dim / 100))
        self.gat_undir_2 = GraphLayer(opt, opt.hidden_dim, opt.hidden_dim, dropout=opt.graph_dropout, alpha=opt.GAT_alpha,
                               n_heads=int(opt.hidden_dim / 100))
        self.gat_undir_3  = GraphLayer(opt, opt.hidden_dim, opt.hidden_dim, dropout=opt.graph_dropout, alpha=opt.GAT_alpha,
                               n_heads=int(opt.hidden_dim / 100))

        #
        #
        self.gat_max_pool = torch.nn.AdaptiveAvgPool2d([1,opt.hidden_dim])

        self.conv1 = nn.Conv1d(opt.hidden_dim, opt.hidden_dim, 5, padding=2)
        self.conv_list  =nn.ModuleList([nn.Conv1d(opt.hidden_dim, opt.hidden_dim, 5, padding=2) for i in range(self.biff_layer_number)])
        # self.conv2 = nn.Conv1d(opt.hidden_dim, opt.hidden_dim, 3, padding=1)
        self.conv_max_pool = torch.nn.AdaptiveAvgPool2d([1, opt.hidden_dim])

        self.grahp_out_liner = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.grahp_liner_drop = nn.Dropout(0.4)

        self.text_out_liner = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.text_liner_drop = nn.Dropout(0.4)

        self.conv_out_liner = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.conv_liner_drop = nn.Dropout(0.4)

        # self.poool = nn.MaxPool1d(9)
        # self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)
        # self.fc1 = nn.Linear(opt.hidden_dim*2, opt.polarities_dim)

        # self.fc2 = nn.Linear(opt.hidden_dim * 2, opt.polarities_dim)

        self.fc3 = nn.Linear(opt.hidden_dim*5, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(opt.dropout)
        self.liner_dropout = nn.Dropout(opt.dropout)

        # self.batch_nor = nn.BatchNorm1d(90)

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
        [text_indices, aspect_indices, left_indices, adj,unadj,speech_list],flage = inputs

        text_len = torch.sum(text_indices != 0, dim=-1)
        # concept_mod_len = torch.sum(concept_mod != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat(
            [left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)  # 获得了 aspect的 两个左右下标

        text = self.embed(text_indices)
        # text = self.batch_nor(text)
        text = self.text_embed_dropout(text)


        # textasp = self.embed(aspect_indices)
        # text = self.batch_nor(text)
        # textasp = self.text_embed_dropout(textasp)

        # text_out, _ = self.text_lstm(text)

        # print(text.shape)
        text_out, _ = self.text_lstm(text, text_len)
        # print(text_out.shape)
        # exit()

        # text_outasp, _ = self.text_lstm(textasp,aspect_len)


        batch_size = text_out.shape[0]
        seq_len = text_out.shape[1]
        hidden_size = text_out.shape[2] // 2
        hidden_size = text_out.shape[2]
        # print(text_out.shape)
        # text_out = text_out.reshape(batch_size, seq_len, hidden_size, -1).mean(dim=-1)
        # print(text_out.shape)
        # concept_mod = self.embed(concept_mod)
        # x = torch.cat([text_out, concept_mod], dim=1)
        x = text_out

        # x_position_out_1 =self.position_weight(x, aspect_double_idx, text_len, aspect_len, seq_len,adj)
        # x_position_out_2 =self.syntactic_distance_position_weight(x, aspect_double_idx, text_len, aspect_len, seq_len,adj)
        x_position_out =self.syntactic_distance_position_weight(x, aspect_double_idx, text_len, aspect_len, seq_len,adj)
        # x_position_out =self.syntactic_alph*x_position_out_2+self.position_alph*x_position_out_1
        #x_position_out = self.position_weight(x, aspect_double_idx, text_len, aspect_len, seq_len,adj)
        # x_conv = torch.relu(self.conv1(x.transpose(1, 2)).transpose(1,2),)
        # x_conv = torch.relu(self.conv2(x_conv.transpose(1,2)).transpose(1,2))
        #
        #
        # # no position
        # # x_position_out = x
        #
        # # x_speech_out = self.speech_weight(x_position_out,aspect_double_idx, text_len, aspect_len, seq_len,speech_list)
        # x_graph = torch.relu(self.gat1(x_position_out , adj,aspect_double_idx))
        #
        # #no position
        # x1 = self.position_weight(x1,aspect_double_idx, text_len, aspect_len, seq_len)
        #
        #x2 =self.syntactic_distance_position_weight(x1, aspect_double_idx, text_len, aspect_len, seq_len,adj)
        # x2 =self.syntactic_distance_position_weight(x2, aspect_double_idx, text_len, aspect_len, seq_len,adj)
        # x2 = torch.relu(self.gat2(x1, adj,aspect_double_idx ))
        # x_graph = torch.relu(self.gat3(x2, adj,aspect_double_idx ))
        # x_graph_undir = torch.relu(self.gat1(x_position_out , adj,aspect_double_idx))
        # x4 = torch.relu(self.gat3(x2, adj))
        # x_graph = x3
        # x_graph = x3
        # for gat_i in self.gat_list:
        #     x_graph+=torch.relu(gat_i(x1,adj))
        # x_graph =  x_graph/len(self.gat_list)



        """  
         Biff layer  
        """


        # x_biff_conv = torch.clone(x_conv)
        # x_biff_garph = torch.clone(x_graph)
        # conv_to_gat
        # x_biff_conv_old = x_biff_conv
        # x_biff_garph_old = x_biff_garph
        x_biff_garph = torch.relu(self.gat1(x_position_out, adj, aspect_double_idx))
        x_biff_conv= torch.relu(self.conv1(x.transpose(1, 2)).transpose(1, 2))
        for i in range(self.biff_layer_number):

            conv_to_gat_weigth = torch.relu(torch.softmax(torch.relu(torch.matmul(self.conv_to_gat[i](x_biff_conv),
                                                            x_biff_garph.transpose(1,2))),dim=2))
            # print(conv_to_gat_weigth.shape)
            gat_to_conv_weigth = torch.relu(torch.softmax(torch.relu(torch.matmul(self.gat_to_conv[i](x_biff_garph,),
                                                                                  x_biff_conv.transpose(1,2))),dim=2))

            x_biff_conv_old = torch.clone(x_biff_conv)
            x_biff_garph_old = torch.clone(x_biff_garph)
            x_biff_conv = torch.matmul(conv_to_gat_weigth,x_biff_garph_old)
            x_biff_garph = torch.matmul(gat_to_conv_weigth,x_biff_conv_old)
            x_biff_conv = self.conv_norm(x_biff_conv+x_biff_conv_old)
            x_biff_garph = self.gat_norm(x_biff_garph+x_biff_garph_old)
            x_biff_conv = torch.relu(self.conv_list[i](x_biff_conv.transpose(1, 2)).transpose(1, 2))
            x_biff_garph = torch.relu(self.gat_list[i](x_biff_garph, adj, aspect_double_idx))



            # x_biff_conv = self.conv_liner[i](x_biff_conv)
            # x_biff_conv = torch.relu(x_biff_conv)
            # x_biff_garph = self.gat_liner[i](x_biff_garph)
            # x_biff_garph =torch.relu(x_biff_garph)
        x_graph = self.conv_norm(x_biff_conv + x_biff_conv_old)
        x_conv = self.gat_norm(x_biff_garph + x_biff_garph_old)
        # x_confuse = self.gat_norm(x_biff_conv+x_biff_garph)
        # x_graph = self.gat_norm(x_biff_garph + x_graph)
        # x_conv = self.conv_norm(x_biff_conv+x_conv)
        # x_confu =self.gat_norm(x_biff_conv+x_biff_garph)
        # x_confu = self.mask(x_confu)


        # x_1 =torch.relu(self.gat1(x_biff_garph,adj))
        # x_2 = torch.relu(self.gat2(x_biff_garph, adj))
        # x_3 = torch.relu(self.gat3(x_biff_garph, adj))
        # x_4 = torch.relu(self.gat4(x_biff_garph, adj))
        # x_5 = torch.relu(self.gat5(x_biff_garph, adj))




        # print(graph_mask.shape)
        graph_mask = self.mask(x_graph, aspect_double_idx)
        graph_max_pool = self.gat_max_pool(graph_mask).squeeze()
        #no mask

        hop = 1
        # hop = 3
        # gat_liner_list = [self.gat_liner1, self.gat_liner2, self.gat_liner3,self.gat_liner3,self.gat_liner4,self.gat_liner5]
        # gat_max_pooling = F.max_pool2d(graph_mask,(hidden_size,1)).squeeze()

        # graph_mask_liner = self.grahp_out_liner(graph_mask)
        # graph_mask_liner = self.grahp_liner_drop(graph_mask_liner)

        for i in range(hop):
            alpha_mat = torch.matmul(graph_mask, text_out.transpose(1, 2))
            if i == hop - 1:

                alpha = torch.softmax(alpha_mat.sum(1, keepdim=True), dim=2)

                a1 = torch.matmul(alpha, text_out).squeeze(1)  # batch_size x hidden_dim
            else:

                alpha_text = torch.softmax(alpha_mat, dim=2)
                a1 = torch.matmul(alpha_text, text_out).squeeze(1)
                # graph_mask = self.layer_norm1(torch.sigmoid(gat_liner(a1))) + graph_mask
                # out_gat = torch.cat([gat_max_pooling,a1],1)
                # out_gat =a1

                # calculate hidden state attention
        # text_speet_state = self.syntactic_distance_position_weight(text_out, aspect_double_idx, text_len, aspect_len, seq_len,adj)
        # text_out_mask = self.mask(text_speet_state , aspect_double_idx)
        #no position
        # text_out_mask = text_out
        # text_out_mask =self.mask(text_out, aspect_double_idx)

        # no mask

        # #对照试验
        # for i in range(hop):
        #     alpha_mat_text = torch.matmul(text_out_mask, text_out.transpose(1, 2))
        #     if i == hop - 1:
        #         alpha_text = torch.softmax(alpha_mat_text.sum(1, keepdim=True), dim=2)
        #         a3 = torch.matmul(alpha_text, text_out).squeeze(1)
        #     else:
        #         # alpha_text = torch.softmax(alpha_mat_text, dim=2)
        #         alpha_text = torch.softmax(alpha_mat_text, dim=2)
        #         a3 = torch.matmul(alpha_text, text_out).squeeze(1)
        #         text_out_mask = lambdaa * self.layer_norm2(torch.sigmoid(a3)) + text_out_mask
        # # # text_liner_list = [self.text_liner1, self.text_liner2, self.text_liner3,self.text_liner4,self.text_liner5]
        x_position_out = self.mask(x_position_out,aspect_double_idx)
        # x_position_out_liner = self.text_out_liner(x_position_out)
        # x_position_out_liner =  self.text_liner_drop(x_position_out_liner)
        for i in range(hop):
            alpha_mat_text = torch.matmul(x_position_out, text_out.transpose(1, 2))
            if i == hop - 1:
                alpha_text = torch.softmax(alpha_mat_text.sum(1, keepdim=True), dim=2)
                a2 = torch.matmul(alpha_text, text_out).squeeze(1)
            # else:
                # text_liner = text_liner_list[i]
                # alpha_text = torch.softmax(alpha_mat_text, dim=2)
                # alpha_text = torch.softmax(alpha_mat_text,dim=2)
                # a2 = torch.matmul(alpha_text, text_out).squeeze(1)
                # text_out_mask = lambdaa * self.layer_norm2(torch.sigmoid(a2)) + text_out_mask
                # text_out_mask =  self.layer_norm2(torch.sigmoid(text_liner(a2))) + text_out_mask


        #  CNN attention
        # no mask
        # conv_liner_list = [self.conv_liner1, self.conv_liner2, self.conv_liner3,self.conv_liner4,self.conv_liner5]
        # print(x_biff_conv.shape)
        # x_conv = self.mask(x_biff_conv, aspect_double_idx)
        # print(x_conv.shape)

        x_conv = self.mask(x_conv, aspect_double_idx)
        x_conv_max_pool = self.conv_max_pool(x_conv).squeeze()
        # x_conv_liner = self.conv_out_liner(x_conv)
        # x_conv_liner = self.conv_liner_drop(x_conv_liner)
        # conv_averge_pooling = F.avg_pool2d(x_conv,(hidden_size,1)).squeeze()
        for i in range(hop):
            alpha_mat_x_conv = torch.matmul(x_conv, text_out.transpose(1, 2))
            if i == hop - 1:
                alpha_x_conv = torch.softmax(alpha_mat_x_conv.sum(1, keepdim=True), dim=2)
                # if flage == False:  # False为测试模式
                #     time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                #     conv_attention = alpha_x_conv.cpu().numpy()
                #     np.save("./attention_numpy/conv_attention_{0}.npy".format(time_str), conv_attention)
                a3 = torch.matmul(alpha_x_conv, x_conv).squeeze(1)  # batch_size x hidden_dim

                # out_conv = torch.cat([conv_averge_pooling,a3],1)

            # else:
            #     # conv_liner = conv_liner_list[i]
            #     alpha_x_conv = torch.softmax(alpha_mat_x_conv, dim=2)
            #     a3 = torch.matmul(alpha_x_conv, text_out).squeeze(1)  #
            #     # x_conv = lambdaa * self.layer_norm3(torch.sigmoid(a3)) + x_conv

        fnout = torch.cat((a1,graph_max_pool,a2,a3,x_conv_max_pool), 1)
        if self.opt.use_lstm_attention:
            # output = self.fc(fnout)
            fnout = self.liner_dropout(fnout)
            output = self.fc3(fnout)
        else:
            output = self.fc(fnout)
        return output

