import torch
import torch.nn as nn
import torch.nn.functional as F
# import  torch.functional as F
import math
import  numpy as np
from layers.dynamic_rnn import DynamicLSTM
from layers.capsulelayer import Caps_Layer
from torch.autograd import Variable



# class GraphAttentionLayer(nn.Module):
#     def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat
#
#         self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#         self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)
#
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#
#
#     def forward(self, input, adj):
#         h = torch.matmul(input, self.W)
#         batch_size = h.size()[0]
#         token_lenth = h.size()[1]
#
#         # h.repeat_interleave(repeats=token_lenth, dim=2).view(batch_size, token_lenth * token_lenth, -1)    [bacth_szie,token_len*token_len,embedding_dem]
#
#         a_input = torch.cat([h.repeat_interleave(repeats=token_lenth,dim=2).view(batch_size,token_lenth * token_lenth, -1), h.repeat_interleave(token_lenth, dim=1)], dim=2).view(batch_size,token_lenth,-1, 2 * self.out_features)  # 这里让每两个节点的向量都连接在一起遍历一次得到 bacth* N * N * (2 * out_features)大小的矩阵
#
#
#         e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
#         zero_vec = -9e15 * torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)
#         attention = torch.softmax(attention, dim=2)             # 这里是一个非线性变换，将有权重的变得更趋近于1，没权重的为0
#
#         attention = F.dropout(attention, self.dropout, training=self.training)
#         h_prime = torch.matmul(attention, h)
#
#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GCapNet(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(GCapNet, self).__init__()
        self.opt = opt
        if opt.use_bert ==False:
            self.embed = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float))
        self.hid_dim = opt.hidden_dim
        # self.text_lstm = nn.LSTM(opt.embed_dim, opt.hidden_dim,
        #                          num_layers=1, bidirectional=True, batch_first=True)

        self.text_lstm = DynamicLSTM(
            opt.embed_dim, opt.hidden_dim, num_layers=2, batch_first=True, bidirectional=True,dropout=0.2,rnn_type="LSTM")
        # self.text_lstmasp = DynamicLSTM(
        #     opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gat1 = GraphAttentionLayer(opt.hidden_dim, opt.hidden_dim)
        self.gat2 = GraphAttentionLayer(opt.hidden_dim, opt.hidden_dim)
        # self.gc22 = GraphConvolution(opt.hidden_dim, opt.hidden_dim)
        self.gat3 = GraphAttentionLayer(opt.hidden_dim, opt.hidden_dim)
        self.gat4 = GraphAttentionLayer(opt.hidden_dim, opt.hidden_dim)
        self.gat5 = GraphAttentionLayer(opt.hidden_dim, opt.hidden_dim)
        self.gat6 = GraphAttentionLayer(opt.hidden_dim, opt.hidden_dim)

        self.conv1 = nn.Conv1d(opt.hidden_dim, opt.hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(opt.hidden_dim, opt.hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv1d(opt.hidden_dim, opt.hidden_dim, 3, padding=1)
        self.conv4 = nn.Conv1d(opt.hidden_dim, opt.hidden_dim, 3, padding=1)
        self.conv5 = nn.Conv1d(opt.hidden_dim, opt.hidden_dim, 3, padding=1)
        # self.poool = nn.MaxPool1d(9)
        self.capsule = Caps_Layer(300,num_capsule=10,dim_capsule=60)
        self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)
        self.fc1 = nn.Linear(opt.hidden_dim*2, opt.polarities_dim)
        # self.fc2 = nn.Linear(opt.hidden_dim*2, opt.polarities_dim)
        self.fc2 = nn.Linear(opt.hidden_dim*3, opt.polarities_dim)
        # self.fc4 = nn.Linear(opt.hidden_dim*4, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.4)

        # self.batch_nor = nn.BatchNorm1d(90)

        self.layer_norm1 = torch.nn.LayerNorm(opt.hidden_dim, eps=1e-12)
        self.layer_norm2 = torch.nn.LayerNorm(opt.hidden_dim, eps=1e-12)
        self.layer_norm3 = torch.nn.LayerNorm(opt.hidden_dim, eps=1e-12)
        # self.layer_norm1 = torch.nn.LayerNorm(opt.hidden_dim*2, eps=1e-12)

        # self.layer_norm2 = torch.nn.LayerNorm(opt.hidden_dim*3, eps=1e-12)

    def speech_weight(self, x, aspect_double_idx, text_len, aspect_len, seq_len,speech_list):
        batch_size = x.shape[0]
        tol_len = x.shape[1]  # sl+cl
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            # weight for text
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append((speech_list[j]))
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i, 1] + 1, text_len[i]):
                weight[i].append((speech_list[j]))
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
            # # weight for concept_mod
            # for j in range(seq_len, seq_len + concept_mod_len[i]):
            #     weight[i].append(1)
            # for j in range(seq_len + concept_mod_len[i], tol_len):
            #     weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        return weight * x  # 根据词性获得不同权重


    def position_weight(self, x, aspect_double_idx, text_len, aspect_len, seq_len):
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


    def mask(self, x, aspect_double_idx):
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
    def aspect_word(self,x, aspect_double_idx):
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
        # traget_word = np.sum(mask * x,axis=1)/(aspect_double_idx[i, 1]-aspect_double_idx[i, 0])

        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        target_lenth = torch.tensor(np.array(aspect_double_idx[:, 1] - aspect_double_idx[:, 0]).reshape(batch_size,-1)).float().to(self.opt.device)
        traget_word = torch.sum(mask * x,dim=1,keepdim=True)/target_lenth
        return  traget_word



    def kmax_pooling(self, x, dim, k):
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        if True:
            return Variable(torch.rand(bsz, self.hid_dim)).cuda()
        else:
            return Variable(torch.zeros(bsz, self.hid_dim))

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj,speech_list = inputs

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
        text_out, _ = self.text_lstm(text, text_len)
        # text_outasp, _ = self.text_lstm(textasp,aspect_len)

        # #################### SHA-RNN ########################
        # #layer_normal
        # query = text_out
        # Key = self.layer_norm(query)
        # value = self.layer_norm1(query)
        # attention_scores = torch.matmul(query, Key.transpose(-1, -2).contiguous()) \
        #                    / math.sqrt(Key.shape[-1])
        # attention_weights = F.softmax(attention_scores, dim=-1)
        # mix = torch.matmul(attention_weights, value)
        # text_out = mix+text_out
        # #################### SHA-RNN ########################

        batch_size = text_out.shape[0]
        seq_len = text_out.shape[1]
        hidden_size = text_out.shape[2] // 2
        text_out = text_out.reshape(batch_size, seq_len, hidden_size, -1).mean(dim=-1)
        # concept_mod = self.embed(concept_mod)
        # x = torch.cat([text_out, concept_mod], dim=1)
        x = text_out
        okk = x
        #
        #
        # x_speech =  self.speech_weight(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len, seq_len),aspect_double_idx, text_len, aspect_len, seq_len,speech_list)
        # x_conv_1 = torch.relu(self.conv1(x_conv_speech.transpose(1, 2)))
        # x_conv_2 = torch.relu(self.conv2(x_conv_speech.transpose(1, 2)))
        # x_conv_3 = torch.relu(self.conv3(x_conv_speech.transpose(1, 2)))
        #
        # #
        # x_conv = 1*( x_conv_1 +x_conv_2+x_conv_3)/3



        # x_conv = torch.relu(self.conv2(self.position_weight(x_conv.transpose(1, 2), aspect_double_idx, text_len, aspect_len, seq_len).transpose(1,2)))

        # gat
        x_position_out =self.position_weight(x, aspect_double_idx, text_len, aspect_len, seq_len,)
        x_speech_out = self.speech_weight(x_position_out,aspect_double_idx, text_len, aspect_len, seq_len,speech_list)

        x_conv = F.relu(self.conv1(x_speech_out.transpose(1, 2)))
        x_conv = F.relu(self.conv2(
            self.position_weight(x_conv.transpose(1, 2), aspect_double_idx, text_len, aspect_len, seq_len).transpose(1,
                                                                                                                     2)))

        x = torch.relu(self.gat1(x_speech_out, adj))
        x2 = torch.relu(self.gat2(x, adj))
        x3 = torch.relu(self.gat3(x, adj))
        x4 = torch.relu(self.gat4(x, adj))
        x5 = torch.relu(self.gat5(x_speech_out, adj))
        x6 = torch.relu(self.gat6(x_speech_out, adj))

        gat_part_out =  0.2*(x2 + x3 + x4 + x5+x6)
        # capsule_out = self.capsule(gat_part_out)
        hop = 5
        lambdaa = 0.01
        # lambdaa = 1
        # graph_mask = x
        gat_out_mask = self.mask(gat_part_out, aspect_double_idx)
        # gat_out_mask = gat_part_out
        for i in range(hop):
            alpha_mat = torch.matmul(gat_out_mask,text_out.transpose(1, 2))
            if i == hop - 1:
                alpha = torch.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
                a1 = torch.matmul(alpha, text_out).squeeze(1)  # batch_size x hidden_dim
            else:
                # alpha = torch.softmax(alpha_mat, dim=2)
                alpha = alpha_mat
                a1 = torch.matmul(alpha, text_out).squeeze(1)  # batch_size x hidden_dim

                # graph_mask = lambdaa*torch.sigmoid(a1)+graph_mask
                # gat_out_mask = self.mask(lambdaa * self.layer_norm1(torch.sigmoid(a1)) + gat_out_mask,aspect_double_idx)
                gat_out_mask = lambdaa * self.layer_norm1(torch.sigmoid(a1)) + gat_out_mask

        # calculate hidden state attention
        text_out_mask = self.mask(x_speech_out, aspect_double_idx)
        # text_out_mask = text_out
        for i in range(hop):
            alpha_mat_text = torch.matmul(text_out_mask, text_out.transpose(1, 2))
            if i == hop - 1:
                alpha_text = torch.softmax(alpha_mat_text.sum(1, keepdim=True), dim=2)
                a2 = torch.matmul(alpha_text, text_out).squeeze(1)  # batch_size x hidden_dim
            else:
                alpha_text = torch.softmax(alpha_mat_text, dim=2)

                # alpha_text = alpha_mat_text
                # alpha_text = torch.softmax(alpha_mat_text.sum(1, keepdim=True), dim=2)
                a2 = torch.matmul(alpha_text, text_out).squeeze(1)  # batch_size x hidden_dim
                # text_out_mask = lambdaa*torch.sigmoid(a2)+text_out_mask
                text_out_mask = lambdaa * self.layer_norm2(torch.sigmoid(a2)) + text_out_mask
                # text_out_mask = a2+text_out_mask
        #
        # # calculate CNN attention
        x_conv_mask = self.mask(x_conv.transpose(1, 2), aspect_double_idx)
        lambdaa = 0.01
        for i in range(hop):
            alpha_mat_x_conv = torch.matmul(x_conv_mask, text_out.transpose(1, 2))
            if i == hop - 1:
                alpha_x_conv = torch.softmax(alpha_mat_x_conv.sum(1, keepdim=True), dim=2)
                a3 = torch.matmul(alpha_x_conv, text_out).squeeze(1)  # batch_size x hidden_dim
            else:
                # alpha_x_conv = torch.softmax(alpha_mat_x_conv, dim=2)
                alpha_x_conv = alpha_mat_x_conv
                a3 = torch.matmul(alpha_x_conv, text_out).squeeze(1)  # batch_size x hidden_dim
                # x_conv =lambdaa* torch.sigmoid(a3)+x_conv
                x_conv_mask = lambdaa * self.layer_norm3(torch.sigmoid(a3))+x_conv_mask
                # text_out_mask = a3+text_out_mask

        fnout = torch.cat((a1,a2, a3), 1)

        # # fnout = torch.cat((a1), 1)
        # fnout =a1
        if self.opt.use_lstm_attention:
            # output = self.fc(fnout)
            # fnout = fnout.view(batch_size,-1)
            output = self.fc2(fnout)
        else:
            output = self.fc(fnout)
        return output




