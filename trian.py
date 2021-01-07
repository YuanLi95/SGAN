# -*- coding: utf-8 -*-
import pickle
import os
import math
import argparse
import random
import numpy
import torch
import torch.nn as nn
from bucket_iterator import BucketIterator
from sklearn import metrics
from data_utils import ABSADatesetReader
from  models.pos_gat import PoS_GAT
from  models.biff_lcf_gat import Biff_Lcf_GAT
from models.Type_aware_GAT import  Type_aware_GAT
from models.Relation_aware_Position import  Relation_aware_Pos
from models.syntax_transformer_model import Syntax_local_transformer
import torch.nn.functional as F
from torchsummary import summary
import  numpy as np
import codecs
import os
from thop import profile
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("log")


import  time
class Instructor:
    def __init__(self, opt):
        self.opt = opt

        absa_dataset = ABSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim,dependency_dim=opt.dependency_edge_dim,position_dim=opt.position_dim, use_bert=opt.use_bert, max_len=70,opt=opt)

        self.train_data_loader = BucketIterator(data=absa_dataset.train_data, batch_size=opt.batch_size, shuffle=True,sort=False,
                                                max_len=70,)
        self.test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=opt.batch_size, shuffle=False,sort=False,
                                               max_len=70)

        self.model = opt.model_class(absa_dataset.embedding_matrix,absa_dataset.dependency_matrix,absa_dataset.position_matrix, opt).to(opt.device)
        # self.model = opt.model_class(absa_dataset.embedding_matrix, opt)
        # self.model = nn.DataParallel(self.model).to(opt.device)
        self._print_args()
        self.global_f1 = 0.

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    def _print_args(self):
        # print(self.model.gat_layer_list)
        # print(self.model.ModuleList)
        # exit()
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def label_smoothing(self, inputs, epsilon=0.1):
        '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
        inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
        epsilon: Smoothing rate.

        For example,

        ```
        import tensorflow as tf
        inputs = tf.convert_to_tensor([[[0, 0, 1],
           [0, 1, 0],
           [1, 0, 0]],
          [[1, 0, 0],
           [1, 0, 0],
           [0, 1, 0]]], tf.float32)

        outputs = label_smoothing(inputs)

        with tf.Session() as sess:
            print(sess.run([outputs]))

        >>
        [array([[[ 0.03333334,  0.03333334,  0.93333334],
            [ 0.03333334,  0.93333334,  0.03333334],
            [ 0.93333334,  0.03333334,  0.03333334]],
           [[ 0.93333334,  0.03333334,  0.03333334],
            [ 0.93333334,  0.03333334,  0.03333334],
            [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
        ```
        '''
        V = inputs.get_shape().as_list()[-1]  # number of channels
        return ((1 - epsilon) * inputs) + (epsilon / V)

    def _train(self, criterion, optimizer):
        max_test_acc = 0
        max_test_f1 = 0
        global_step = 0
        continue_not_increase = 0
        # self.model = torch.load("state_dict/12_6/Relation_aware_Pos_lap14_0.7435956949939592.pkl")
        # test_acc, test_f1, test_loss, t_targets_all, t_outputs_all= self._evaluate_acc_f1(criterion)
        # exit()


        for epoch in range(self.opt.num_epoch):
            increase_flag = False
            print('>' * 100)
            print('epoch: ', epoch)

            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_data_loader):

                global_step += 1

                # switch model to training mode, clear gradient accumulators'
                self.model.train()
                # print(self.model)
                optimizer.zero_grad()
                # print(sample_batched)
                # print(self.opt.inputs_cols)
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                #

                # a_numpy = np.array(inputs)
                #
                # #
                # #
                # np.save('./state_dict/twitter_numpy.npy', a_numpy)
                inputs = [inputs,True]


                # print()

                # outputs = torch.nn.DataParallel(self.model(inputs),device_ids=[1,0])
                outputs,alpha_attention = self.model(inputs)
                # flop, para = profile(self.model, input_size=(inputs), )
                # print("%.2fM" % (flop / 1e6), "%.2fM" % (para / 1e6))

                # print(outputs)?
                ## ce loss#########################################
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                # predict the best model
                # self.model.load_state_dict(torch.load('state_dict/' + self.opt.model_name + '_' + self.opt.dataset + '.pkl'))

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    test_acc, test_f1, test_loss,t_targets_all, t_outputs_all,alpha_attention = self._evaluate_acc_f1(criterion)
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                    if test_f1 > max_test_f1:
                        increase_flag = True
                        max_test_f1 = test_f1

                        if self.opt.save and test_f1 > self.global_f1:
                            self.global_f1 = test_f1
                            # np.save("./attention_numpy/ture_outputs_all.npy", t_targets_all)
                            # np.save("./attention_numpy/pre_outputs_all.npy", t_outputs_all)
                            torch.save(self.model,
                                       'state_dict/test/' + self.opt.model_name + '_' + self.opt.dataset+'_'+str(max_test_f1) + '.pkl')
                            print('>>> this best model saved.this f1 is {:.4f}'.format(max_test_f1))
                            report = metrics.classification_report(t_targets_all.cpu(),
                                                                   t_outputs_all.cpu(),
                                                                   labels=[0, 1, 2], )
                            print(report)
                    print('\r >>> this repeat f1 is {:.4f}'.format(max_test_f1 ))
                    print('\r lr:{:E} loss: {:.4f}, acc: {:.4f}, test_loss_all{:.4f}，test_acc: {:.4f}, test_f1: {:.4f}'.format(optimizer.param_groups[0]['lr'],loss.item(), train_acc,
                                                                                                               test_loss,test_acc, test_f1))
            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase%3==0:
                    new_lr = opt.lr_de*optimizer.param_groups[0]['lr']
                    optimizer.param_groups[0]['lr'] = new_lr
                if continue_not_increase >= 7:
                    print('early stop.')
                    return max_test_acc, max_test_f1
                    break
            else:
                continue_not_increase = 0

        return max_test_acc, max_test_f1

    def _evaluate_acc_f1(self,criterion):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total,test_loss_all = 0, 0,0
        t_targets_all, t_outputs_all, = None, None,
        attention_all = []
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]

                t_targets = t_sample_batched['polarity'].to(opt.device)
                t_inputs = [t_inputs,False]

                t_outputs,attention_i = self.model(t_inputs)

                test_loss = criterion(t_outputs,t_targets)
                test_loss_all += test_loss.item()
                # print(t_outputs)
                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
                for i in attention_i.cpu().numpy().tolist():
                    attention_all.append(i)
        # print(len(attention_all))
        # print(t_targets_all.shape)
        test_acc = n_test_correct / n_test_total
        test_loss_all = test_loss_all
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')

        # out_target = t_targets_all.cpu().unsqueeze(-1)
        # out_putput = torch.argmax(t_outputs_all, -1).cpu().unsqueeze(-1)
        # not_correct = []
        # for i in range(out_putput.shape[0]):
        #     if out_putput[i] != out_target[i]:
        #         not_correct.append(i)
        #
        # opt_put_reslut = torch.cat((out_target, out_putput), dim=-1)
        # print(opt_put_reslut.shape)
        # opt_put_reslut = opt_put_reslut.numpy().tolist()
        # print(len(attention_all))
        #
        # file = open('./attention_numpy/opt_put_{0}_{1}.pickle'.format(self.opt.model_name, self.opt.dataset), 'wb')
        # file_2 = open('./attention_numpy/not_correct_{0}_{1}.txt'.format(self.opt.model_name, self.opt.dataset), 'w')
        # file_3 = open('./attention_numpy/{0}_{1}_attention.pickle'.format(self.opt.model_name, self.opt.dataset), 'wb')
        #
        # pickle.dump(opt_put_reslut, file)
        # file_2.write(str(not_correct))
        # pickle.dump(attention_all, file_3)
        # file.close()
        # file_2.close()
        # file_3.close()
        #
        # print(f1)
        # print(test_acc)
        # print(test_loss_all)



        return test_acc, f1,test_loss_all,t_targets_all.cpu(),torch.argmax(t_outputs_all, -1).cpu(),attention_all

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        time_str = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
        f_out = codecs.open('log/' + self.opt.model_name + '_' + self.opt.dataset +'_Syn_Layer_'+str(self.opt.Syn_Layer) +'_val.txt', 'a+',encoding="utf-8")
        total = sum([param.nelement() for param in self.model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))

        f_out.write('time:{0}\n'.format(time_str))
        arguments = " "
        for arg in vars(self.opt):
            arguments += '{0}: {1} '.format(arg, getattr(self.opt, arg))
        f_out.write(arguments)
        print('repeat: ', (i + 1))
        self._reset_params()
        # optimizer.param_groups[0]['lr'] = opt.learning_rate
        max_test_acc, max_test_f1 = self._train(criterion, optimizer)
        print("----------------------------")
        print('max_test_acc: {0}     max_test_f1: {1}'.format(max_test_acc, max_test_f1))
        print('#' * 100)
        f_out.write('max_test_acc: {0}, max_test_f1: {1}\n'.format(max_test_acc, max_test_f1))
        f_out.close()
        return  max_test_acc,max_test_f1


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', default='asgcn_new', type=str)
    parser.add_argument('--model_name', default='Relation_aware_Pos', type=str)  # Syntax_local_transformer,pos_gat，Type_aware_gat
    parser.add_argument('--dataset', default='rest14', type=str, help='lap14,twitter, rest14, rest15, rest16')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.005, type=float)
    parser.add_argument('--l2reg', default=0.0001, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--position_dim',default=100,type=int)
    parser.add_argument('--position_drop', default=0.4, type=float)



    parser.add_argument('--use_edge_weight', default="yes", type=str)
    parser.add_argument('--dependency_edge_dim',default=100,type=int)
    parser.add_argument('--edge_hidden_dim', default=600, type=int)

    # parser.add_argument('--gat_hidden_dim',default=100,type=int )
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--save', default=True, type=bool)
    parser.add_argument('--seed', default=3, type=int)  # 776
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--use_lstm_attention', default=True, type=bool)
    parser.add_argument('--use_bert', default=False, type=bool)
    parser.add_argument('--use_speech_weight', default=True, type=bool)
    parser.add_argument('--lcf', default="cdw",type=str)
    parser.add_argument('--Syn_Layer',default=2,type=int)
    parser.add_argument('--SRD',default=3,type=int)
    parser.add_argument('--lr_de', default=0.5, type=float)
    parser.add_argument('--context_length_type', default="context", type=str)  #context or syn
    parser.add_argument('--max_syntactic_distance', default=8, type=int)
    parser.add_argument('--GAT_alpha', default=0.2, type=float)
    parser.add_argument('--n_heads', default=4, type=int)
    parser.add_argument('--biff_layer_number', default=2, type=int)
    parser.add_argument('--use_gat_gru',default="yes",type=str)
    parser.add_argument('--use_p_embedding',default="yes" ,type =str)

    parser.add_argument('--hop', default=2, type=int)
    parser.add_argument("--train_type",default="trian",type = str)
    #transformer  encoder
    # parser.add_argument('--use_transformer',default=True,type=bool)
    parser.add_argument('--max_length', default=100, type=int)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--model_dim', default=300, type=int)
    parser.add_argument('--num_heads', default=6, type=int)
    parser.add_argument('--ffn_dim', default=512, type=int)
    parser.add_argument('--text_embed_dropout', default=0.5, type=float)
    parser.add_argument('--edge_embed_dropout', default=0.5, type=float)
    parser.add_argument('--liner_dropout', default=0.5, type=float)

    parser.add_argument('--graph_dropout', default=0.3, type=float)

    #gat_type
    parser.add_argument('--use_scaled_dot', default=False, type=bool)
    opt = parser.parse_args()
    model_classes = {

        'pos_gat': PoS_GAT,
        'Syntax_local_transformer':Syntax_local_transformer,
        'biff_lcf_gat' :Biff_Lcf_GAT,
        'Type_aware_gat':Type_aware_GAT,
        'Relation_aware_Pos':Relation_aware_Pos,
    }
    if opt.use_speech_weight ==True:
        input_colses = {
            'Syntax_local_transformer': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph_dir','dependency_graph_redir','dependency_graph_undir','dependency_edge_matrix','dependency_edge_matrix_re','dependency_edge_matrix_undir','position_syntax_indices ','speech_list'],
            'pos_gat': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph_dir','dependency_graph_redir','dependency_graph_undir','dependency_edge_matrix','dependency_edge_matrix_re','dependency_edge_matrix_undir','position_syntax_indices','speech_list'],
            'biff_lcf_gat': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph_dir','dependency_graph_redir','dependency_graph_undir','dependency_edge_matrix','dependency_edge_matrix_re','dependency_edge_matrix_undir','position_syntax_indices', 'speech_list'],
            'Type_aware_gat': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph_dir','dependency_graph_redir','dependency_graph_undir', 'dependency_edge_matrix','dependency_edge_matrix_re','dependency_edge_matrix_undir','position_syntax_indices', 'speech_list'],
            'Relation_aware_Pos': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph_dir','dependency_graph_redir','dependency_graph_undir', 'dependency_edge_matrix','dependency_edge_matrix_re','dependency_edge_matrix_undir','position_syntax_indices', 'speech_list'],

        }
    else:
        input_colses = {
            'Syntax_local_transformer': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph_dir','dependency_edge_matrix','dependency_graph_undir','position_syntax_indices'],
            'pos_gat': ['text_indices', 'aspect_indices', 'left_indices','dependency_graph_dir','dependency_graph_undir','position_syntax_indices'],
            'biff_lcf_gat': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph_dir',
                        'dependency_graph_undir','position_syntax_indices'],

        }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }


    opt.model_class = model_classes[opt.model_name]
    # summary(opt.model_class,input_size=(32,32,300))
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    # opt.device = torch.device('cpu')
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)


    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
    repeats = 3
    max_test_acc_avg = 0
    max_test_f1_avg = 0
    # if opt.train_type =="test":
    #     model_state= 'state_dict/' + opt.model_name + '_' + opt.dataset + '.pkl'
    #     model = ins.model
    #     model.load_state_dict(torch.load(model_state))
    #     print(model)
    #     exit()
    for i in range(repeats):
        ins = Instructor(opt)
        max_test_acc,max_test_f1=ins.run()
        max_test_acc_avg += max_test_acc
        max_test_f1_avg += max_test_f1
    f_out = codecs.open('log/' + opt.model_name + '_' + opt.dataset +'_Syn_Layer_'+str(opt.Syn_Layer)+ '_val.txt', 'a+',encoding="utf-8")
    print("max_test_acc_avg:", max_test_acc_avg / repeats)
    print("max_test_f1_avg:", max_test_f1_avg / repeats)
    f_out.write('max_test_acc_avg: {0}, max_test_f1_avg: {1}\n'.format(max_test_acc_avg / repeats,
                                                                       max_test_f1_avg / repeats))
    f_out.write("\n")


