# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import  torch


def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            try:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                print('WARNING: corrupted word vector of {} when being loaded from GloVe.'.format(tokens[0]))
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):

    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)

    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(embed_dim), 1 / np.sqrt(embed_dim), (1, embed_dim))
        fname = './GloVe/glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
                print(vec.shape)
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


def build_dependency_matrix(dependency2idx, dependency_dim, type):
    embedding_matrix_file_name = '{0}_{1}_dependency_matrix.pkl'.format(str(dependency_dim), type)
    # if os.path.exists(embedding_matrix_file_name):
    #     print('loading embedding_matrix:', embedding_matrix_file_name)
    #     embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    # else:
    print('loading edge vectors ...')
    embedding_matrix = np.zeros((len(dependency2idx), dependency_dim))  # idx 0 and 1 are all-zeros
    embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(dependency_dim), 1 / np.sqrt(dependency_dim), (1, dependency_dim))
    # embedding_matrix[1, :] = np.random.uniform(-1, 0.25, (1, dependency_dim))

    print('building edge_matrix:', embedding_matrix_file_name)
    pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix



def build_position_matrix(position2idx, position_dim, type):
    embedding_matrix_file_name = '{0}_{1}_position_matrix.pkl'.format(str(position_dim), type)

    embedding_matrix = np.zeros((len(position2idx), position_dim))  # idx 0 and 1 are all-zeros
    # embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(position_dim), 1 / np.sqrt(position_dim), (1, position_dim))
    embedding_matrix[1, :] = np.random.uniform(-0.25, 0.25, (1, position_dim))


    print('building position_matrix:', embedding_matrix_file_name)
    pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


class Dependecynizer(object):
    def __init__(self, dependency2idx=None):
        if dependency2idx is None:
            self.dependency2idx = {}
            self.idx2dependency = {}
            self.idx2dependency_number={}
            self.idx = 0
            self.dependency2idx['<pad>'] = self.idx
            self.idx2dependency[self.idx] = '<pad>'
            self.idx2dependency_number['<pad>']=1
            self.idx += 1
            self.dependency2idx['<unk>'] = self.idx
            self.idx2dependency[self.idx] = '<unk>'
            self.idx2dependency_number['<unk>'] = 1
            self.idx += 1
        else:
            self.dependency2idx = dependency2idx
            self.idx2dependency = {v: k for k, v in dependency2idx.items()}
            self.idx2dependency_number = {v: k for k, v in dependency2idx.items()}
        self.idx2dependency_number = {}
    def fit_on_dependency(self, dependency_edge):
        dependency_edges = dependency_edge.lower()
        dependency_edges = dependency_edges.split()
        for dependency_edge in dependency_edges:
            if dependency_edge not in self.dependency2idx:
                self.dependency2idx[dependency_edge] = self.idx
                self.idx2dependency[self.idx] = dependency_edge
                self.idx2dependency_number[dependency_edge]=1
                self.idx += 1
            else:
                self.idx2dependency_number[dependency_edge] += 1
    def dependency_to_index(self,dependency_edge,idx2gragh_dir):
        edge_matrix = np.zeros_like(idx2gragh_dir,dtype=int)
        edge_matrix_re = np.zeros_like(idx2gragh_dir, dtype=int)
        edge_matrix_undir = np.zeros_like(idx2gragh_dir, dtype= int)
        matrix_len = (edge_matrix.shape)[0]

        unknownidx = 1
        for i in dependency_edge:
            try:
                if (matrix_len>int(i[0]))&(matrix_len>int(i[2])):
                    edge_matrix[i[0]][i[2]] = self.dependency2idx[i[1]] if i[1] in self.dependency2idx else unknownidx
                    edge_matrix_re[i[2]][i[0]] = self.dependency2idx[i[1]] if i[1] in self.dependency2idx else unknownidx

                    edge_matrix_undir[i[2]][i[0]] = self.dependency2idx[i[1]] if i[1] in self.dependency2idx else unknownidx
                    edge_matrix_undir[i[0]][i[2]] = self.dependency2idx[i[1]] if i[1] in self.dependency2idx else unknownidx

            except IndexError:
                print(matrix_len)
                print(dependency_edge)
        return edge_matrix,edge_matrix_re,edge_matrix_undir


class Positionnizer(object):
    def __init__(self, position2idx=None):
        if position2idx is None:
            self.position2idx = {}
            self.idx2position = {}
            self.idx = 0
            self.position2idx['<pad>'] = self.idx
            self.idx2position[self.idx] = '<pad>'
            self.idx += 1
            self.position2idx['<unk>'] = self.idx
            self.idx2position[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.position2idx = position2idx
            self.idx2position = {v: k for k, v in position2idx.items()}

    def fit_on_position(self, syntax_positions):
        for syntax_position in syntax_positions:
            if syntax_position not in self.position2idx:
                self.position2idx[syntax_position] = self.idx
                self.idx2position[self.idx] = syntax_position

                self.idx += 1
    def position_to_index(self,position_sequence):
        position_sequence = position_sequence.astype(np.str)
        unknownidx = 1
        position_matrix = [self.position2idx[w] if w in self.position2idx else unknownidx for w in position_sequence]
        return position_matrix


class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence


class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        dependency_all =''
        syntax_position_all= []
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            dependency_file = open(fname+".dependency",'rb')
            syntax_file = open(fname+".syntax","rb")
            # print(fname+".dependency")
            lines = fin.readlines()
            dependency = pickle.load(dependency_file)
            fin.close()
            dependency_file.close()
            syntax_position = pickle.load(syntax_file)
            syntax_file.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                # text += text_raw + " "
                text += text_raw + " "
                dependency_all+=  " ".join([i[1] for i in dependency[i]])
                dependency_all+=" "
                for j in syntax_position[i]:
                    syntax_position_all.append(str(j))
            # print(syntax_position_all)
        return text,dependency_all,syntax_position_all

    @staticmethod
    def __read_data__(fname, tokenizer,dependency_tokenizer,position_tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        fin = open(fname + 'dir.graph', 'rb')
        idx2gragh_dir = pickle.load(fin)
        fin.close()
        fin_redir = open(fname + 'redir.graph', 'rb')
        idx2gragh_redir = pickle.load(fin_redir)
        fin_redir.close()
        fin_undir = open(fname + 'undir.graph', 'rb')
        idx2gragh_undir = pickle.load(fin_undir)
        fin_undir.close()


        fin = open(fname + '.speech', 'rb')
        speech_all = pickle.load(fin)
        fin.close()
        fin = open(fname + '.dependency', 'rb')
        dependency_file = pickle.load(fin)
        fin.close()

        fin = open(fname + '.syntax', 'rb')
        position_syntax_file = pickle.load(fin)
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):

            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)

            context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_indices = tokenizer.text_to_sequence(text_left)
            polarity = int(polarity) + 1
            #

            dependency_graph_dir = idx2gragh_dir[i]
            dependency_graph_redir = idx2gragh_redir[i]
            dependency_graph_undir = idx2gragh_undir[i]


            dependency_edge_matrix,dependency_edge_matrix_re,dependency_edge_matrix_undir = dependency_tokenizer.dependency_to_index(dependency_file[i],idx2gragh_dir[i])

            position_syntax_matrix = position_tokenizer.position_to_index(position_syntax_file[i])

            speech_list = speech_all[i]

            data = {
                'text_indices': text_indices,
                'context_indices': context_indices,
                'aspect_indices': aspect_indices,
                'left_indices': left_indices,
                'polarity': polarity,
                'dependency_graph_dir': dependency_graph_dir,
                'dependency_graph_redir':dependency_graph_redir,
                'dependency_graph_undir': dependency_graph_undir,
                'dependency_edge_matrix': dependency_edge_matrix,
                'dependency_edge_matrix_re':dependency_edge_matrix_re,
                'dependency_edge_matrix_undir':dependency_edge_matrix_undir,
                'speech_list' : speech_list,
                'position_syntax_matrix':position_syntax_matrix,
            }

            all_data.append(data)
        return all_data

    def __init__(self, position_dim,dataset='rest14', embed_dim=300,dependency_dim=100,use_bert=False,max_len=70,opt=1):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            'twitter': {
                'train': './datasets/acl-14-short-data/train.raw',
                'test': './datasets/acl-14-short-data/test.raw'
            },
            'rest14': {
                'train': './datasets/semeval14/restaurant_train.raw',
                'test': './datasets/semeval14/restaurant_test.raw'
            },
            'lap14': {
                'train': './datasets/semeval14/laptop_train.raw',
                'test': './datasets/semeval14/laptop_test.raw'
            },
            'rest15': {
                'train': './datasets/semeval15/restaurant_train.raw',
                'test': './datasets/semeval15/restaurant_test.raw'
            },
            'rest16': {
                'train': './datasets/semeval16/restaurant_train.raw',
                'test': './datasets/semeval16/restaurant_test.raw'
            },

        }
        text,dependency_all,syntax_position_all = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']])
        # print(dependency_all)
        # print(text)
        # exit()

        # if use_bert=False

        if os.path.exists(dataset + '_word2idx.pkl'):
            print("loading {0} tokenizer...".format(dataset))
            with open(dataset + '_word2idx.pkl', 'rb') as f:
                word2idx = pickle.load(f)
                tokenizer = Tokenizer(word2idx=word2idx)
        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_text(text)
            with open(dataset + '_word2idx.pkl', 'wb') as f:
                pickle.dump(tokenizer.word2idx, f)
         #构建边的映射关系
        if os.path.exists(dataset + '_dependency2idx.pkl'):
            print("loading {0} tokenizer...".format(dataset))
            with open(dataset + '_dependency2idx.pkl', 'rb') as f:
                dependency2idx = pickle.load(f)
                dependency_tokenizer = Dependecynizer(dependency2idx=dependency2idx)
                #
        else:

            dependency_tokenizer = Dependecynizer()
            dependency_tokenizer.fit_on_dependency (dependency_all)
            with open(dataset + '_dependency2number.pkl', 'wb') as f:
                pickle.dump(dependency_tokenizer.idx2dependency_number,f)
                print(dependency_tokenizer.idx2dependency_number)
            with open(dataset + '_dependency2idx.pkl', 'wb') as f:
                pickle.dump(dependency_tokenizer.dependency2idx, f)
                print(dependency_tokenizer.dependency2idx)

        # 构建position的映射关系
        if os.path.exists(dataset + '_position2idx.pkl'):
            print("loading {0} position_tokenizer...".format(dataset))
            with open(dataset + '_position2idx.pkl', 'rb') as f:
                position2idx = pickle.load(f)
                position_tokenizer = Positionnizer(position2idx=position2idx )
        else:
            position_tokenizer = Positionnizer()
            position_tokenizer.fit_on_position(syntax_position_all)
            with open(dataset + '_position2idx.pkl', 'wb') as f:
                pickle.dump(position_tokenizer.position2idx, f)
        print(dependency_tokenizer.dependency2idx)
        # print(position_tokenizer.position2idx)
        # print(dependency_tokenizer.dependency2idx)

        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)




        self.dependency_matrix = build_dependency_matrix(dependency_tokenizer.dependency2idx,dependency_dim,dataset)

        self.position_matrix = build_position_matrix(position_tokenizer.position2idx, position_dim, dataset)


        # #构建edge
        # token_index = []
        # dependency_number = self.dependency_matrix.shape[0]
        #
        # for i in range(1,dependency_number+1):
        #     token_index.append(i)
        # token_index =torch.Tensor(token_index)
        #
        # unadj_relationship = torch.zeros((dependency_number,dependency_number))
        # unadj_relationship[0] = torch.Tensor([int(i) for i in range(dependency_number)])
        # unadj_relationship = unadj_relationship.repeat(2,1,1)
        # unadj =torch.zeros((dependency_number,dependency_number)).long()
        # unadj[0] =torch.Tensor([int(1) for i in range(dependency_number)]).long()
        # token_index = token_index.unsqueeze(0).repeat(2,1).long()
        # position_index = torch.zeros_like(token_index).long()
        # unadj = unadj.unsqueeze(0).repeat(2,1,1).long()
        #
        # aspect_index = [0 for i in range(dependency_number)]
        # aspect_left = [0 for i in range(dependency_number)]
        # for i in range(1,100):
        #     aspect_left[i-1] = i
        # aspect_index[0]=32
        # aspect_index[1] =33
        # aspect_index = torch.Tensor([aspect_index]).repeat(2,1).long()
        # aspect_left = torch.Tensor([aspect_left]).repeat(2,1).long()
        # pad = torch.ones(token_index.shape[1]).repeat(2,1).long()
        # # print(aspect_left)
        # edge_inputs = [token_index,aspect_index , aspect_left,unadj,unadj,unadj,unadj_relationship,unadj_relationship,unadj_relationship,position_index,pad]
        #
        # edge_inputs = [i.to(opt.device) for i in edge_inputs]
        #
        # # print(edge_inputs)
        # a_numpy = np.array(edge_inputs)
        # print(a_numpy)
        # np.save('./state_dict/{0}_numpy.npy'.format(opt.dataset), a_numpy)
        # exit()





        self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer,dependency_tokenizer,position_tokenizer))

        self.test_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['test'], tokenizer,dependency_tokenizer,position_tokenizer))
