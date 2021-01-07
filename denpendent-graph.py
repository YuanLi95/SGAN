# -*- coding: utf-8 -*-

import numpy as np
import pickle
import os
import spacy
import torch
import sys
import scipy.sparse as sp
from spacy import displacy
from pathlib import Path
import  data_utils
from data_utils import Tokenizer
import networkx as nx
nlp = spacy.load('en_core_web_sm')


def aspect_short_path(G, target):
    """"
    """
    d = nx.shortest_path_length(G, target=target)
    distance_list = []
    for node in G.nodes():
        try:
            distance_list.append(d[node])
        except KeyError:
            distance_list.append(-1)
    return distance_list



def dependency_adj_matrix(text,aspect_double_idx):
    # text = "Great food but the service was dreadful !"

    document = nlp(text)


    seq_len = len(text.split())
    matrix_dir = np.zeros([seq_len,seq_len]).astype('float32')
    Syntactic_dependence=[]
    matrix_undir = np.zeros([seq_len, seq_len]).astype('float32')
    matrix_redir = np.zeros([seq_len, seq_len]).astype('float32')
    for token in document:
        if token.i <seq_len:
            # Syntactic_dependence.append([token.i, token.dep_.lower(), token.head.i])
            # Syntactic_dependence.append([token.head.i, token.dep_.lower(), token.i])
            Syntactic_dependence.append([token.head.i, (token.head.pos_ + token.dep_).lower(), token.i])
            Syntactic_dependence.append([token.i, (token.head.pos_ + token.dep_).lower(),token.head.i ])
            # matrix_dir[token.i][token.i] = 1
            # matrix_undir[token.i][token.i] = 1
            # matrix_redir[token.i][token.i] = 1
            for child in token.children:
                if child.i <seq_len:
                    # matrix_dir[token.i][child.i] = 1
                    matrix_dir[child.i][token.i] = 1
                    matrix_undir[token.i][child.i] = 1
                    matrix_undir[child.i][token.i] = 1
                    matrix_redir[token.i][child.i] = 1
    G = nx.from_numpy_matrix(matrix_undir)
    aspect_begin_idx = aspect_double_idx[0]
    aspect_end_idx = aspect_double_idx[1]
    distance_aspect_begin = np.array(aspect_short_path(G, aspect_begin_idx))
    distance_aspect_end = np.array(aspect_short_path(G, aspect_end_idx))
    distance_aspect = np.array((distance_aspect_begin + distance_aspect_end)/2).astype(np.int32)
    distance_aspect[aspect_double_idx] = 0
    distance_aspect[aspect_end_idx]=0


    # matrix= normalize(matrix)
    return matrix_dir,matrix_redir,matrix_undir,Syntactic_dependence,distance_aspect


def token_speech_weight(token):
    if token == "ADJ":  # 关注形容词
        simple_weigth = 1.0
    elif (token =="CCONJ"):
        simple_weigth = 1.0
    elif (token == "ADV"):  # 副词及其连接词
        simple_weigth = 1.0
    else:
        simple_weigth = 1.0

    return simple_weigth

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 对每一个特征进行归一化
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

# 获得词性的一个list
def  Part_of_speech_list(text):
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    return  pos

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph_dir = {}
    idx2graph_redir ={}
    idx2graph_undir ={}
    Syntactic_dependence_all = {}
    idx2positon = {}
    part_of_speech = {}
    fout_dir= open(filename +'dir'+ '.graph', 'wb')
    fout_redir = open(filename+"redir"+ '.graph', 'wb')
    fout_undir = open(filename +'undir'+ '.graph', 'wb')
    speech = open(filename + '.speech', 'wb')
    dependency_analysis = open(filename+'.dependency','wb')
    fout_syntax_position = open(filename + '.syntax', 'wb')

    for i in range(0, len(lines), 3):

        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        text = text_left+" "+aspect+""+text_right
        words = text.split()
        aspect_list = aspect.split()
        text_left_list = text_left.split()
        aspect_double_idx = [len(text_left_list), len(text_left_list)+len(aspect_list)-1]

        part_of_speech_vector = Part_of_speech_list(text_left + ' '+aspect +' '+ text_right)
        adj_matrix_dir,adj_matrix_redir, adj_matrix_undir,Syntactic_dependence,distance_aspect = dependency_adj_matrix(text_left + ' ' + aspect + ' ' + text_right,aspect_double_idx)
        idx2graph_dir[i] = adj_matrix_dir
        idx2graph_redir[i] = adj_matrix_redir
        idx2graph_undir[i] = adj_matrix_undir
        part_of_speech[i] = list(map(lambda x :token_speech_weight(x),part_of_speech_vector))
        Syntactic_dependence_all[i] = Syntactic_dependence
        #syntax_position_distance
        idx2positon[i] = distance_aspect


    pickle.dump(idx2graph_dir, fout_dir)
    pickle.dump(idx2graph_undir, fout_undir)
    pickle.dump(idx2graph_redir,fout_redir)
    pickle.dump(part_of_speech, speech)
    # print(Syntactic_dependence)
    pickle.dump(Syntactic_dependence_all,dependency_analysis)
    pickle.dump(idx2positon,fout_syntax_position)
    fout_dir.close()
    fout_redir.close()
    fout_undir.close()
    speech.close()
    dependency_analysis.close()
    fout_syntax_position.close()

if __name__ == '__main__':
    # process('./datasets/acl-14-short-data/train.raw')
    # process('./datasets/acl-14-short-data/test.raw')
    process('./datasets/semeval14/restaurant_train.raw')
    process('./datasets/semeval14/restaurant_test.raw')
    # process('./datasets/semeval14/laptop_train.raw')
    # process('./datasets/semeval14/laptop_test.raw')
    # process('./datasets/semeval15/restaurant_train.raw')
    # process('./datasets/semeval15/restaurant_test.raw')
    # process('./datasets/semeval16/restaurant_train.raw')
    # process('./datasets/semeval16/restaurant_test.raw')
    # # fin= open("./datasets/acl-14-short-data/train.raw.dependency","rb")
    # dependency = pickle.load(fin)
    # print(dependency[0])
    # dependency_0 = [i[1] for i in dependency[0]]
    # print(dependency_0)
    # fin.close()
    # exit()
    # doc = nlp("The price is reasonable although the service is poor .")
    # svg = spacy.displacy.render(doc, style="dep", jupyter=False)
    # file_name = '-'.join([w.text for w in doc if not w.is_punct]) + ".svg"
    # output_path = Path("./" + file_name)
    # output_path.open("w", encoding="utf-8").write(svg)
    # exit()
