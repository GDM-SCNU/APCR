# coding=utf-8
# Author: Jung
# Time: 2022/9/22 18:17


import warnings
warnings.filterwarnings("ignore")

import dgl.function as fn
import torch
import numpy as np
import dgl
from dgl.nn import GraphConv
import pickle as pkl
import torch.nn as nn
import argparse
from scipy import sparse
from sklearn import metrics
from ACDM.berpoDecoder import *
import random
import scipy.io as scio
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torch import LongTensor
import networkx as nx
from sklearn.cluster import KMeans
from torch.nn.parameter import Parameter
import pandas as pd
from scipy import sparse
import pickle as pkl
from ACDM.multiplex_data import *
import matplotlib.pyplot as plt
import seaborn as sns
from ACDM.t_sne import visualization

random.seed(826)
np.random.seed(826)
torch.manual_seed(826)
torch.cuda.manual_seed(826)

class Multiplex_VAGE(nn.Module):
    def __init__(self, graph_list, k, hid1_dim, hid2_dim):
        super(Multiplex_VAGE, self).__init__()
        self.vgae_list = nn.ModuleList([VAGE(graph_list[i], hid1_dim, hid2_dim) for i in range(len(graph_list))])
        self.num_nodes, self.feat_dim = graph_list[0].ndata['feat'].shape
        self.k = k


        self.feat_parameter = nn.Parameter(torch.FloatTensor(self.feat_dim, hid2_dim))
        torch.nn.init.xavier_uniform_(self.feat_parameter)
        self.attention = Attention(hid2_dim)






    def run(self):
        loss = 0
        emb_list = []
        com_list = []
        for i in range(len(self.vgae_list)):
            model = self.vgae_list[i]
            sampled_z, A_pred = model()
            emb_list.append(sampled_z)
            log_lik = model.norm * F.binary_cross_entropy(A_pred.view(-1), model.adj.view(-1), weight=model.weight_tensor)
            loss = loss + log_lik
            kl_divergence = 0.5 / A_pred.size(0) * (1 + 2 * model.logstd - model.mean ** 2 - torch.exp(model.logstd) ** 2).sum(1).mean()
            loss = loss - kl_divergence
            loss = loss + F.mse_loss(sampled_z, attr @ self.feat_parameter)
            loss = loss + torch.linalg.norm(self.feat_parameter.t() @ self.feat_parameter - torch.eye(self.feat_parameter.shape[1]))

        emb = self.attention(torch.stack(emb_list[:-1], dim=1))
        loss = loss + self._hilbert_schmidt_independence_criterion(emb, emb_list[-1]) * 1e-10#* 5e-8

        return loss, emb_list
    def _hilbert_schmidt_independence_criterion(self, emb1, emb2):
        R = torch.eye(self.num_nodes) - (1 / self.num_nodes) * torch.ones(self.num_nodes, self.num_nodes)
        K1 = torch.mm(emb1, emb1.t())  # 矩阵相乘
        K2 = torch.mm(emb2, emb2.t())
        RK1 = torch.mm(R, K1)
        RK2 = torch.mm(R, K2)
        HSIC = torch.trace(torch.mm(RK1, RK2))
        return HSIC

    def common_loss(self, emb1, emb2):
        emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
        emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
        emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
        emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
        cov1 = torch.matmul(emb1, emb1.t())
        cov2 = torch.matmul(emb2, emb2.t())
        cost = torch.mean((cov1 - cov2) ** 2)
        return cost

class Attention(nn.Module):
    def __init__(self, emb_dim, hidden_size= 64): # 10
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(emb_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)

class VAGE(nn.Module):
    def __init__(self, graph, hid1_dim, hid2_dim):
        super(VAGE, self).__init__()
        self.graph = graph
        self.label = graph.ndata['label']
        self.feat = graph.ndata['feat'].to(torch.float32)
        self.feat_dim = self.feat.shape[1]
        self.base_gcn = GraphConv(self.feat_dim, hid1_dim)
        self.gcn_mean = GraphConv(hid1_dim, hid2_dim)
        self.gcn_logstddev = GraphConv(hid1_dim, hid2_dim)
        self.hid2_dim = hid2_dim
        self.hid1_dim = hid1_dim
        self.adj = graph.adjacency_matrix().to_dense()
        self.norm = self.adj.shape[0] * self.adj.shape[0] / float((self.adj.shape[0] * self.adj.shape[0] - self.adj.sum()) * 2)
        self.pos_weight = float(self.adj.shape[0] * self.adj.shape[0] - self.adj.sum()) / self.adj.sum()

        weight_mask = self.adj.view(-1) == 1
        self.weight_tensor = torch.ones(weight_mask.size(0))
        self.weight_tensor[weight_mask] = self.pos_weight


    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))

        return A_pred

    def forward(self):
        hidden = self.base_gcn(self.graph, self.feat)
        hidden = torch.relu(hidden)
        self.mean = self.gcn_mean(self.graph, hidden)
        self.logstd = self.gcn_logstddev(self.graph, hidden)
        gaussian_noise = torch.randn(self.feat.size(0), self.hid2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean

        A_pred = self.dot_product_decode(sampled_z)

        return sampled_z, A_pred

def compute_nmi(pred, labels):
    return metrics.normalized_mutual_info_score(labels, pred)

def compute_ac(pred, labels):
    return metrics.accuracy_score(labels, pred)

def computer_f1(pred, labels):
    return metrics.f1_score(labels, pred, average='macro')

def computer_ari(true_labels, pred_labels):
    return metrics.adjusted_rand_score(true_labels, pred_labels)


"""
    A: adjacency - numpy
    num_coms: number of community
    z: pred (argmax())
    o: np.argsort(z)
"""
def plot_sparse_clustered_adjacency(A, num_coms, z, o, ax=None, markersize=0.25):
    if ax is None:
        ax = plt.gca()

    colors = sns.color_palette('hls', num_coms)
    sns.set_style('white')

    crt = 0
    for idx in np.where(np.diff(z[o]))[0].tolist() + [z.shape[0]]:
        ax.axhline(y=idx, linewidth=0.5, color='black', linestyle='--')
        ax.axvline(x=idx, linewidth=0.5, color='black', linestyle='--')
        crt = idx + 1

    ax.spy(A[o][:, o], markersize=markersize)
    ax.tick_params(axis='both', which='both', labelbottom='off', labelleft='off', labeltop='off')


if __name__ == "__main__":

    _, _, k, labels, adj_list, attr = load_data("scholat")
    for i in range(len(adj_list)):
        adj_list[i].ndata['label'] = torch.from_numpy(labels)
        adj_list[i].ndata['feat'] = attr

    model = Multiplex_VAGE(adj_list, k, 64, 32) # 100 64
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    feat = attr.t() @ attr

    for epoch in range(200):
        model.train()

        loss, emb_list = model.run()
        Z = torch.cat(emb_list, dim=1)
        # Z = emb_list



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        """ 绘制t-sne可视化 """
        # visualization(epoch, Z, labels)
        # continue
        model.eval()
        for i in range(3):
            pred = KMeans(n_clusters=k, n_init=16).fit_predict(Z.detach().numpy())

            """ 用于绘制对角线散点图 """
            # plt.figure(figsize=[10, 10])
            # # z = np.argmax(model.community_parameter.detach().numpy(), 1)
            # z = pred
            # o = np.argsort(z)
            # plot_sparse_clustered_adjacency(adj_list[-1].adjacency_matrix().to_dense(), k, z, o, markersize=0.05)
            # plt.savefig(r"fig\\" + "scholat_" + str(epoch) + "_" + str(i) + ".pdf")
            # plt.show()

            nmi = compute_nmi(pred, labels)
            ac = compute_ac(pred, labels)
            f1 = computer_f1(pred, labels)
            ari = computer_ari(labels, pred)
            print(
                'epoch={}, nmi: {:.4f}, f1_score={:.4f},  ac = {:.4f}, ari= {:.4f}'.format(
                    epoch,
                    nmi,
                    f1,
                    ac,
                    ari,
                ))
