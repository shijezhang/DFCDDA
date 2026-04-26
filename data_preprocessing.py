import os
import dgl
import torch
import random
import numpy as np
import pandas as pd
import networkx as nx
from parse_args import args
from scipy import sparse as sp
from sklearn.model_selection import StratifiedKFold


def get_data(original_data_path):
    d_data = dict()

    drf = pd.read_csv(os.path.join(original_data_path, 'DrugFingerprint.csv')).iloc[:, 1:].to_numpy()
    drg = pd.read_csv(os.path.join(original_data_path, 'DrugGIP.csv')).iloc[:, 1:].to_numpy()

    dip = pd.read_csv(os.path.join(original_data_path, 'DiseasePS.csv')).iloc[:, 1:].to_numpy()
    dig = pd.read_csv(os.path.join(original_data_path, 'DiseaseGIP.csv')).iloc[:, 1:].to_numpy()

    d_data['drug_number'] = int(drf.shape[0])
    d_data['disease_number'] = int(dig.shape[0])

    d_data['drf'] = drf
    d_data['drg'] = drg
    d_data['dip'] = dip
    d_data['dig'] = dig

    d_data['drdi'] = pd.read_csv(os.path.join(original_data_path, 'DrugDiseaseAssociationNumber.csv'),
                                 dtype=int).to_numpy()
    d_data['drpr'] = pd.read_csv(os.path.join(original_data_path, 'DrugProteinAssociationNumber.csv'),
                                 dtype=int).to_numpy()
    d_data['dipr'] = pd.read_csv(os.path.join(original_data_path, 'ProteinDiseaseAssociationNumber.csv'),
                                 dtype=int).to_numpy()

    d_data['drugfeature'] = pd.read_csv(os.path.join(original_data_path, 'Drug_mol2vec.csv'), header=None).iloc[:,
                            1:].to_numpy()
    d_data['diseasefeature'] = pd.read_csv(os.path.join(original_data_path, 'DiseaseFeature.csv'), header=None).iloc[:,
                               1:].to_numpy()
    d_data['proteinfeature'] = pd.read_csv(os.path.join(original_data_path, 'Protein_ESM.csv'), header=None).iloc[:,
                               1:].to_numpy()

    d_data['drugfeature'] = torch.FloatTensor(d_data['drugfeature'])
    d_data['diseasefeature'] = torch.FloatTensor(d_data['diseasefeature'])
    d_data['proteinfeature'] = torch.FloatTensor(d_data['proteinfeature'])

    d_data['protein_number'] = d_data['proteinfeature'].shape[0]

    d_data['drf_knn'] = get_sparse_knn(d_data['drf'])
    d_data['drg_knn'] = get_sparse_knn(d_data['drg'])
    d_data['dip_knn'] = get_sparse_knn(d_data['dip'])
    d_data['dig_knn'] = get_sparse_knn(d_data['dig'])

    d_data['fold_data'] = get_fold_data(d_data['drug_number'], d_data['disease_number'])

    return d_data


def get_adj(edges, size):
    edges_tensor = torch.LongTensor(edges).t()
    values = torch.ones(len(edges))
    adj = torch.sparse.LongTensor(edges_tensor, values, size).to_dense().long()

    return adj


def generate_drug_disease_training_samples(d_data, args):
    drdi_matrix = get_adj(d_data['drdi'], (d_data['drug_number'], d_data['disease_number']))
    l_one_index = []
    l_zero_index = []

    for i in range(drdi_matrix.shape[0]):
        for j in range(drdi_matrix.shape[1]):
            if drdi_matrix[i][j] >= 1:
                l_one_index.append([i, j])
            else:
                l_zero_index.append([i, j])

    random.seed(args.seed)
    random.shuffle(l_one_index)
    random.shuffle(l_zero_index)

    l_zero_index = l_zero_index[:int(args.negative_rate * len(l_one_index))]
    index = np.array(l_one_index + l_zero_index, dtype=int)
    label = np.array([1] * len(l_one_index) + [0] * len(l_zero_index), dtype=int)
    samples = np.concatenate((index, np.expand_dims(label, axis=1)), axis=1)

    drdi_p = samples[samples[:, 2] == 1, :]
    drdi_n = samples[samples[:, 2] == 0, :]

    drs_mean = (d_data['drf'] + d_data['drg']) / 2
    dis_mean = (d_data['dip'] + d_data['dig']) / 2

    drs = np.where(d_data['drf'] == 0, d_data['drg'], drs_mean)
    dis = np.where(d_data['dip'] == 0, d_data['dig'], dis_mean)

    d_data['dr_sim'] = drs
    d_data['di_sim'] = dis
    d_data['all_samples'] = samples
    d_data['all_drdi'] = samples[:, :2]
    d_data['all_drdi_p'] = drdi_p
    d_data['all_drdi_n'] = drdi_n
    d_data['all_label'] = label


def K_fold(d_data, args):
    skf = StratifiedKFold(n_splits=args.K_fold, random_state=None, shuffle=False)
    X = d_data['all_drdi']
    Y = d_data['all_label']
    X_train_all, X_test_all, Y_train_all, Y_test_all = [], [], [], []

    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_train = np.expand_dims(Y_train, axis=1).astype('float64')
        Y_test = np.expand_dims(Y_test, axis=1).astype('float64')
        X_train_all.append(X_train)
        X_test_all.append(X_test)
        Y_train_all.append(Y_train)
        Y_test_all.append(Y_test)

    d_data['X_train'] = X_train_all
    d_data['X_test'] = X_test_all
    d_data['Y_train'] = Y_train_all
    d_data['Y_test'] = Y_test_all


def KNN_matrix(matrix, k, isBool=True):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape, dtype=float)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)

    for i in range(num):
        if isBool:
            knn_graph[i, idx_sort[i, :k]] = 1
            knn_graph[idx_sort[i, :k], i] = 1
        else:
            knn_graph[i, idx_sort[i, :k]] = matrix[i, idx_sort[i, :k]]
            knn_graph[idx_sort[i, :k], i] = matrix[idx_sort[i, :k], i]

        knn_graph[i, i] = 1
    return knn_graph


def knn_graph(matrix, k):
    k_neighbor = np.argpartition(-matrix, kth=k, axis=1)[:, :k]
    row_index = np.arange(k_neighbor.shape[0]).repeat(k_neighbor.shape[1])
    col_index = k_neighbor.reshape(-1)
    edges = np.array([row_index, col_index]).astype(int).T

    knn = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(matrix.shape[0], matrix.shape[0]),
                        dtype=np.float32)
    knn_sp_graph = knn + knn.T.multiply(knn.T > knn) - knn.multiply(knn.T > knn)

    return knn_sp_graph


def process_similarity_graph(d_data, args):
    drdr_matrix_bool = KNN_matrix(d_data['dr_sim'], args.knn_neighbor, isBool=True)
    didi_matrix_bool = KNN_matrix(d_data['di_sim'], args.knn_neighbor, isBool=True)

    d_data["drdr_matrix"] = drdr_matrix_bool
    d_data["didi_matrix"] = didi_matrix_bool

    drdr_matrix = KNN_matrix(d_data['dr_sim'], args.knn_neighbor, isBool=False)
    didi_matrix = KNN_matrix(d_data['di_sim'], args.knn_neighbor, isBool=False)

    drdr_nx = nx.from_numpy_array(drdr_matrix)
    didi_nx = nx.from_numpy_array(didi_matrix)

    drdr_graph = dgl.from_networkx(drdr_nx)
    didi_graph = dgl.from_networkx(didi_nx)

    drdr_graph.ndata['sim_feature'] = torch.tensor(d_data['dr_sim'])
    didi_graph.ndata['sim_feature'] = torch.tensor(d_data['di_sim'])

    return drdr_graph, didi_graph


def process_data(args):
    original_data_path = os.path.join(os.getcwd(), "./dataset/{}".format(args.dataset))
    d_data = get_data(original_data_path)

    generate_drug_disease_training_samples(d_data, args)
    K_fold(d_data, args)

    drdr_graph, didi_graph = process_similarity_graph(d_data, args)

    return d_data, drdr_graph, didi_graph


def dgl_heterograph(d_data, drdi):
    l_drdi, l_drpr, l_dipr = [], [], []
    l_didr, l_prdr, l_prdi = [], [], []

    for i in range(drdi.shape[0]):
        l_drdi.append(drdi[i])
        l_didr.append(drdi[i][::-1])

    for i in range(d_data['drpr'].shape[0]):
        l_drpr.append(d_data['drpr'][i])
        l_prdr.append(d_data['drpr'][i][::-1])

    for i in range(d_data['dipr'].shape[0]):
        l_dipr.append(d_data['dipr'][i])
        l_prdi.append(d_data['dipr'][i][::-1])

    node_dict = {
        'drug': d_data['drug_number'],
        'disease': d_data['disease_number'],
        'protein': d_data['protein_number']
    }

    heterograph_dict = {
        ('drug', 'association', 'disease'): (l_drdi),
        ('drug', 'association', 'protein'): (l_drpr),
        ('disease', 'association', 'protein'): (l_dipr),
        ('disease', 'association', 'drug'): (l_didr),
        ('protein', 'association', 'drug'): (l_prdr),
        ('protein', 'association', 'disease'): (l_prdi)

    }

    heterograph = dgl.heterograph(heterograph_dict, num_nodes_dict=node_dict)

    return heterograph


def get_sparse_knn(similarity_matrix):
    knn_sp = knn_graph(similarity_matrix, args.knn_neighbor)
    knn_sp = normalize(knn_sp + sp.eye(knn_sp.shape[0]))
    knn_sp = sparse_mx_to_torch_sparse_tensor(knn_sp)

    return knn_sp


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


def get_fold_data(dr_num, di_num):
    fold_data = {}
    cv_data_dict = get_cv_data()
    for cv in range(0, 10):
        train_data, test_data, values = cv_data_dict[cv]
        shuffled_idx = np.random.permutation(train_data.shape[0])
        train_rel_info = train_data.iloc[shuffled_idx[::]]
        test_rel_info = test_data
        train_pairs, train_values = generate_pair_value(train_rel_info)
        test_pairs, test_values = generate_pair_value(test_rel_info)

        train_enc_graph = generate_enc_graph(train_pairs, train_values, values, dr_num, di_num, add_support=True)
        train_dec_graph = generate_dec_graph(train_pairs, dr_num, di_num)
        train_truths = torch.FloatTensor(train_values)

        test_enc_graph = generate_enc_graph(test_pairs, test_values, values, dr_num, di_num, add_support=True)
        test_dec_graph = generate_dec_graph(test_pairs, dr_num, di_num)
        test_truths = torch.FloatTensor(test_values)
        fold_data[cv] = {'train': [train_enc_graph, train_dec_graph, train_truths],
                         'test': [test_enc_graph, test_dec_graph, test_truths]}

    return fold_data


def get_cv_data():
    cv_data = {}
    for fold_id in range(10):
        fold_path = os.path.join('./dataset/{}/fold/'.format(args.dataset), str(fold_id))
        train_path = os.path.join(fold_path, "data_train.csv")
        test_path = os.path.join(fold_path, "data_test.csv")

        train_df = pd.read_csv(train_path, index_col=0, header=0).astype(
            {"drug": int, "disease": int, "label": int})
        test_df = pd.read_csv(test_path, index_col=0, header=0).astype(
            {"drug": int, "disease": int, "label": int})

        train_data_info = train_df.rename(columns={
            'drug': 'drug_id',
            'disease': 'disease_id',
            'label': 'values'
        })
        test_data_info = test_df.rename(columns={
            'drug': 'drug_id',
            'disease': 'disease_id',
            'label': 'values'
        })

        values = np.unique(train_data_info['values'].values)
        cv_data[fold_id] = [train_data_info, test_data_info, values]

    return cv_data


def generate_dec_graph(rating_pairs, dr_num, di_num):
    ones = np.ones_like(rating_pairs[0])
    drug_disease_rel_coo = sp.coo_matrix(
        (ones, rating_pairs),
        shape=(dr_num, di_num), dtype=np.float32)
    g = dgl.bipartite_from_scipy(drug_disease_rel_coo, utype='_U', etype='_E', vtype='_V')
    return dgl.heterograph({('drug', 'rate', 'disease'): g.edges()},
                           num_nodes_dict={'drug': dr_num, 'disease': di_num})


def generate_enc_graph(rating_pairs, rating_values, possible_rel_values, dr_num, di_num, add_support=False):
    data_dict = dict()
    num_nodes_dict = {'drug': dr_num, 'disease': di_num}
    rating_row, rating_col = rating_pairs

    for rating in possible_rel_values:
        ridx = np.where(
            rating_values == rating)
        rrow = rating_row[ridx]
        rcol = rating_col[ridx]
        rating = to_etype_name(rating)
        data_dict.update({
            ('drug', str(rating), 'disease'): (rrow, rcol),
            ('disease', 'rev-%s' % str(rating), 'drug'): (rcol, rrow)
        })

    graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

    if add_support:
        def _calc_norm(x):
            x = x.numpy().astype('float32')
            x[x == 0.] = np.inf
            x = torch.FloatTensor(1. / np.sqrt(x))
            return x.unsqueeze(1)

        drug_ci = []
        drug_cj = []
        disease_ci = []
        disease_cj = []

        for r in possible_rel_values:
            r = to_etype_name(r)
            drug_ci.append(graph['rev-%s' % r].in_degrees())
            disease_ci.append(graph[r].in_degrees())
            drug_cj.append(graph[r].out_degrees())
            disease_cj.append(graph['rev-%s' % r].out_degrees())

        drug_ci = _calc_norm(sum(drug_ci))
        disease_ci = _calc_norm(sum(disease_ci))
        drug_cj = _calc_norm(sum(drug_cj))
        disease_cj = _calc_norm(sum(disease_cj))

        graph.nodes['drug'].data.update({'ci': drug_ci, 'cj': drug_cj})
        graph.nodes['disease'].data.update({'ci': disease_ci, 'cj': disease_cj})

    return graph


def generate_pair_value(rel_info):
    rating_pairs = (np.array([ele for ele in rel_info["drug_id"]], dtype=np.int64),
                    np.array([ele for ele in rel_info["disease_id"]], dtype=np.int64))
    rating_values = rel_info["values"].values.astype(np.float32)
    return rating_pairs, rating_values


def to_etype_name(rating):
    return str(rating).replace('.', '_')
