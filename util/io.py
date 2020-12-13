import io
import pickle
from tqdm import tqdm
import torch
import networkx as nx
import numpy as np

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_vectors(fname, word2id):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    if "numberbatch" in fname:
        fin.readline() # skip the first line
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        if tokens[0] in word2id:
            data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def load_nx_graph_hopk(path, word2id, keyword2id, remove_self_loop=False):
    g = nx.read_gpickle(path)
    
    graph_dict = {}
    edge_index = []
    edge_mask = np.zeros((len(keyword2id), len(keyword2id)))

    node2id = {
        "<pad>": 0,
        "<unk>": 1
    }
    
    max_node_length = 5
    nodeid2wordid = [
        [word2id["<pad>"]]*max_node_length,
        [word2id["<unk>"]]*max_node_length
    ]

    for node1, node2, in g.edges:
        if remove_self_loop and node1 == node2:
            continue

        if node1 not in node2id:
            node2id[node1] = len(node2id)
            nodeid2wordid.append([word2id[w] for w in node1.split("_")[:max_node_length]] + [word2id["<pad>"]]*(max_node_length-1 - node1.count("_"))) # node2id mapping to word2id
        if node2 not in node2id:
            node2id[node2] = len(node2id)
            nodeid2wordid.append([word2id[w] for w in node2.split("_")[:max_node_length]] + [word2id["<pad>"]]*(max_node_length-1 - node2.count("_"))) # node2id mapping to word2id
        edge_index.append([node2id[node1], node2id[node2]])

        if node1 in keyword2id and node2 in keyword2id:
            edge_mask[keyword2id[node1], keyword2id[node2]] = 1
            edge_mask[keyword2id[node2], keyword2id[node1]] = 1

    graph_dict["edge_index"] = edge_index
    graph_dict["nodeid2wordid"] = nodeid2wordid
    graph_dict["node2id"] = node2id
    graph_dict["edge_mask"] = edge_mask
    return graph_dict


def load_edge_mask(path, keyword2id, remove_self_loop=True):
    g = nx.read_gpickle(path)
    edge_mask = np.zeros((len(keyword2id), len(keyword2id)))
    for node1, node2, in g.edges:
        if remove_self_loop and node1 == node2:
            continue

        if node1 in keyword2id and node2 in keyword2id:
            edge_mask[keyword2id[node1], keyword2id[node2]] = 1
            edge_mask[keyword2id[node2], keyword2id[node1]] = 1
    
    return edge_mask

