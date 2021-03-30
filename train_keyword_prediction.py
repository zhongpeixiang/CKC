import argparse
import itertools
import json
import logging
import multiprocessing as mp
import os
import pickle
import random
import re
import string
import sys
import time
from collections import Counter, OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import networkx as nx

from model import KW_GNN
from util.io import load_pickle, save_pickle, load_vectors, load_nx_graph_hopk
from util.tool import count_parameters
from util.data import pad_and_clip_data, build_vocab, convert_convs_to_ids, create_batches_keyword_prediction

logging.basicConfig(level = logging.INFO, \
                    format = '%(asctime)s  %(levelname)-5s %(message)s', \
                    datefmt =  "%Y-%m-%d-%H-%M-%S")

def cprint(*args):
    text = ""
    for arg in args:
        text += "{0} ".format(arg)
    logging.info(text)

def compute_CE(logits, target, batch_vocab_mask=None):
    """
        logits: (batch, vocab_size)
        target: (batch, seq_len)
        batch_vocab_mask: (batch, vocab_size)
        target_weight: (batch, seq_len)
    """
    target_mask = target.ne(0).float() # (batch, seq_len)
    # cprint("-"*60)
    # cprint(target_mask.sum())

    if batch_vocab_mask is not None:
        logits = (1-batch_vocab_mask)*(-5e4) + batch_vocab_mask*logits # (batch, vocab_size), masked logits
        target_mask = target_mask * torch.gather(batch_vocab_mask, dim=1, index=target) # (batch, seq_len), masked target mask
    
    logits = F.log_softmax(logits, dim=-1)
    loss = -1 * (torch.gather(logits, dim=1, index=target) * target_mask).sum() # negative log-likelihood loss
    loss = loss/target_mask.sum()
    return loss


def compute_metrics(logits, target, batch_vocab_mask=None):
    """
        logits: (batch, vocab_size)
        target: (batch, seq_len)
        batch_vocab_mask: (batch, vocab_size)
    """
    # logits = torch.rand_like(logits) # random baseline
    if batch_vocab_mask is not None:
        logits = (1-batch_vocab_mask)*(-5e4) + batch_vocab_mask*logits # (batch, vocab_size), masked logits
    
    # recall@k
    sorted_indices = logits.sort(descending=True)[1]
    targets = target.tolist()
    
    precisions = []
    recalls = []
    ks = [1, 3, 5]
    for k in ks:
        # sorted_indices[:,:k]: (batch_size, k)
        precision_k = []
        recall_k = []
        for tgts, topk in zip(targets, sorted_indices[:,:k].tolist()):
            tgts = [t for t in tgts if t != 0] # tgts
            if len(tgts) == 0:
                continue
            num_hit = len(set(topk).intersection(set(tgts)))
            precision_k.append(num_hit/len(topk))
            recall_k.append(num_hit/len(tgts))
        precisions.append(np.mean(precision_k))
        recalls.append(np.mean(recall_k))
    
    return precisions, recalls


def run_epoch(data_iter, model, optimizer, epoch, training, device, fp16=False, amp=None, \
    step_scheduler=None, keyword_mask_matrix=None, keywordid2wordid=None, CN_hopk_edge_index=None, use_utterance_concepts=False):
    epoch_loss = []
    precision = []
    recall = []
    print_every = 100000
    for i, batch in tqdm(enumerate(data_iter), total=len(data_iter)):
        batch_X_keywords = torch.LongTensor(batch["batch_X_keywords"]).to(device) # (batch_size, max_kw_context_len)
        batch_y = torch.LongTensor(batch["batch_y"]).to(device) # (batch_size, max_kw_len)
        batch_X_utterances = None
        if len(batch["batch_X_utterances"]) > 0 and model.utterance_encoder_name != "":
            batch_X_utterances = torch.LongTensor(batch["batch_X_utterances"]).to(device) # (batch_size, max_context_len, max_seq_len)
        
        batch_X_concepts = None
        if use_utterance_concepts:
            batch_X_concepts = torch.LongTensor(batch["batch_X_concepts"]).to(device) # (batch_size, max_seq_len)

        if i==0:
            cprint("batch keywords and y shape: ", batch_X_keywords.shape, batch_y.shape)
            if batch_X_utterances is not None:
                cprint("batch_X_utterances shape: ", batch_X_utterances.shape)
            if batch_X_concepts is not None:
                cprint("batch_X_concepts shape: ", batch_X_concepts.shape)
        
        if training:
            optimizer.zero_grad()
        
        logits = model(CN_hopk_edge_index, batch_X_keywords, x_utter=batch_X_utterances, x_concept=batch_X_concepts) # logits: (batch_size, keyword_vocab_size)
        
        if i==0:
            cprint("logits shape: ", logits.shape)

        # keyword vocab mask
        batch_vocab_mask = None
        if keyword_mask_matrix is not None:
            # keyword_mask_matrix[batch_X_keywords]: (batch_size, max_kw_context_len, keyword_vocab_size)
            batch_vocab_mask = keyword_mask_matrix[batch_X_keywords].sum(dim=1).clamp(min=0, max=1) # (batch_size, keyword_vocab_size)

        loss = compute_CE(logits, batch_y, batch_vocab_mask)

        batch_precision, batch_recall = compute_metrics(logits, batch_y, batch_vocab_mask)
        precision.append(batch_precision)
        recall.append(batch_recall)

        # save predictions for case study
        if training:
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            if fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            
            optimizer.step()

            if step_scheduler is not None:
                step_scheduler.step()
        epoch_loss.append(loss.item())

        if i != 0 and i%print_every == 0:
            cprint("loss: ", np.mean(epoch_loss[-print_every:]))
            if not training:
                cprint("valid precision: ", np.mean(precision[-print_every:], axis=0))
                cprint("valid recall: ", np.mean(recall[-print_every:], axis=0))

    loss = np.mean(epoch_loss)
    precision = np.mean(precision, axis=0)
    recall = np.mean(recall, axis=0)
    return loss, (precision.tolist(), recall.tolist())


def main(config, progress):
    # save config
    with open("./log/configs.json", "a") as f:
        json.dump(config, f)
        f.write("\n")
    cprint("*"*80)
    cprint("Experiment progress: {0:.2f}%".format(progress*100))
    cprint("*"*80)
    metrics = {}

    # data hyper-params
    data_path = config["data_path"]
    keyword_path = config["keyword_path"]
    pretrained_wordvec_path = config["pretrained_wordvec_path"]
    data_dir = "/".join(data_path.split("/")[:-1])
    dataset = data_path.split("/")[-2] # convai2 or casual
    test_mode = bool(config["test_mode"])
    save_model_path = config["save_model_path"]
    min_context_len = config["min_context_len"]
    max_context_len = config["max_context_len"]
    max_sent_len = config["max_sent_len"]
    max_keyword_len = config["max_keyword_len"]
    max_vocab_size = config["max_vocab_size"]
    max_keyword_vocab_size = config["max_keyword_vocab_size"]
    remove_self_loop = bool(config["remove_self_loop"])
    
    # model hyper-params
    config_id = config["config_id"]
    model = config["model"]
    gnn = config["gnn"]
    aggregation = config["aggregation"]
    utterance_encoder = config["utterance_encoder"]
    use_last_k_utterances = config["use_last_k_utterances"]
    use_CN_hopk_graph = config["use_CN_hopk_graph"]
    use_utterance_concepts = bool(config["use_utterance_concepts"])
    combine_node_emb = config["combine_node_emb"] # replace, mean, max, concat, 
    concept_encoder = config["concept_encoder"]
    embed_size = config["embed_size"]
    use_pretrained_word_embedding = bool(config["use_pretrained_word_embedding"])
    fix_word_embedding = bool(config["fix_word_embedding"])
    hidden_size = config["hidden_size"]
    n_layers = config["n_layers"]
    bidirectional = bool(config["bidirectional"])
    n_heads = config["n_heads"]
    dropout = config["dropout"]
    
    # training hyper-params
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    lr = config["lr"]
    lr_decay = config["lr_decay"]
    seed = config["seed"]
    device = torch.device(config["device"])
    fp16 = bool(config["fp16"])
    fp16_opt_level = config["fp16_opt_level"]

    # set seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if "convai2" in data_dir and min_context_len != 2:
        raise ValueError("convai2 dataset has min context len of 2")
    if use_pretrained_word_embedding and str(embed_size) not in pretrained_wordvec_path:
        raise ValueError("embedding size and pretrained_wordvec_path not match")

    # load data
    cprint("Loading conversation data...")
    train, valid, test = load_pickle(data_path)
    train_keyword, valid_keyword, test_keyword = load_pickle(keyword_path)
    
    if test_mode:
        cprint("Testing model...")
        train = train + valid
        train_keyword = train_keyword + valid_keyword
        valid = test
        valid_keyword = test_keyword

    cprint(len(train), len(train_keyword), len(valid), len(valid_keyword))
    cprint("sample train: ", train[0])
    cprint("sample train keyword: ", train_keyword[0])
    cprint("sample valid: ", valid[0])
    cprint("sample valid keyword: ", valid_keyword[0])

    # clip and pad data
    train_padded_convs, train_padded_keywords = pad_and_clip_data(train, train_keyword, min_context_len, max_context_len+1, max_sent_len, max_keyword_len)
    valid_padded_convs, valid_padded_keywords = pad_and_clip_data(valid, valid_keyword, min_context_len, max_context_len+1, max_sent_len, max_keyword_len)
    cprint(len(train_padded_convs), len(train_padded_keywords), len(valid_padded_convs), len(valid_padded_keywords))
    cprint("sample padded train: ", train_padded_convs[0])
    cprint("sample padded train keyword: ", train_padded_keywords[0])
    cprint("sample padded valid: ", valid_padded_convs[0])
    cprint("sample padded valid keyword: ", valid_padded_keywords[0])

    # build vocab
    if "convai2" in data_dir:
        test_padded_convs, _ = pad_and_clip_data(test, test_keyword, min_context_len, max_context_len+1, max_sent_len, max_keyword_len)
        word2id = build_vocab(train_padded_convs + valid_padded_convs + test_padded_convs, max_vocab_size) # use entire dataset for vocab as done in (tang 2019)
    else:
        word2id = build_vocab(train_padded_convs, max_vocab_size)
    keyword2id = build_vocab(train_padded_keywords, max_keyword_vocab_size)
    id2keyword = {idx:w for w, idx in keyword2id.items()}
    for w in keyword2id:
        if w not in word2id:
            word2id[w] = len(word2id) # add OOV keywords to word2id
    id2word = {idx:w for w, idx in word2id.items()}
    keywordid2wordid = [word2id[id2keyword[i]] if id2keyword[i] in word2id else word2id["<unk>"] for i in range(len(keyword2id))]
    vocab_size = len(word2id)
    keyword_vocab_size = len(keyword2id)
    cprint("vocab size: ", vocab_size)
    cprint("keyword vocab size: ", keyword_vocab_size)
    
    CN_hopk_edge_index, CN_hopk_nodeid2wordid, keywordid2nodeid, node2id = None, None, None, None
    keyword_mask_matrix = None
    if use_CN_hopk_graph > 0:
        cprint("Loading CN_hopk edge index...")
        """
            CN_graph_dict: {
                edge_index: 2D list (num_edges, 2), 
                edge_type: list (num_edges, ), 
                edge_weight: list (num_edges, ), 
                relation2id: {},
                nodeid2wordid: 2D list (num_nodes, 10)
            }
        """
        CN_hopk_graph_path = "./data/{0}/CN_graph_{1}hop_ge1.pkl".format(dataset, use_CN_hopk_graph)
        cprint("Loading graph from ", CN_hopk_graph_path)
        CN_hopk_graph_dict = load_nx_graph_hopk(CN_hopk_graph_path, word2id, keyword2id)
        CN_hopk_edge_index = torch.LongTensor(CN_hopk_graph_dict["edge_index"]).transpose(0,1).to(device) # (2, num_edges)
        CN_hopk_nodeid2wordid = torch.LongTensor(CN_hopk_graph_dict["nodeid2wordid"]).to(device) # (num_nodes, 10)
        node2id = CN_hopk_graph_dict["node2id"]
        id2node = {idx:w for w,idx in node2id.items()}
        keywordid2nodeid = [node2id[id2keyword[i]] if id2keyword[i] in node2id else node2id["<unk>"] for i in range(len(keyword2id))]
        keywordid2nodeid = torch.LongTensor(keywordid2nodeid).to(device)
        keyword_mask_matrix = torch.from_numpy(CN_hopk_graph_dict["edge_mask"]).float() # numpy array of (keyword_vocab_size, keyword_vocab_size)
        cprint("building keyword mask matrix...")
        if remove_self_loop:
            keyword_mask_matrix[torch.arange(keyword_vocab_size), torch.arange(keyword_vocab_size)] = 0
        cprint("keyword mask matrix non-zeros ratio: ", keyword_mask_matrix.mean())
        cprint("average number of neighbors: ", keyword_mask_matrix.sum(dim=1).mean())
        cprint("sample keyword mask matrix: ", keyword_mask_matrix[:8,:8])
        keyword_mask_matrix = keyword_mask_matrix.to(device)
        
        cprint("edge index shape: ", CN_hopk_edge_index.shape)
        cprint("edge index[:,:8]", CN_hopk_edge_index[:,:8])
        cprint("nodeid2wordid shape: ", CN_hopk_nodeid2wordid.shape)
        cprint("nodeid2wordid[:5,:8]", CN_hopk_nodeid2wordid[:5,:8])
        cprint("keywordid2nodeid shape: ", keywordid2nodeid.shape)
        cprint("keywordid2nodeid[:8]", keywordid2nodeid[:8])

    # convert edge index
    if utterance_encoder != "":
        keywordid2wordid = torch.LongTensor(keywordid2wordid).to(device)
        cprint("keywordid2wordid shape: ", keywordid2wordid.shape)
        cprint("keywordid2wordid", keywordid2wordid[:8])

    # convert tokens to ids
    train_conv_ids = convert_convs_to_ids(train_padded_convs, word2id)
    valid_conv_ids = convert_convs_to_ids(valid_padded_convs, word2id)
    train_keyword_ids = convert_convs_to_ids(train_padded_keywords, keyword2id)
    valid_keyword_ids = convert_convs_to_ids(valid_padded_keywords, keyword2id)
    cprint(len(train_conv_ids), len(train_keyword_ids), len(valid_conv_ids), len(valid_keyword_ids))

    cprint("sample train token ids: ", train_conv_ids[0])
    cprint("sample train keyword ids: ", train_keyword_ids[0])
    cprint("sample valid token ids: ", valid_conv_ids[0])
    cprint("sample valid keyword ids: ", valid_keyword_ids[0])
    num_examples = len(train_keyword_ids)

    # create model
    if model in ["KW_GNN"]:
        model_kwargs = {
            "embed_size": embed_size,
            "vocab_size": vocab_size,
            "keyword_vocab_size": keyword_vocab_size,
            "hidden_size": hidden_size,
            "output_size": hidden_size,
            "n_layers": n_layers,
            "gnn": gnn,
            "aggregation": aggregation,
            "n_heads": n_heads,
            "dropout": dropout,
            "bidirectional": bidirectional,
            "utterance_encoder": utterance_encoder,
            "keywordid2wordid": keywordid2wordid,
            "keyword_mask_matrix": keyword_mask_matrix,
            "nodeid2wordid": CN_hopk_nodeid2wordid,
            "keywordid2nodeid": keywordid2nodeid,
            "concept_encoder": concept_encoder,
            "combine_node_emb": combine_node_emb
        }

    cprint("Building model...")
    model = globals()[config["model"]](**model_kwargs)
    # cprint(model.edge_weight.shape, model.edge_weight.requires_grad)
    
    pretrained_word_embedding = None
    if use_pretrained_word_embedding:
        # load pretrained word embedding
        cprint("Loading pretrained word embeddings...")
        pretrained_wordvec_name = pretrained_wordvec_path.split("/")[-1][:-4]
        word_vectors_path = os.path.join(data_dir, "word_vectors_{0}.pkl".format(pretrained_wordvec_name))
        keyword2id = word2id

        if os.path.exists(word_vectors_path):
            cprint("Loading pretrained word embeddings from ", word_vectors_path)
            with open(word_vectors_path, "rb") as f:
                word_vectors = pickle.load(f)
        else:
            cprint("Loading pretrained word embeddings from scratch...")
            word_vectors = load_vectors(pretrained_wordvec_path, keyword2id)
            cprint("Saving pretrained word embeddings to ", word_vectors_path)
            with open(word_vectors_path, "wb") as f:
                pickle.dump(word_vectors, f)

        print("loaded word vector size: ", len(word_vectors))
        pretrained_word_embedding = np.zeros((len(keyword2id), embed_size))
        for w, i in keyword2id.items():
            if w in word_vectors:
                pretrained_word_embedding[i] = np.array(word_vectors[w])
            else:
                pretrained_word_embedding[i] = np.random.randn(embed_size)/9
            
        pretrained_word_embedding[0] = 0 # 0 for PAD embedding
        pretrained_word_embedding = torch.from_numpy(pretrained_word_embedding).float()
        cprint("word embedding size: ", pretrained_word_embedding.shape)
        
        model.init_embedding(pretrained_word_embedding, fix_word_embedding)
    
    cprint(model)
    cprint("number of parameters: ", count_parameters(model))
    model.to(device)

    # optimization
    amp = None
    if fp16:
        from apex import amp
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: lr_decay ** epoch)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1/(1+lr_decay*step/(num_examples/batch_size)))
    if fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    # training
    epoch_train_losses = []
    epoch_valid_losses = []
    epoch_valid_precisions = []
    epoch_valid_recalls = []
    best_model_statedict = {}
    cprint("Start training...")
    for epoch in range(epochs):
        cprint("-"*80)
        cprint("Epoch", epoch+1)
        train_batches = create_batches_keyword_prediction(train_conv_ids, train_keyword_ids, 2*max_keyword_len, batch_size, \
            shuffle=True, remove_self_loop=remove_self_loop, keywordid2wordid=keywordid2wordid, \
                    keyword_mask_matrix=keyword_mask_matrix.cpu().numpy(), use_last_k_utterances=use_last_k_utterances, use_utterance_concepts=use_utterance_concepts, \
                        keyword2id=keyword2id, node2id=node2id, id2word=id2word)
        valid_batches = create_batches_keyword_prediction(valid_conv_ids, valid_keyword_ids, 2*max_keyword_len, batch_size, \
            shuffle=False, remove_self_loop=remove_self_loop, keywordid2wordid=keywordid2wordid, \
                    keyword_mask_matrix=keyword_mask_matrix.cpu().numpy(), use_last_k_utterances=use_last_k_utterances, use_utterance_concepts=use_utterance_concepts, \
                        keyword2id=keyword2id, node2id=node2id, id2word=id2word)

        cprint("train batches 1st example: ")
        for k, v in train_batches[0].items():
            if k == "batch_X_keywords":
                cprint(k, v[0], [id2keyword[w] for w in v[0]])
            if k == "batch_X_utterances":
                utters = []
                for utter in v[0]:
                    utters.append([id2word[w] for w in utter])
                cprint(k, v[0], utters)
            if k == "batch_X_concepts" and len(v) > 0:
                cprint(k, v[0], [id2node[w] for w in v[0]])
            if k == "batch_y":
                cprint(k, v[0], [id2keyword[w] for w in v[0]])
        
        model.train()
        train_loss, (train_precision, train_recall) = run_epoch(train_batches, model, optimizer, epoch=epoch, training=True, device=device, \
            fp16=fp16, amp=amp, step_scheduler=scheduler, keyword_mask_matrix=keyword_mask_matrix, keywordid2wordid=keywordid2wordid, \
                CN_hopk_edge_index=CN_hopk_edge_index, use_utterance_concepts=use_utterance_concepts)
        cprint("Config id: {}, Epoch {}: train precision: {}, train recall: {}"
            .format(config_id, epoch+1, train_precision, train_recall))
        
        model.eval()
        valid_loss, (valid_precision, valid_recall) = run_epoch(valid_batches, model, optimizer, epoch=epoch, training=False, device=device, \
            keyword_mask_matrix=keyword_mask_matrix, keywordid2wordid=keywordid2wordid, \
                            CN_hopk_edge_index=CN_hopk_edge_index, use_utterance_concepts=use_utterance_concepts)
        
        # scheduler.step()
        cprint("Config id: {}, Epoch {}: train loss: {}, valid loss: {}, valid precision: {}, valid recall: {}"
            .format(config_id, epoch+1, train_loss, valid_loss, valid_precision, valid_recall))
        if scheduler is not None:
            cprint("Current learning rate: ", scheduler.get_last_lr())
        
        epoch_train_losses.append(train_loss)
        epoch_valid_losses.append(valid_loss)
        epoch_valid_precisions.append(valid_precision)
        epoch_valid_recalls.append(valid_recall)

        if save_model_path != "":
            if epoch == 0:
                for k, v in model.state_dict().items():
                    best_model_statedict[k] = v.cpu()
            else:
                if epoch_valid_recalls[-1][0] == max([recall1 for recall1, _, _ in epoch_valid_recalls]):
                    for k, v in model.state_dict().items():
                        best_model_statedict[k] = v.cpu()

        # early stopping
        if len(epoch_valid_recalls) >= 3 and epoch_valid_recalls[-1][0] < epoch_valid_recalls[-2][0] and epoch_valid_recalls[-2][0] < epoch_valid_recalls[-3][0]:
            break

    config.pop("seed")
    config.pop("config_id")
    metrics["config"] = config
    metrics["score"] = max([recall[0] for recall in epoch_valid_recalls])
    metrics["epoch"] = np.argmax([recall[0] for recall in epoch_valid_recalls]).item()
    metrics["recall"] = epoch_valid_recalls[metrics["epoch"]]
    metrics["precision"] = epoch_valid_precisions[metrics["epoch"]]

    if save_model_path:
        cprint("Saving model to ", save_model_path)
        best_model_statedict["word2id"] = keyword2id
        best_model_statedict["model_kwargs"] = model_kwargs
        torch.save(best_model_statedict, save_model_path)
    
    return metrics


def clean_config(configs):
    cleaned_configs = []
    for config in configs:
        if config not in cleaned_configs:
            cleaned_configs.append(config)
    return cleaned_configs


def merge_metrics(metrics):
    avg_metrics = {"score" : 0}
    std_metrics = {}

    num_metrics = len(metrics)
    for metric in metrics:
        for k in metric:
            if isinstance(metric[k], list):
                if k in avg_metrics:
                    avg_metrics[k] += np.array(metric[k])
                else:
                    avg_metrics[k] = np.array(metric[k])
            elif k == "score":
                avg_metrics[k] += metric[k]
                
            if k == "config" or k == "epoch":
                continue
            if k in std_metrics:
                std_metrics[k].append(metric[k])
            else:
                std_metrics[k] = [metric[k]]
    
    for k, v in avg_metrics.items():
        if k == "score":
            avg_metrics[k] = v/num_metrics
        else:
            avg_metrics[k] = (v/num_metrics).tolist()
    
    for k,v in std_metrics.items():
        std_metrics[k] = np.array(v).std(axis=0).tolist()

    return avg_metrics, std_metrics


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Model for Keyword Prediction")
    parser.add_argument('--config', help='Config to read details', required=True)
    parser.add_argument('--note', help='Experiment note', default="")
    args = parser.parse_args()
    cprint("Experiment note: ", args.note)
    with open(args.config) as configfile:
        config = json.load(configfile) # config is now a python dict
    
    # pass experiment config to main
    parameters_to_search = OrderedDict() # keep keys in order
    other_parameters = {}
    keys_to_omit = ["kernel_sizes"] # keys that allow a list of values
    for k, v in config.items():
        # if value is a list provided that key is not device, or kernel_sizes is a nested list
        if isinstance(v, list) and k not in keys_to_omit:
            parameters_to_search[k] = v
        elif k in keys_to_omit and isinstance(config[k], list) and isinstance(config[k][0], list):
            parameters_to_search[k] = v
        else:
            other_parameters[k] = v

    if len(parameters_to_search) == 0:
        config_id = time.perf_counter()
        config["config_id"] = config_id
        print(config)
        output = main(config, progress=1)
        print("-"*80)
        print(output["config"])
        print("Best epoch: ", output["epoch"])
        print("Best score: ", output["score"])
        print("Best recall: ", output["recall"])
        print("Best precision: ", output["precision"])
    else:
        all_configs = []
        for i, r in enumerate(itertools.product(*parameters_to_search.values())):
            specific_config = {}
            for idx, k in enumerate(parameters_to_search.keys()):
                specific_config[k] = r[idx]
            
            # merge with other parameters
            merged_config = {**other_parameters, **specific_config}
            all_configs.append(merged_config)
        
        # cprint all configs
        for config in all_configs:
            config_id = time.perf_counter()
            config["config_id"] = config_id
            logging.critical("config id: {0}".format(config_id))
            print(config)
            print("\n")

        # multiprocessing
        num_configs = len(all_configs)
        # mp.set_start_method('spawn')
        pool = mp.Pool(processes=config["processes"])
        results = [pool.apply_async(main, args=(x,i/num_configs)) for i,x in enumerate(all_configs)]
        outputs = [p.get() for p in results]

        # if run multiple models using different seed and get the averaged result
        if "seed" in parameters_to_search:
            all_metrics = []
            all_cleaned_configs = clean_config([output["config"] for output in outputs])
            for config in all_cleaned_configs:
                metrics_per_config = []
                for output in outputs:
                    if output["config"] == config:
                        metrics_per_config.append(output)
                avg_metrics, std_metrics = merge_metrics(metrics_per_config)
                all_metrics.append((config, avg_metrics, std_metrics))
            # log metrics
            print("Average evaluation result across different seeds: ")
            for config, metric, std_metric in all_metrics:
                cprint("-"*80)
                cprint(config)
                cprint(metric)
                cprint(std_metric)

            # save to log
            with open("./log/{0}.txt".format(time.perf_counter()), "a+") as f:
                for config, metric, std_metric in all_metrics:
                    f.write(json.dumps("-"*80) + "\n")
                    f.write(json.dumps(config) + "\n")
                    f.write(json.dumps(metric) + "\n")
                    f.write(json.dumps(std_metric) + "\n")

        else:
            for output in outputs:
                print("-"*80)
                print(output["config"])
                print("Best epoch: ", output["epoch"])
                print("Best score: ", output["score"])
                print("Best recall: ", output["recall"])
                print("Best precision: ", output["precision"])

            # save to log
            with open("./log/{0}.txt".format(time.perf_counter()), "a+") as f:
                for output in outputs:
                    f.write(json.dumps("-"*80) + "\n")
                    f.write(json.dumps(output) + "\n")
