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
import json
from collections import Counter, OrderedDict, defaultdict
import functools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from model import KW_GNN, CoGraphMatcher
from util.io import load_pickle, save_pickle, load_vectors, load_nx_graph_hopk
from util.tool import count_parameters, convert_ids_to_sent
from util.data import pad_and_clip_data, build_vocab, convert_convs_to_ids, \
    pad_and_clip_candidate, convert_candidates_to_ids, create_batches_retrieval, extract_keywords_from_candidates

logging.basicConfig(level = logging.INFO, \
                    format = '%(asctime)s  %(levelname)-5s %(message)s', \
                    datefmt =  "%Y-%m-%d-%H-%M-%S")

def cprint(*args):
    text = ""
    for arg in args:
        text += "{0} ".format(arg)
    logging.info(text)

def compute_CE(logits):
    """
        CE loss
        logits: (batch, num_candidates)
    """
    logits = F.log_softmax(logits, dim=-1)
    loss = -1 * (logits[:,0]).mean() # negative log-likelihood loss
    loss = loss/logits.shape[0] # average over batch size
    return loss

def compute_metrics(logits):
    """
        logits: (batch, num_candidates)
    """
    # logits = torch.rand_like(logits) # random baseline
    batch_size, num_candidates = logits.shape
    
    # recall@k
    sorted_indices = logits.sort(descending=True)[1] # (batch, num_candidates)
    targets = [[0]]*batch_size
    
    precisions = []
    recalls = []
    ks = [1, 3, 5]
    for k in ks:
        # sorted_indices[:,:k]: (batch_size, k)
        precision_k = []
        recall_k = []
        for tgts, topk in zip(targets, sorted_indices[:,:k].tolist()):
            num_hit = len(set(topk).intersection(set(tgts)))
            precision_k.append(num_hit/len(topk))
            recall_k.append(num_hit/len(tgts))
        precisions.append(np.mean(precision_k))
        recalls.append(np.mean(recall_k))

    MRR = 0
    for tgts, topk in zip(targets, sorted_indices.tolist()):
        rank = topk.index(tgts[0])+1
        MRR += 1/rank
    MRR = MRR/batch_size
    
    return precisions, recalls, MRR

def run_epoch(data_iter, model, optimizer, training, device, fp16=False, amp=None, kw_model=None, keyword_mask_matrix=None, \
    step_scheduler=None, keywordid2wordid=None, CN_hopk_edge_index=None):
    epoch_loss = []
    precision = []
    recall = []
    MRR = []
    print_every = 1000
    for i, batch in tqdm(enumerate(data_iter), total=len(data_iter)):
        batch_context = torch.LongTensor(batch["batch_context"]).to(device) # (batch_size, max_context_len, max_sent_len)
        batch_candidate = torch.LongTensor(batch["batch_candidates"]).to(device) # (batch_size, num_candidates, max_sent_len)
        if i==0:
            cprint("batch_context: ", batch_context.shape)
            cprint("batch_candidate: ", batch_candidate.shape)
        batch_context_kw, batch_candidate_kw = None, None
        batch_context_concepts, batch_candidate_concepts = None, None
        batch_context_for_keywords_prediction, batch_context_concepts_for_keywords_prediction = None, None
        if "batch_context_kw" in batch:
            # keyword ids
            batch_context_kw = torch.LongTensor(batch["batch_context_kw"]).to(device) # (batch_size, max_kw_context_len)
            batch_candidate_kw = torch.LongTensor(batch["batch_candidates_kw"]).to(device) # (batch_size, num_candidates, max_kw_seq_len)
            if i==0:
                cprint("batch_context_kw: ", batch_context_kw.shape)
                cprint("batch_candidate_kw: ", batch_candidate_kw.shape)
        
        if "batch_context_concepts" in batch:
            # node ids
            batch_context_concepts = torch.LongTensor(batch["batch_context_concepts"]).to(device) # (batch_size, max_context_len, max_sent_len)
            batch_candidate_concepts = torch.LongTensor(batch["batch_candidates_concepts"]).to(device) # (batch_size, max_context_len, max_sent_len)
            if i==0:
                cprint("batch_context_concepts: ", batch_context_concepts.shape)
                cprint("batch_candidate_concepts: ", batch_candidate_concepts.shape)

        if "batch_context_for_keywords_prediction" in batch:
            batch_context_for_keywords_prediction = torch.LongTensor(batch["batch_context_for_keywords_prediction"])\
                .to(device) # (batch_size, last_k_utterances, max_sent_len)
            batch_context_concepts_for_keywords_prediction = torch.LongTensor(batch["batch_context_concepts_for_keywords_prediction"])\
                .to(device) # (batch_size, last_k_utterances, max_sent_len)
            if i==0:
                cprint("batch_context_for_keywords_prediction: ", batch_context_for_keywords_prediction.shape)
                cprint("batch_context_concepts_for_keywords_prediction: ", batch_context_concepts_for_keywords_prediction.shape)

        top_kws = None
        if kw_model:
            # use predicted keywords to validate keyword-augmented retrieval
            with torch.no_grad():
                if isinstance(kw_model, KW_GNN):
                    kw_logits, _ = kw_model(None, None, CN_hopk_edge_index, batch_context_kw, x_utter=batch_context_for_keywords_prediction, \
                        x_concept=batch_context_concepts_for_keywords_prediction) # (batch_size, keyword_vocab_size)
                else:
                    kw_logits = kw_model(batch_context_kw) # (batch_size, keyword_vocab_size)
                if keyword_mask_matrix is not None:
                    batch_vocab_mask = keyword_mask_matrix[batch_context_kw].sum(dim=1).clamp(min=0, max=1) # (batch_size, keyword_vocab_size)
                    kw_logits = (1-batch_vocab_mask)*(-5e4) + batch_vocab_mask*kw_logits # (batch, vocab_size), masked logits
                top_kws = kw_logits.topk(3, dim=-1)[1] # (batch_size, 3), need to convert to vocab token id based on word2id
        
        if training:
            optimizer.zero_grad()
        logits = model(batch_context, batch_candidate, top_kws, batch_candidate_kw, batch_context_concepts, batch_candidate_concepts, \
            CN_hopk_edge_index) # logits: (batch_size, num_candidates)

        # compute loss
        loss = compute_CE(logits)

        # compute valid recall
        if not training:
            valid_precision, valid_recall, valid_MRR = compute_metrics(logits)
            precision.append(valid_precision)
            recall.append(valid_recall)
            MRR.append(valid_MRR)

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

        if i!=0 and i%print_every == 0:
            cprint("loss: ", np.mean(epoch_loss[-print_every:]))
            if not training:
                cprint("valid precision: ", np.mean(precision[-print_every:], axis=0))
                cprint("valid recall: ", np.mean(recall[-print_every:], axis=0))
                cprint("valid MRR: ", np.mean(MRR[-print_every:], axis=0))
    
    loss = np.mean(epoch_loss)
    if training:
        return loss, (None, None, None)
    else:
        precision = np.mean(precision, axis=0)
        recall = np.mean(recall, axis=0)
        MRR = np.mean(MRR, axis=0)
        return loss, (precision.tolist(), recall.tolist(), MRR)


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
    load_kw_prediction_path = config["load_kw_prediction_path"]
    min_context_len = config["min_context_len"]
    max_context_len = config["max_context_len"]
    max_sent_len = config["max_sent_len"]
    max_keyword_len = config["max_keyword_len"]
    max_vocab_size = config["max_vocab_size"]
    max_keyword_vocab_size = config["max_keyword_vocab_size"]
    flatten_context = config["flatten_context"]
    
    # model hyper-params
    config_id = config["config_id"]
    model = config["model"]
    use_CN_hopk_graph = config["use_CN_hopk_graph"]
    use_utterance_concepts = use_CN_hopk_graph > 0
    concept_encoder = config["concept_encoder"]
    combine_word_concepts = config["combine_word_concepts"]
    gnn = config["gnn"]
    encoder = config["encoder"]
    aggregation = config["aggregation"]
    use_keywords = bool(config["use_keywords"])
    keyword_score_weight = config["keyword_score_weight"]
    keyword_encoder = config["keyword_encoder"] # mean, max, GRU, any_max
    embed_size = config["embed_size"]
    use_pretrained_word_embedding = bool(config["use_pretrained_word_embedding"])
    fix_word_embedding = bool(config["fix_word_embedding"])
    gnn_hidden_size = config["gnn_hidden_size"]
    gnn_layers = config["gnn_layers"]
    encoder_hidden_size = config["encoder_hidden_size"]
    encoder_layers = config["encoder_layers"]
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
    if use_keywords and load_kw_prediction_path == "":
        raise ValueError("kw model path needs to be provided when use_keywords is True")
    
    # load data
    cprint("Loading conversation data...")
    train, valid, test = load_pickle(data_path)
    train_keyword, valid_keyword, test_keyword = load_pickle(keyword_path)
    train_candidate, valid_candidate = None, None
    # load 20 candidates
    train_candidate, valid_candidate, test_candidate = load_pickle(os.path.join(data_dir, "candidate.pkl"))

    if test_mode:
        cprint("Testing model...")
        train = train + valid
        train_keyword = train_keyword + valid_keyword
        valid = test
        valid_keyword = test_keyword
        train_candidate = train_candidate + valid_candidate
        valid_candidate = test_candidate

    cprint("sample train: ", train[0])
    cprint("sample train keyword: ", train_keyword[0])
    cprint("sample valid: ", valid[0])
    cprint("sample valid keyword: ", valid_keyword[0])

    # clip and pad data
    train_padded_convs, train_padded_keywords = pad_and_clip_data(train, train_keyword, min_context_len, max_context_len+1, max_sent_len, max_keyword_len)
    valid_padded_convs, valid_padded_keywords = pad_and_clip_data(valid, valid_keyword, min_context_len, max_context_len+1, max_sent_len, max_keyword_len)
    train_padded_candidates = pad_and_clip_candidate(train_candidate, max_sent_len)
    valid_padded_candidates = pad_and_clip_candidate(valid_candidate, max_sent_len)

    # build vocab
    if "convai2" in data_dir:
        test_padded_convs, _ = pad_and_clip_data(test, test_keyword, min_context_len, max_context_len+1, max_sent_len, max_keyword_len)
        word2id = build_vocab(train_padded_convs + valid_padded_convs + test_padded_convs, max_vocab_size) # use entire dataset for vocab
    else:
        word2id = build_vocab(train_padded_convs, max_vocab_size)
    keyword2id = build_vocab(train_padded_keywords, max_keyword_vocab_size)
    id2keyword = {idx:w for w, idx in keyword2id.items()}
    for w in keyword2id:
        if w not in word2id:
            word2id[w] = len(word2id) # add OOV keywords to word2id
    id2word = {idx:w for w, idx in word2id.items()}
    cprint("keywords that are not in word2id: ", set(keyword2id.keys()) - set(word2id.keys()))
    vocab_size = len(word2id)
    keyword_vocab_size = len(keyword2id)
    cprint("vocab size: ", vocab_size)
    cprint("keyword vocab size: ", keyword_vocab_size)

    # create a mapping from keyword id to word id
    keywordid2wordid = None
    train_candidate_keyword_ids, valid_candidate_keyword_ids = None, None
    if use_keywords:
        keywordid2wordid = [word2id[id2keyword[i]] if id2keyword[i] in word2id else word2id["<unk>"] for i in range(len(keyword2id))]
        keywordid2wordid = torch.LongTensor(keywordid2wordid).to(device)

        # load candidate keywords
        candidate_keyword_path = os.path.join(data_dir, "candidate_keyword.pkl")
        if os.path.exists(candidate_keyword_path):
            cprint("Loading candidate keywords from ", candidate_keyword_path)
            train_candidate_keywords, valid_candidate_keywords, test_candidate_keywords = load_pickle(candidate_keyword_path)
        else:
            cprint("Creating candidate keywords...")
            train_candidate_keywords = extract_keywords_from_candidates(train_candidate, keyword2id)
            valid_candidate_keywords = extract_keywords_from_candidates(valid_candidate, keyword2id)
            test_candidate_keywords = extract_keywords_from_candidates(test_candidate, keyword2id)
            save_pickle((train_candidate_keywords, valid_candidate_keywords, test_candidate_keywords), candidate_keyword_path)

        if test_mode:
            train_candidate_keywords = train_candidate_keywords + valid_candidate_keywords
            valid_candidate_keywords = test_candidate_keywords
        
        # pad
        cprint("Padding candidate keywords...")
        train_padded_candidate_keywords = pad_and_clip_candidate(train_candidate_keywords, max_keyword_len)
        valid_padded_candidate_keywords = pad_and_clip_candidate(valid_candidate_keywords, max_keyword_len)

        # convert candidates to ids
        cprint("Converting candidate keywords to ids...")
        train_candidate_keyword_ids = convert_candidates_to_ids(train_padded_candidate_keywords, keyword2id)
        valid_candidate_keyword_ids = convert_candidates_to_ids(valid_padded_candidate_keywords, keyword2id)

    # load CN graph
    CN_hopk_edge_index, CN_hopk_nodeid2wordid, keywordid2nodeid, node2id, CN_hopk_edge_matrix_mask = None, None, None, None, None
    if use_CN_hopk_graph > 0:
        cprint("Loading CN_hopk edge index...")
        """
            CN_graph_dict: {
                edge_index: 2D list (num_edges, 2), 
                edge_weight: list (num_edges, ), 
                nodeid2wordid: 2D list (num_nodes, 10),
                edge_mask: numpy array of (keyword_vocab_size, keyword_vocab_size)
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
        
        cprint("edge index shape: ", CN_hopk_edge_index.shape)
        cprint("edge index[:,:8]", CN_hopk_edge_index[:,:8])
        cprint("nodeid2wordid shape: ", CN_hopk_nodeid2wordid.shape)
        cprint("nodeid2wordid[:5,:8]", CN_hopk_nodeid2wordid[:5,:8])
        cprint("keywordid2nodeid shape: ", keywordid2nodeid.shape)
        cprint("keywordid2nodeid[:8]", keywordid2nodeid[:8])

    # convert tokens to ids
    train_conv_ids = convert_convs_to_ids(train_padded_convs, word2id)
    valid_conv_ids = convert_convs_to_ids(valid_padded_convs, word2id)
    train_keyword_ids = convert_convs_to_ids(train_padded_keywords, keyword2id)
    valid_keyword_ids = convert_convs_to_ids(valid_padded_keywords, keyword2id)
    train_candidate_ids, valid_candidate_ids = None, None
    train_candidate_ids = convert_candidates_to_ids(train_padded_candidates, word2id)
    valid_candidate_ids = convert_candidates_to_ids(valid_padded_candidates, word2id)
    
    keyword_mask_matrix = None
    if use_CN_hopk_graph > 0:
        keyword_mask_matrix = torch.from_numpy(CN_hopk_graph_dict["edge_mask"]).float() # numpy array of (keyword_vocab_size, keyword_vocab_size)
        cprint("building keyword mask matrix...")
        keyword_mask_matrix[torch.arange(keyword_vocab_size), torch.arange(keyword_vocab_size)] = 0 # remove self loop
        cprint("keyword mask matrix non-zeros ratio: ", keyword_mask_matrix.mean())
        cprint("average number of neighbors: ", keyword_mask_matrix.sum(dim=1).mean())
        cprint("sample keyword mask matrix: ", keyword_mask_matrix[:8,:8])
        keyword_mask_matrix = keyword_mask_matrix.to(device)
    
    num_examples = len(train_conv_ids)
    cprint("sample train token ids: ", train_conv_ids[0])
    cprint("sample train keyword ids: ", train_keyword_ids[0])
    cprint("sample valid token ids: ", valid_conv_ids[0])
    cprint("sample valid keyword ids: ", valid_keyword_ids[0])
    cprint("sample train candidate ids: ", train_candidate_ids[0])
    cprint("sample valid candidate ids: ", valid_candidate_ids[0])
    if use_keywords:
        cprint("sample train candidate keyword ids: ", train_candidate_keyword_ids[0])
        cprint("sample valid candidate keyword ids: ", valid_candidate_keyword_ids[0])

    # create model
    if model in ["CoGraphMatcher"]:
        model_kwargs = {
            "embed_size": embed_size,
            "vocab_size": vocab_size,
            "gnn_hidden_size": gnn_hidden_size,
            "gnn_layers": gnn_layers,
            "encoder_hidden_size": encoder_hidden_size,
            "encoder_layers": encoder_layers,
            "n_heads": n_heads,
            "CN_hopk_edge_matrix_mask": CN_hopk_edge_matrix_mask, 
            "nodeid2wordid": CN_hopk_nodeid2wordid, 
            "keywordid2wordid": keywordid2wordid,
            "keywordid2nodeid": keywordid2nodeid,
            "concept_encoder": concept_encoder,
            "gnn": gnn,
            "encoder": encoder,
            "aggregation": aggregation,
            "use_keywords": use_keywords,
            "keyword_score_weight": keyword_score_weight,
            "keyword_encoder": keyword_encoder,
            "dropout": dropout,
            "combine_word_concepts": combine_word_concepts
        }

    # create keyword model
    kw_model = ""
    use_last_k_utterances = -1
    if use_keywords:
        kw_model = load_kw_prediction_path.split("/")[-1][:-3] # keyword prediction model name
        if "GNN" in kw_model:
            kw_model = "KW_GNN"
            use_last_k_utterances = 2

        # load pretrained model
        cprint("Loading weights from ", load_kw_prediction_path)
        kw_model_checkpoint = torch.load(load_kw_prediction_path, map_location=device)
        if "word2id" in kw_model_checkpoint:
            keyword2id = kw_model_checkpoint.pop("word2id")
        if "model_kwargs" in kw_model_checkpoint:
            kw_model_kwargs = kw_model_checkpoint.pop("model_kwargs")
            kw_model = globals()[kw_model](**kw_model_kwargs)
        kw_model.load_state_dict(kw_model_checkpoint)
        kw_model.to(device)
        kw_model.eval() # set to evaluation mode, no training required

    cprint("Building model...")
    model = globals()[config["model"]](**model_kwargs)
    
    cprint("Initializing pretrained word embeddings...")
    pretrained_word_embedding = None
    if use_pretrained_word_embedding:
        # load pretrained word embedding
        cprint("Loading pretrained word embeddings...")
        pretrained_wordvec_name = pretrained_wordvec_path.split("/")[-1][:-4]
        word_vectors_path = os.path.join(data_dir, "word_vectors_{0}.pkl".format(pretrained_wordvec_name))
        if os.path.exists(word_vectors_path):
            cprint("Loading pretrained word embeddings from ", word_vectors_path)
            with open(word_vectors_path, "rb") as f:
                word_vectors = pickle.load(f)
        else:
            cprint("Loading pretrained word embeddings from scratch...")
            word_vectors = load_vectors(pretrained_wordvec_path, word2id)
            cprint("Saving pretrained word embeddings to ", word_vectors_path)
            with open(word_vectors_path, "wb") as f:
                pickle.dump(word_vectors, f)
        
        cprint("pretrained word embedding size: ", len(word_vectors))
        pretrained_word_embedding = np.zeros((len(word2id), embed_size))
        for w, i in word2id.items():
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
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1/(1+lr_decay*step/(num_examples/batch_size)))
    
    if fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    # training
    epoch_train_losses = []
    epoch_valid_losses = []
    epoch_valid_precisions = []
    epoch_valid_recalls = []
    epoch_valid_MRRs = []
    best_model_statedict = {}
    cprint("Start training...")
    for epoch in range(epochs):
        cprint("-"*80)
        cprint("Epoch", epoch+1)
        train_batches = create_batches_retrieval(train_conv_ids, train_keyword_ids, train_candidate_ids, train_candidate_keyword_ids, \
            2*max_keyword_len, batch_size, shuffle=True, use_keywords=use_keywords, use_candidate_keywords=use_keywords, use_utterance_concepts=use_utterance_concepts, \
                node2id=node2id, id2word=id2word, flatten_context=flatten_context, use_last_k_utterances=use_last_k_utterances)
        valid_batches = create_batches_retrieval(valid_conv_ids, valid_keyword_ids, valid_candidate_ids, valid_candidate_keyword_ids, \
            2*max_keyword_len, batch_size, shuffle=False, use_keywords=use_keywords, use_candidate_keywords=use_keywords, use_utterance_concepts=use_utterance_concepts, \
                node2id=node2id, id2word=id2word, flatten_context=flatten_context, use_last_k_utterances=use_last_k_utterances)

        if epoch == 0:
            cprint("number of optimization steps per epoch: ", len(train_batches)) # 3361
            cprint("train batches 1st example: ")
            for k, v in train_batches[0].items():
                if k == "batch_context":
                    utters = []
                    for utter in v[0]:
                        utters.append([id2word[w] for w in utter])
                    cprint("\n", k, v[0], utters)
                if k == "batch_candidates":
                    utters = []
                    for utter in v[0]:
                        utters.append([id2word[w] for w in utter])
                    cprint("\n", k, v[0], utters)
                if k == "batch_context_kw":
                    cprint("\n", k, v[0], [id2keyword[w] for w in v[0]])
                if k == "batch_candidates_kw":
                    utters = []
                    for utter in v[0]:
                        utters.append([id2keyword[w] for w in utter])
                    cprint("\n", k, v[0], utters)
                if k == "batch_context_concepts":
                    if len(v[0][0])>0:
                        utters = []
                        for utter in v[0]:
                            utters.append([id2node[w] for w in utter])
                        cprint("\n", k, v[0], utters)
                if k == "batch_candidates_concepts":
                    utters = []
                    for utter in v[0]:
                        utters.append([id2node[w] for w in utter])
                    cprint("\n", k, v[0], utters)
                if k == "batch_context_for_keyword_prediction":
                    utters = []
                    for utter in v[0]:
                        utters.append([id2word[w] for w in utter])
                    cprint("\n", k, v[0], utters)
                if k == "batch_context_concepts_for_keyword_prediction":
                    cprint("\n", k, v[0], [id2node[w] for w in v[0]])
        
        model.train()
        train_loss, (_, _, _) = run_epoch(train_batches, model, optimizer, training=True, device=device, fp16=fp16, amp=amp, \
            kw_model=kw_model, keyword_mask_matrix=keyword_mask_matrix, step_scheduler=scheduler, keywordid2wordid=keywordid2wordid, \
                CN_hopk_edge_index=CN_hopk_edge_index)
        
        model.eval()
        valid_loss, (valid_precision, valid_recall, valid_MRR) = run_epoch(valid_batches, model, optimizer, training=False, device=device, \
            kw_model=kw_model, keyword_mask_matrix=keyword_mask_matrix, keywordid2wordid=keywordid2wordid, CN_hopk_edge_index=CN_hopk_edge_index)
        
        # scheduler.step()
        cprint("Config id: {0}, Epoch {1}: train loss: {2:.4f}, valid loss: {3:.4f}, valid precision: {4}, valid recall: {5}, valid MRR: {6}"
            .format(config_id, epoch+1, train_loss, valid_loss, valid_precision, valid_recall, valid_MRR))
        if scheduler is not None:
            cprint("Current learning rate: ", scheduler.get_last_lr())
        epoch_train_losses.append(train_loss)
        epoch_valid_losses.append(valid_loss)
        epoch_valid_precisions.append(valid_precision)
        epoch_valid_recalls.append(valid_recall)
        epoch_valid_MRRs.append(valid_MRR)

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
    metrics["MRR"] = epoch_valid_MRRs[metrics["epoch"]]
    metrics["precision"] = epoch_valid_precisions[metrics["epoch"]]
    
    if save_model_path and seed == 1:
        cprint("Saving model to ", save_model_path)
        best_model_statedict["word2id"] = word2id
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
    avg_metrics = {"score" : 0, "MRR": 0}
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
            elif k == "MRR":
                avg_metrics[k] += metric[k]
                
            if k == "config" or k == "epoch":
                continue
            if k in std_metrics:
                std_metrics[k].append(metric[k])
            else:
                std_metrics[k] = [metric[k]]
    
    for k, v in avg_metrics.items():
        if k == "score" or k == "MRR":
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
        cprint(config)
        output = main(config, progress=1)
        cprint("-"*80)
        print(output["config"])
        print("Best epoch: ", output["epoch"])
        print("Best score: ", output["score"])
        print("Best recall: ", output["recall"])
        print("Best precision: ", output["precision"])
        print("Best MRR: ", output["MRR"])
    else:
        all_configs = []
        for i, r in enumerate(itertools.product(*parameters_to_search.values())):
            specific_config = {}
            for idx, k in enumerate(parameters_to_search.keys()):
                specific_config[k] = r[idx]
            
            # merge with other parameters
            merged_config = {**other_parameters, **specific_config}
            all_configs.append(merged_config)
        
        #   cprint all configs
        for config in all_configs:
            config_id = time.perf_counter()
            config["config_id"] = config_id
            logging.critical("config id: {0}".format(config_id))
            cprint(config)
            cprint("\n")

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
            cprint("Average evaluation result across different seeds: ")
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
                print("Best MRR: ", output["MRR"])


            # save to log
            with open("./log/{0}.txt".format(time.perf_counter()), "a+") as f:
                for output in outputs:
                    f.write(json.dumps("-"*80) + "\n")
                    f.write(json.dumps(output) + "\n")
