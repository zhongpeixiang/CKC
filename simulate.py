import os
from tqdm import tqdm
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from model import KW_Neural, KW_PMI, KW_Kernel, KW_DKRN, KW_GNN, SMN, CoGraphMatcher

from util.io import load_pickle, save_pickle, load_edge_mask, load_nx_graph_hopk
from util.data import kw_tokenize, pad_sentence, process_history, process_candidate, is_reach_goal, process_candidate_GNN, extract_concepts
from util.tool import DialogueManager, sample_top_candidates, select_response, select_response_strategy_keyword, select_keyword_graph

def chat(chat_model, kw_model, history, candidate_out, DM, device, is_base_agent=False):
    word2id, id2word, keyword2id, target_kw, CN_hopk_edge_index, node2id = DM.word2id, DM.id2word, DM.keyword2id, DM.target_kw, DM.CN_hopk_edge_index, DM.node2id

    context_utterances, context_keywords = process_history(history, DM.word2id, DM.keyword2id, max_seq_len=30, max_context_len=8, max_keyword_len=10)
    # context_utterances: a list of padded utterances (token ids)

    next_kw = "<pad>"
    if kw_model is not None:
        # predict keyword distribution for the next turn
        context_keywords = torch.LongTensor(context_keywords).to(device)
        context_utterances_for_keyword_prediction = []
        for i, utter in enumerate(context_utterances):
            if utter[0] == 0:
                break
        context_utterances_for_keyword_prediction = context_utterances[i-2:i] if i>= 2 else context_utterances[i-1:i]
        context_concepts_for_keyword_prediction = []
        for utter in context_utterances_for_keyword_prediction:
            utter_concepts = extract_concepts(utter, id2word, node2id, 30)
            context_concepts_for_keyword_prediction.extend([w for w in utter_concepts if w != node2id["<pad>"]])
        context_concepts_for_keyword_prediction = pad_sentence(context_concepts_for_keyword_prediction, 30, node2id["<pad>"])
        context_utterances_for_keyword_prediction = torch.LongTensor(context_utterances_for_keyword_prediction).to(device)
        context_concepts_for_keyword_prediction = torch.LongTensor(context_concepts_for_keyword_prediction).to(device)
        # print(context_keywords.shape, context_utterances_for_keyword_prediction.shape, context_concepts_for_keyword_prediction.shape)
        # print(context_keywords, context_utterances_for_keyword_prediction, context_concepts_for_keyword_prediction)
        with torch.no_grad():
            kw_logits = kw_model(None, None, CN_hopk_edge_index, context_keywords.unsqueeze(0), x_utter=context_utterances_for_keyword_prediction.unsqueeze(0), \
                x_concept=context_concepts_for_keyword_prediction.unsqueeze(0))[0][0].cpu() # kw_logits: (keyword_vocab_size)

        # choose the most suitable keyword
        target_kw_id = keyword2id[target_kw] if target_kw in keyword2id else keyword2id["<unk>"]

        kw_mask = keyword_mask_matrix[context_keywords.cpu()].sum(dim=0).clamp(min=0, max=1) # (keyword_vocab_size)
        next_kw, score = select_keyword_graph(kw_logits, target_kw_id, DM, kw_mask)

        DM.next_kws.append(next_kw)
        DM.scores.append(score)

    # keyword-augmented response retrieval
    with torch.no_grad():
        if is_base_agent:
            context_out = chat_model.encode_context(torch.LongTensor(context_utterances).to(device).unsqueeze(0), torch.LongTensor([word2id[next_kw]]).to(device).unsqueeze(0))
            candidate_logits = chat_model.predict(context_out, candidate_out)[0] # (pool_size, )
        else:
            context = torch.LongTensor(context_utterances).to(device).unsqueeze(0) # (1, max_context_len, seq_len)
            context_concepts = []
            for utter in context_utterances:
                context_concepts.append(extract_concepts(utter, id2word, node2id, 30))
            context_concepts = torch.LongTensor(context_concepts).to(device).unsqueeze(0) # (1, max_context_len, seq_len)
            context_keywords = torch.LongTensor([keyword2id[next_kw]]).to(device).unsqueeze(0) # (1, 1)
            
            # print(context.shape, context_concepts.shape, context_keywords.shape)
            # print(context, context_concepts, context_keywords)
            context_out, context_mask, context_concept_out, context_concept_mask, context_keywords_concept_out, context_keywords_out = \
                chat_model.encode_context_offline(context, context_concepts, context_keywords, CN_hopk_edge_index)
            # print("context_out: ", context_out.shape)
            # print("context_mask: ", context_mask.shape)
            # print("context_concept_out: ", context_concept_out.shape)
            # print("context_concept_mask: ", context_concept_mask.shape)
            # print("context_keywords_concept_out: ", context_keywords_concept_out.shape)
            # print("context_keywords_out: ", context_keywords_out.shape)
            candidate_logits = chat_model.predict(context_out, context_mask, context_concept_out, context_concept_mask, context_keywords_concept_out, context_keywords_out, \
                candidate_out, candidate_mask, None, None, candidate_keywords_concept_out, candidate_keywords_out)[0] # (pool_size, )
        candidate_probs = torch.softmax(candidate_logits, dim=-1) # (pool_size, )
    top_candidates = sample_top_candidates(candidate_probs, DM) # greedy, random and top-k

    # choose the response
    if is_base_agent:
        response, response_prob = select_response(top_candidates, DM)
    else:
        response, response_prob = select_response_strategy_keyword(top_candidates, DM)
        DM.response_probs.append(response_prob)
    DM.reply_list.append(response)

    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--chat_model', type=str, required=True)
    parser.add_argument('--kw_model', type=str, default="")
    parser.add_argument('--num_sessions', type=int, required=True)
    parser.add_argument('--max_turns', type=int, default=8)
    parser.add_argument('--device', type=int, default=0, help="-1 for cpu, >=0 for gpu ids")

    args = parser.parse_args()
    dataset = args.dataset
    chat_model_name = args.chat_model
    kw_model_name = args.kw_model
    num_sessions = args.num_sessions
    max_turns = args.max_turns
    device = args.device if args.device >= 0 else "cpu"
    device = torch.device(device)

    apply_commonsense = "Commonsense" in chat_model_name

    # load starting corpus: a list of greeting sentences from the training conversations
    # load target set: a list of target keywords
    print("Loading data...")
    start_corpus = load_pickle("./data/{0}/start_corpus.pkl".format(dataset))
    target_set = load_pickle("./data/{0}/target_set.pkl".format(dataset))
    candidate_pool = load_pickle("./data/{0}/candidate_pool.pkl".format(dataset)) # all candidates from training set 
    # candidate_pool = random.sample(candidate_pool, 2000)

    # load chat model
    chat_model_path = "./saved_model/{0}/{1}.pt".format(dataset, chat_model_name)
    print("Loading chat model from {0}...".format(chat_model_path))
    chat_model_checkpoint = torch.load(chat_model_path, map_location=device)
    word2id = chat_model_checkpoint.pop("word2id")
    id2word = {idx: w for w, idx in word2id.items()}
    print("word vocab size: ", len(word2id))
    chat_model_kwargs = chat_model_checkpoint.pop("model_kwargs")
    if "Commonsense" in chat_model_name:
        # chat_model_name.replace("_Commonsense", "")
        # chat_model_name = chat_model_name if "_" not in chat_model_name else chat_model_name.split("_")[0]
        chat_model_name = "CoGraphMatcher"
    chat_model = globals()[chat_model_name](**chat_model_kwargs) # create model
    chat_model.load_state_dict(chat_model_checkpoint)
    chat_model.to(device)
    chat_model.eval()

    # load human model
    human_model_path = "./saved_model/{0}/{1}.pt".format(dataset, "SMN")
    print("Loading human model from {0}...".format(human_model_path))
    human_model_checkpoint = torch.load(human_model_path, map_location=device)
    human_model_checkpoint.pop("word2id")
    # id2word = {idx: w for w, idx in word2id.items()}
    human_model_kwargs = human_model_checkpoint.pop("model_kwargs")
    human_model = globals()["SMN"](**human_model_kwargs) # create model
    human_model.load_state_dict(human_model_checkpoint)
    human_model.to(device)
    human_model.eval()

    # load keyword model
    kw_model = None
    keyword2id = None
    kw_model_path = "./saved_model/{0}/{1}.pt".format(dataset, kw_model_name)
    print("Loading keyword model from {0}...".format(kw_model_path))
    kw_model_checkpoint = torch.load(kw_model_path, map_location=device)
    keyword2id = kw_model_checkpoint.pop("word2id") # this is actually word2id for KW_GNN model
    keyword2id = load_pickle("./data/{0}/keyword2id.pkl".format(dataset))
    kw_model_kwargs = kw_model_checkpoint.pop("model_kwargs")
    kw_model = globals()["KW_GNN"](**kw_model_kwargs) # create model
    kw_model.load_state_dict(kw_model_checkpoint)
    kw_model.to(device)
    kw_model.eval()
    keyword_vocab_size = len(keyword2id)
    print("keyword vocab size: ", keyword_vocab_size)

    keywordid2wordid = None
    print("keywords that are not in word2id: ", set(keyword2id.keys()) - set(word2id.keys()))
    id2keyword = {idx:w for w, idx in keyword2id.items()}
    keywordid2wordid = [word2id[id2keyword[i]] if id2keyword[i] in word2id else word2id["<unk>"] for i in range(len(keyword2id))]

    keyword_mask_matrix = None
    keyword_graph_distance_matrix = None
    CN_hopk_edge_index = None
    node2id = None
    if apply_commonsense:
        CN_hopk_graph_path = "./data/{0}/CN_graph_{1}hop_ge1.pkl".format(dataset, 1)
        print("Loading graph from ", CN_hopk_graph_path)
        CN_hopk_graph_dict = load_nx_graph_hopk(CN_hopk_graph_path, word2id, keyword2id)
        CN_hopk_edge_index = torch.LongTensor(CN_hopk_graph_dict["edge_index"]).transpose(0,1).to(device) # (2, num_edges)
        # CN_hopk_nodeid2wordid = torch.LongTensor(CN_hopk_graph_dict["nodeid2wordid"]).to(device) # (num_nodes, 10)
        node2id = CN_hopk_graph_dict["node2id"]
        id2node = {idx:w for w,idx in node2id.items()}
        print("building keyword mask matrix...")
        keyword_mask_matrix = torch.from_numpy(CN_hopk_graph_dict["edge_mask"]).float() # numpy array of (keyword_vocab_size, keyword_vocab_size)
        keyword_mask_matrix[torch.arange(keyword_vocab_size), torch.arange(keyword_vocab_size)] = 1 # add self loop
        print(keyword_mask_matrix[:8,:8])

        print("Loading node graph distance matrix...")
        keyword_graph_distance_matrix = np.ones((len(keyword2id), len(keyword2id))) * 1000
        keyword_graph_distance_dict = load_pickle("./data/{0}/keyword_graph_weighted_distance_dict.pkl".format(dataset))
        for node1, node2 in keyword_graph_distance_dict.keys():
            keyword_graph_distance_matrix[keyword2id[node1], keyword2id[node2]] = keyword_graph_distance_dict[(node1, node2)]
        print(keyword_graph_distance_matrix[:8,:8])

    # init GNN node embeddings first
    print("initializing GNN node embeddings...")
    chat_model.init_gnn_emb(CN_hopk_edge_index)

    # encode candidate pool
    print("sample candidates: ", candidate_pool[:3])
    candidate_pool_ids, candidate_concept_ids = process_candidate_GNN(candidate_pool, word2id, id2word, node2id, 30) # clip, pad and numericalize, (pool_size, seq_len)
    # load candidate keywords
    candidate_keyword_path = "./data/{0}/candidate_pool_keyword.pkl".format(dataset)
    candidate_keyword_ids = []
    if os.path.exists(candidate_keyword_path):
        print("Loading candidate keywords from ", candidate_keyword_path)
        candidate_keyword_ids = load_pickle(candidate_keyword_path)
    
    if len(candidate_keyword_ids) != len(candidate_pool):
        print("Creating candidate keywords...")
        candidate_keyword_ids = []
        pad_token = "<pad>"
        for cand in tqdm(candidate_pool):
            cand_kw_tokens = pad_sentence(kw_tokenize(cand), 10, pad_token)
            cand_kw_tokens = [keyword2id[w] if w in keyword2id else keyword2id["<unk>"] for w in cand_kw_tokens]
            candidate_keyword_ids.append(cand_kw_tokens)
        save_pickle(candidate_keyword_ids, candidate_keyword_path)
    
    print("sample candidate_pool_ids: ", len(candidate_pool_ids), candidate_pool_ids[:3], [[id2word[t] for t in u if t!=0] for u in candidate_pool_ids[:3]])
    print("sample candidate_concept_ids: ", len(candidate_concept_ids), candidate_concept_ids[:3], [[id2node[t] for t in u if t!=0] for u in candidate_concept_ids[:3]])
    print("sample candidate_keyword_ids: ", len(candidate_keyword_ids), candidate_keyword_ids[:3], [[id2keyword[t] for t in u if t!=0] for u in candidate_keyword_ids[:3]])

    # encode candidates for CoGraphMatcher
    print("Encoding candidate pool for chat model...")
    with torch.no_grad():
        chunk_size = 2000
        chunk_ids = list(range(0, len(candidate_pool_ids), chunk_size)) + [len(candidate_pool_ids)]
        chunk_candidate_outs = []
        for s, e in zip(chunk_ids[:-1], chunk_ids[1:]):
            chunk_candidate_outs.append(chat_model.encode_candidate_offline(\
                torch.LongTensor(candidate_pool_ids[s:e]).to(device).unsqueeze(0), \
                    torch.LongTensor(candidate_concept_ids[s:e]).to(device).unsqueeze(0), \
                        torch.LongTensor(candidate_keyword_ids[s:e]).to(device).unsqueeze(0), \
                            CN_hopk_edge_index)) # (chunk_size, out_size)
        candidate_out, candidate_mask, candidate_concept_out, candidate_concept_mask, candidate_keywords_concept_out, candidate_keywords_out = zip(*chunk_candidate_outs)
        candidate_out = torch.cat(candidate_out, dim=0) # (pool_size, m, out_size)
        candidate_mask = torch.cat(candidate_mask, dim=0) # (pool_size, m)
        candidate_keywords_concept_out = torch.cat(candidate_keywords_concept_out, dim=0) # (pool_size, out_size)
        candidate_keywords_out = torch.cat(candidate_keywords_out, dim=0) # (pool_size, out_size)
    print("candidate_out: ", candidate_out.shape)
    print("candidate_mask: ", candidate_mask.shape)
    print("candidate_keywords_concept_out: ", candidate_keywords_concept_out.shape)
    print("candidate_keywords_out: ", candidate_keywords_out.shape)

    # encode candidate pool for SMN
    print("Encoding candidate pool for human model...")
    with torch.no_grad():
        chunk_candidate_outs = []
        for s, e in zip(chunk_ids[:-1], chunk_ids[1:]):
            chunk_candidate_outs.append(human_model.encode_candidate(torch.LongTensor(candidate_pool_ids[s:e]).to(device).unsqueeze(0))) # (1, chunk_size, out_size)
        candidate_out_human = torch.cat(chunk_candidate_outs, dim=1) # (1, pool_size, out_size)
    print("candidate_out_human: ", candidate_out_human.shape)

    # dialogue manager
    DM = DialogueManager(word2id, id2word, keyword2id, keywordid2wordid, candidate_pool, bool(kw_model), \
        keyword_mask_matrix, keyword_graph_distance_matrix, CN_hopk_edge_index, node2id)

    # chat
    print("Start chatting...")
    sessions_done = 0
    sessions_success = 0
    num_turns = []
    smoothness = []
    for session_id in range(num_sessions):
        history = []
        history.append(random.choice(start_corpus))
        target_kw = None
        while target_kw not in word2id:
            target_kw = random.choice(target_set)

        score = 0
        success = False

        # init DM
        DM.target_kw = target_kw
        DM.scores.append(score)
        
        # start chatting
        print('-'*100)
        print("SESSION ", session_id)
        print('START: ' + history[0])
        for i in range(max_turns):
            # human model
            reply = chat(human_model, None, history, candidate_out_human, DM, device, True)
            print('AGENT (BASE): ', reply)
            history.append(reply)
            
            # chat model
            reply = chat(chat_model, kw_model, history, candidate_out, DM, device)
            if len(DM.next_kws) != 0 and DM.next_kws[-1] != "<pad>":
                print("PREDICTED KEYWORD: ", DM.next_kws[-1])
            print('AGENT: ', reply)
            history.append(reply)
            
            if is_reach_goal(history[-2] + history[-1], target_kw, keyword2id):
                print('Successfully chat to the target \'{}\'.'.format(target_kw))
                success = True
                sessions_success += 1
                num_turns.append(i+1)
                break
        if not success:
            print('Failed by reaching the maximum turn, target: \'{}\'.'.format(target_kw))
        if len(DM.next_kws) != 0:
            print("PREDICTED KEYWORDS: ", DM.next_kws)
            print("SCORES: ", DM.scores)
        smoothness.append(np.mean(DM.response_probs))
        sessions_done += 1
        print("SUCCESS RATE: {0:.4f}".format(sessions_success/sessions_done))
        print("AVERAGE NUM TURNS: {0:.4f}".format(np.mean(num_turns) if len(num_turns)>0 else 0))
        print("SMOOTHNESS: {0:.6f}".format(np.mean(smoothness)))
        DM.clear_dialogue()