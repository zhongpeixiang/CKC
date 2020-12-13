import numpy as np
import torch
import torch.nn.functional as F

from util.data import kw_tokenize

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def convert_ids_to_sent(token_ids, id2owrd):
    words = []
    for idx in token_ids:
        if id2owrd[idx] == "<pad>":
            break
        words.append(id2owrd[idx])
    return " ".join(words)

class DialogueManager():
    def __init__(self, word2id, id2word, keyword2id, keywordid2wordid, candidate_pool, use_keywords, \
        keyword_mask_matrix=None, keyword_graph_distance_matrix=None, CN_hopk_edge_index=None, node2id=None):
        # global info
        self.word2id = word2id
        self.id2word = id2word
        self.keyword2id = keyword2id
        self.keywordid2wordid = keywordid2wordid
        self.candidate_pool = candidate_pool
        self.use_keywords = use_keywords
        self.keyword_mask_matrix = keyword_mask_matrix
        self.keyword_graph_distance_matrix = keyword_graph_distance_matrix
        self.CN_hopk_edge_index = CN_hopk_edge_index
        self.node2id = node2id

        # dialogue-level info
        self.target_kw = None
        self.next_kws = []
        self.reply_list = []
        self.scores = []
        self.response_probs = []

    def clear_dialogue(self):
        self.target_kw = None
        self.next_kws = []
        self.reply_list = []
        self.scores = []
        self.response_probs = []


def select_keyword_graph(kw_logits, target_kw_id, DM, kw_mask):
    # kw_logits: (vocab, )
    # kw_mask: (vocab, )

    # select keywords based on path distance to target, sorted by probs
    id2word, keywordid2wordid, current_score, keyword_graph_distance_matrix = DM.id2word, DM.keywordid2wordid, DM.scores[-1], DM.keyword_graph_distance_matrix
    
    # kw_logits: (vocab, )
    num_neighbors = kw_mask.sum().long().item()
    
    masked_kw_logits = kw_logits * kw_mask + (-5e4) * (1 - kw_mask)
    masked_kw_logits = masked_kw_logits.sort(descending=True)[1] # (vocab, )
    results = []
    for kw_id in masked_kw_logits.tolist()[:num_neighbors]:
        # print(id2word[keywordid2wordid[target_kw_id]], id2word[keywordid2wordid[kw_id]], kw_mask[kw_id])
        if kw_id not in [0,1] and keywordid2wordid[kw_id] not in [0,1]: # skip pad and unk
            w_id = keywordid2wordid[kw_id]
            if keyword_graph_distance_matrix[kw_id, target_kw_id] == 0:
                score = 100
                return id2word[w_id], score
            else:
                score = 1/keyword_graph_distance_matrix[kw_id, target_kw_id]
            results.append((w_id, score))
            # print(id2word[w_id], id2word[keywordid2wordid[target_kw_id]], score)
            if score > current_score:
                return id2word[w_id], score
    
    # if no keyword found for masked logits, repeat without mask
    # print("next-turn keyword not found in graph neighbors...")
    results = []
    kw_logits = kw_logits.sort(descending=True)[1] # (vocab, )
    for kw_id in kw_logits.tolist():
        if kw_id not in [0,1] and keywordid2wordid[kw_id] not in [0,1]: # skip pad and unk
            w_id = keywordid2wordid[kw_id]
            if keyword_graph_distance_matrix[kw_id, target_kw_id] == 0:
                score = 100
                return id2word[w_id], score
            else:
                score = 1/keyword_graph_distance_matrix[kw_id, target_kw_id]
            results.append((w_id, score))
            # print(id2word[w_id], id2word[keywordid2wordid[target_kw_id]], score)
            if score > current_score:
                return id2word[w_id], score

    best_w_id, best_score = sorted(results, key=lambda x: x[1], reverse=True)[0] # no better found, select the best keyword
    return id2word[best_w_id], best_score


def sample_top_candidates(candidate_probs, DM):
    # top_candidates = candidate_probs.topk(100)[1].tolist()
    probs, indices = candidate_probs.topk(100)
    top_candidates = list(zip(indices.tolist(), probs.tolist()))
    return top_candidates


def select_response(top_candidates, DM):
    candidate_pool, reply_list = DM.candidate_pool, DM.reply_list
    for idx, prob in top_candidates:
        if candidate_pool[idx] not in reply_list: # avoid repeat
            return candidate_pool[idx], prob
    return candidate_pool[top_candidates[0][0]], top_candidates[0][1]


def select_response_strategy_keyword(top_candidates, DM):
    target_kw, candidate_pool, reply_list, word2id, keyword2id, score, keyword_graph_distance_matrix = \
        DM.target_kw, DM.candidate_pool, DM.reply_list, DM.word2id, DM.keyword2id, DM.scores[-1], DM.keyword_graph_distance_matrix
    found = False
    reply = None
    response_prob = None
    for idx, prob in top_candidates:
        if candidate_pool[idx] not in reply_list: # avoid repeat
            if reply is None:
                reply = candidate_pool[idx]
                response_prob = prob
            for wd in kw_tokenize(candidate_pool[idx]):
                if wd in keyword2id and wd in word2id:
                    tmp_score = 1/(keyword_graph_distance_matrix[keyword2id[wd], keyword2id[target_kw]] + 1e-6)
                    if tmp_score > score:
                        reply = candidate_pool[idx]
                        response_prob = prob
                        if DM.use_keywords:
                            DM.next_kws[-1] = wd
                            DM.scores[-1] = tmp_score
                        else:
                            DM.next_kws.append(wd)
                            DM.scores.append(tmp_score)
                        found = True
                        break
        if found == False:
            continue
        break
    return reply, response_prob
