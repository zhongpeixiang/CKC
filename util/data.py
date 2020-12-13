import random
from collections import Counter
from itertools import chain
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import torch
from torch_geometric.data import Data
import numpy as np


##########################################################################
# from target-guided chat repo
##########################################################################
_lemmatizer = WordNetLemmatizer()
def tokenize(example, ppln):
    for fn in ppln:
        example = fn(example)
    return example


def kw_tokenize(string):
    return tokenize(string, [nltk_tokenize, lower, pos_tag, to_basic_form])


def simp_tokenize(string):
    return tokenize(string, [nltk_tokenize, lower])


def nltk_tokenize(string):
    return nltk.word_tokenize(string)


def lower(tokens):
    if not isinstance(tokens, str):
        return [lower(token) for token in tokens]
    return tokens.lower()


def pos_tag(tokens):
    return nltk.pos_tag(tokens)


def to_basic_form(tokens):
    if not isinstance(tokens, tuple):
        return [to_basic_form(token) for token in tokens]
    word, tag = tokens
    if tag.startswith('NN'):
        pos = 'n'
    elif tag.startswith('VB'):
        pos = 'v'
    elif tag.startswith('JJ'):
        pos = 'a'
    else:
        return word
    return _lemmatizer.lemmatize(word, pos)

from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

brown_ic = wordnet_ic.ic('ic-brown.dat')


def calculate_linsim(a, b):
    linsim = -1
    syna = wn.synsets(a)
    synb = wn.synsets(b)
    for sa in syna:
        for sb in synb:
            try:
                linsim = max(linsim, sa.lin_similarity(sb, brown_ic))
            except:
                pass
    return linsim


def is_reach_goal(context, goal, keyword2id):
    context = kw_tokenize(context)
    if goal in context:
        return True

    # for wd in context:
    #     if wd in keyword2id:
    #         rela = calculate_linsim(wd, goal)
    #         if rela > 0.9:
    #             return True
    return False


def make_context(string, keyword2id):
    string = kw_tokenize(string)
    context = []
    for word in string:
        if word in keyword2id:
            context.append(word)
    return context
######################################################################################
######################################################################################

def pad_sentence(sent, max_sent_len, pad_token):
    if len(sent) >= max_sent_len:
        return sent[:max_sent_len]
    else:
        return sent + (max_sent_len - len(sent)) * [pad_token]

def pad_and_clip_data(convs, keywords, min_context_len, max_context_len, max_sent_len, max_keyword_len):
    """
        convs: list of tokenized conversations
        keywords: list of conv keywords

        return: [(context, response)], [context_keywords, response_keywords)]
    """
    print("clipping and padding utterances and conversations...")
    assert len(convs) == len(keywords)
    padded_convs, padded_keywords = [], []
    pad_token = "<pad>"
    pad_sent = max_sent_len*[pad_token]
    pad_sent_kw = max_keyword_len*[pad_token]
    
    for k, (conv, conv_kw) in tqdm(enumerate(zip(convs, keywords)), total=len(convs)):
        padded_conv, padded_conv_kw = [], []
        
        # clip and pad sents and sent keywords
        for sent, sent_kw in zip(conv, conv_kw):
            padded_conv.append(pad_sentence(sent, max_sent_len, pad_token))
            padded_conv_kw.append(pad_sentence(sent_kw, max_keyword_len, pad_token))
        
        # clip and pad conversations, start with min of 2 context utterrances, 
        # if response has no keyword, skip this example
        for i in range(min_context_len+1, len(padded_conv)+1):
            if padded_conv_kw[i-1][0] != "<pad>":
                start_idx = max(0, (i-max_context_len))
                padded_convs.append((padded_conv[start_idx: i-1] + (max_context_len - (i - start_idx)) * [pad_sent], [padded_conv[i-1]]))
                padded_keywords.append((padded_conv_kw[start_idx: i-1] + (max_context_len - (i - start_idx)) * [pad_sent_kw], [padded_conv_kw[i-1]]))
    
    return padded_convs, padded_keywords

def pad_and_clip_candidate(candidates, max_sent_len):
    print("clipping and padding candidates...")
    padded_candidates = []
    pad_token = "<pad>"
    
    for ex_cand in tqdm(candidates):
        padded_ex_cand = []
        for cand in ex_cand:
            padded_ex_cand.append(pad_sentence(cand, max_sent_len, pad_token))
        padded_candidates.append(padded_ex_cand)
    return padded_candidates

def build_vocab(data, vocab_size):
    """
    data: tokenized conversations
    """
    print("building conversation vocabulary...")
    word_counter = Counter()
    for ex_context, ex_response in tqdm(data, total=len(data)):
        for sent in (ex_context + ex_response):
            word_counter.update(sent)
    word_counter.pop("<pad>")
    
    word2id = {
        "<pad>": 0,
        "<unk>": 1
    }
    
    for w,cnt in word_counter.most_common(vocab_size-len(word2id)):
        word2id[w] = len(word2id)
    
    print("vocab size: ", len(word2id))
    return word2id

def convert_convs_to_ids(convs, word2id):
    print("converting {0} conversations to token ids...".format(len(convs)))
    conv_ids = []
    for ex_context, ex_response in tqdm(convs, total=len(convs)):
        ex_context_id = []
        ex_response_id = []
        for sent in ex_context:
            ex_context_id.append([word2id[w] if w in word2id else word2id["<unk>"] for w in sent])
        for sent in ex_response:
            ex_response_id.append([word2id[w] if w in word2id else word2id["<unk>"] for w in sent])
        conv_ids.append((ex_context_id, ex_response_id))
    return conv_ids

def convert_candidates_to_ids(candidates, word2id):
    print("converting {0} conversation candidates to token ids...".format(len(candidates)))
    candidate_ids = []
    for ex_cand in tqdm(candidates):
        ex_cand_ids = []
        for cand in ex_cand:
            ex_cand_ids.append([word2id[w] if w in word2id else word2id["<unk>"] for w in cand])
        candidate_ids.append(ex_cand_ids)
    return candidate_ids


def create_batches_keyword_prediction(convs, keywords, max_keyword_context_len, batch_size, shuffle=True, remove_self_loop=False, \
    keywordid2wordid=None, keyword_mask_matrix=None, use_last_k_utterances=-1, use_utterance_concepts=False, keyword2id=None, node2id=None, id2word=None):
    """
        convs: N x [((context_len, max_sent_len), (1, max_sent_len))]
        keywords: N x [((context_len, max_keyword_len), (1, max_keyword_len))]
    """
    print("Creating batches...")
    if shuffle:
        conv_keyword_pairs = list(zip(convs, keywords))
        random.shuffle(conv_keyword_pairs)
        convs, keywords = zip(*conv_keyword_pairs)
    
    max_sent_len = len(convs[0][0][0])
    # batch_indices = list(range(0, len(convs), batch_size)) + [len(convs)]
    data = []
    total_pairs = 0
    total_concepts = 0
    batch_X_keywords = []
    batch_X_utterances = []
    batch_X_concepts = []
    batch_y = []

    for conv, conv_kw in tqdm(zip(convs, keywords)):
        response_idx = 0
        for i in range(len(conv_kw[0])-1, -1, -1):
            if conv_kw[0][i][0] != 0:
                response_idx = i+1
                break
                
        # contextual keywords of last 2 utterances
        context_kw = []
        for w in chain(reversed(conv_kw[0][response_idx-2]), reversed(conv_kw[0][response_idx-1])):
            if w != 0:
                context_kw.append(w)
        context_kw = pad_sentence(context_kw, max_keyword_context_len, 0)

        # response keywords
        response_kw = []
        for w in conv_kw[1][0]:
            if w != 0:
                if remove_self_loop:
                    if w not in context_kw:
                        response_kw.append(w)
                else:
                    response_kw.append(w)
        response_kw = pad_sentence(response_kw, max_keyword_context_len//2, 0)
        
        if keyword_mask_matrix is not None:
            # keep only those neighboring keywords in the response_kw
            filtered_response_kw = []
            for s_kw in context_kw:
                if s_kw == 0:
                    continue
                for e_kw in response_kw:
                    if e_kw == 0:
                        continue
                    # if e_kw is a neighbor of s_kw, keep it
                    if keyword_mask_matrix[s_kw, e_kw] == 1 and e_kw not in filtered_response_kw:
                        filtered_response_kw.append(e_kw)
            response_kw = pad_sentence(filtered_response_kw, max_keyword_context_len//2, 0)

        if context_kw[0] == 0 or response_kw[0] == 0:
            continue
        
        total_pairs += len([1 for w in context_kw if w != 0]) * len([1 for w in response_kw if w != 0])

        batch_X_keywords.append(context_kw) # input the contextual keywords in one sentence
        batch_y.append(response_kw) # output is the response keywords in one sentence
        if use_last_k_utterances <= 0:
            batch_X_utterances.append(conv[0])
        else:
            # get last k utterances from conv[0]
            valid_utterances = []
            for utter in conv[0]:
                if utter[0] != 0:
                    valid_utterances.append(utter)
            valid_utterances = valid_utterances[-use_last_k_utterances:]
            # pad the rest utterances
            valid_utterances = valid_utterances + [[0]*len(valid_utterances[0])] * (use_last_k_utterances - len(valid_utterances))
            batch_X_utterances.append(valid_utterances)

        if use_utterance_concepts:
            # batch_X_utterances[-1]: (max_context_len, max_sent_len)
            # extract all concepts from batch_X_utterances[-1]
            utterance_concepts = []
            for utter in batch_X_utterances[-1]:
                if utter[0] == 0:
                    continue
                utter = [id2word[w] for w in utter if w != 0]
                all_utter_ngrams = []
                for n in range(5, 0, -1):
                    all_utter_ngrams.extend(ngrams(utter, n))
                for w in all_utter_ngrams:
                    w = "_".join(w)
                    if w in node2id and not any([w in ngram for ngram in utterance_concepts]):
                        utterance_concepts.append(w)
            total_concepts += len(utterance_concepts)
            utterance_concepts = [node2id[w] for w in utterance_concepts]
            utterance_concepts = pad_sentence(utterance_concepts, max_sent_len, node2id["<pad>"])
            batch_X_concepts.append(utterance_concepts)


        if len(batch_y) == batch_size:
            data.append({
                "batch_X_keywords": batch_X_keywords,
                "batch_X_utterances": batch_X_utterances,
                "batch_X_concepts": batch_X_concepts,
                "batch_y": batch_y
            })
            batch_X_keywords = []
            batch_X_utterances = []
            batch_X_concepts = []
            batch_y = []
    
    # last batch
    if len(batch_y) > 0:
        data.append({
            "batch_X_keywords": batch_X_keywords,
            "batch_X_utterances": batch_X_utterances,
            "batch_X_concepts": batch_X_concepts,
            "batch_y": batch_y
        })
    
    print("total number of turns: ", len(convs))
    print("total number of keyword transitions: ", total_pairs)
    print("total number of concepts: ", total_concepts)
    return data


def extract_keywords_from_candidates(candidates, keyword2id):
    keywords = []
    for conv_cands in tqdm(candidates):
        conv_cand_kws = []
        for sent in conv_cands:
            simple_tokens = kw_tokenize(" ".join(sent))
            conv_cand_kws.append([w for w in simple_tokens if w in keyword2id])
        keywords.append(conv_cand_kws)
    return keywords

def extract_concepts(utter, id2word, node2id, max_sent_len):
    if utter[0] == 0:
        return [node2id["<pad>"]]*max_sent_len
    utter_concepts = []
    utter = [id2word[w] for w in utter if w != 0]
    all_utter_ngrams = []
    for n in range(5, 0, -1):
        all_utter_ngrams.extend(ngrams(utter, n))
    for w in all_utter_ngrams:
        w = "_".join(w)
        if w in node2id and not any([w in ngram for ngram in utter_concepts]):
            utter_concepts.append(w)
    utter_concepts = [node2id[w] for w in utter_concepts]
    utter_concepts = pad_sentence(utter_concepts, max_sent_len, node2id["<pad>"])
    return utter_concepts


def create_batches_retrieval(convs, keywords, candidates, candidate_keywords, max_keyword_context_len, batch_size, shuffle=True, use_keywords=False, \
    use_candidate_keywords=False, use_utterance_concepts=False, node2id=None, id2word=None, flatten_context=False, use_last_k_utterances=-1):
    """
        convs: (N, context_len, max_sent_len)
        keywords: (N, context_len, max_keyword_len)
        candidates: (N, num_candidates, max_sent_len)
        candidate_keywords: (N, num_candidates, max_keyword_len)
    """
    if shuffle:
        if candidate_keywords is None:
            conv_keyword_candidate_tuples = list(zip(convs, keywords, candidates))
            random.shuffle(conv_keyword_candidate_tuples)
            convs, keywords, candidates = zip(*conv_keyword_candidate_tuples)
        else:
            conv_keyword_candidate_tuples = list(zip(convs, keywords, candidates, candidate_keywords))
            random.shuffle(conv_keyword_candidate_tuples)
            convs, keywords, candidates, candidate_keywords = zip(*conv_keyword_candidate_tuples)
    
    max_sent_len = len(convs[0][0][0])
    data = []
    batch_context = []
    batch_context_kw = []
    batch_candidates = []
    batch_candidates_kw = []
    batch_context_concepts = []
    batch_candidates_concepts = []
    batch_context_for_keywords_prediction = []
    batch_context_concepts_for_keywords_prediction = []
    for i in range(len(convs)):
        conv = convs[i]
        conv_kw = keywords[i]
        conv_cand = candidates[i]
        conv_cand_kw = []
        conv_concepts = []
        conv_cand_concepts = []

        response_idx = 0
        for i in range(len(conv_kw[0])-1, -1, -1):
            if conv_kw[0][i][0] != 0:
                response_idx = i+1
                break
        
        # contextual keywords of last 2 utterances
        if use_keywords:
            context_kw = []
            for w in conv_kw[0][response_idx-2] + conv_kw[0][response_idx-1]:
                if w != 0:
                    context_kw.append(w)
            context_kw = pad_sentence(context_kw, max_keyword_context_len, 0)
            batch_context_kw.append(context_kw)

        if use_candidate_keywords:
            conv_cand_kw = candidate_keywords[i]
            batch_candidates_kw.append(conv_cand_kw)
        
        batch_context.append(conv[0])
        batch_candidates.append(conv_cand)
        
        if use_utterance_concepts:
            # batch_X_utterances[-1]: (max_context_len, max_sent_len)
            # extract all concepts from batch_X_utterances[-1]
            for utter in batch_context[-1]:
                utter_concepts = extract_concepts(utter, id2word, node2id, max_sent_len) # padded concepts
                conv_concepts.append(utter_concepts)
            batch_context_concepts.append(conv_concepts)

            # candidate concepts
            for utter in batch_candidates[-1]:
                utter_concepts = extract_concepts(utter, id2word, node2id, max_sent_len)
                conv_cand_concepts.append(utter_concepts)
            batch_candidates_concepts.append(conv_cand_concepts)

        if flatten_context:
            # flatten context into a single sentence
            flattened_context = []
            for utter in batch_context[-1]:
                flattened_context.extend([w for w in utter if w != 0])
            flattened_context = pad_sentence(flattened_context, max_sent_len*len(batch_context[-1]), 0)
            batch_context[-1] = [flattened_context]

            if use_utterance_concepts:
                flattened_context_concepts = []
                for utter in batch_context_concepts[-1]:
                    flattened_context_concepts.extend([w for w in utter if w != 0])
                flattened_context_concepts = pad_sentence(flattened_context_concepts, max_sent_len*len(batch_context_concepts[-1]), 0)
                batch_context_concepts[-1] = [flattened_context_concepts]
        
        if use_last_k_utterances > 0:
            # get last k utterances from conv[0]
            valid_utterances = []
            for utter in conv[0]:
                if utter[0] != 0:
                    valid_utterances.append(utter)
            valid_utterances = valid_utterances[-use_last_k_utterances:]
            # pad the rest utterances
            valid_utterances = valid_utterances + [[0]*len(valid_utterances[0])] * (use_last_k_utterances - len(valid_utterances))
            batch_context_for_keywords_prediction.append(valid_utterances)

            if use_utterance_concepts:
                # batch_X_utterances[-1]: (max_context_len, max_sent_len)
                # extract all concepts from batch_X_utterances[-1]
                conv_concepts_for_keywords_prediction = []
                for utter in batch_context_for_keywords_prediction[-1]:
                    utter_concepts = extract_concepts(utter, id2word, node2id, max_sent_len) # padded concepts
                    conv_concepts_for_keywords_prediction.extend([w for w in utter_concepts if w != node2id["<pad>"]])
                conv_concepts_for_keywords_prediction = pad_sentence(conv_concepts_for_keywords_prediction, max_sent_len, node2id["<pad>"])
                batch_context_concepts_for_keywords_prediction.append(conv_concepts_for_keywords_prediction)

        batch_data = {
                "batch_context": batch_context,
                "batch_candidates": batch_candidates
            }
        
        if use_keywords:
            batch_data["batch_context_kw"] = batch_context_kw
        if use_candidate_keywords:
            batch_data["batch_candidates_kw"] = batch_candidates_kw
        if use_utterance_concepts:
            batch_data["batch_context_concepts"] = batch_context_concepts
            batch_data["batch_candidates_concepts"] = batch_candidates_concepts
        if use_last_k_utterances > 0:
            batch_data["batch_context_for_keywords_prediction"] = batch_context_for_keywords_prediction
            batch_data["batch_context_concepts_for_keywords_prediction"] = batch_context_concepts_for_keywords_prediction
        
        if len(batch_context) == batch_size:
            data.append(batch_data)
            batch_context = []
            batch_context_kw = []
            batch_candidates = []
            batch_candidates_kw = []
            batch_context_concepts = []
            batch_candidates_concepts = []
            batch_context_for_keywords_prediction = []
            batch_context_concepts_for_keywords_prediction = []
        
        try:
            assert conv[1][0] == conv_cand[0]
        except AssertionError:
            print("the first candidate is not gold response")
            print(conv[0])
            print(conv[1][0])
            print(conv_cand)
            exit()
    
    if len(batch_context) > 0:
        batch_data = {
                "batch_context": batch_context,
                "batch_candidates": batch_candidates
            }
        if use_keywords:
            batch_data["batch_context_kw"] = batch_context_kw
        if use_candidate_keywords:
            batch_data["batch_candidates_kw"] = batch_candidates_kw
        if use_utterance_concepts:
            batch_data["batch_context_concepts"] = batch_context_concepts
            batch_data["batch_candidates_concepts"] = batch_candidates_concepts
        if use_last_k_utterances > 0:
            batch_data["batch_context_for_keywords_prediction"] = batch_context_for_keywords_prediction
            batch_data["batch_context_concepts_for_keywords_prediction"] = batch_context_concepts_for_keywords_prediction
        data.append(batch_data)
    
    return data


def process_history(history, word2id, keyword2id, max_seq_len=30, max_context_len=8, max_keyword_len=10):
    if len(history) > max_context_len:
        history = history[-max_context_len:]

    # extract keywords
    context_keywords = None
    if keyword2id is not None:
        if len(history) == 1:
            context_keywords = make_context(history[-1], keyword2id)
        else:
            context_keywords = make_context(history[-2] + history[-1], keyword2id)
        
        # clip and pad context_keywords
        context_keywords = pad_sentence(context_keywords, 2*max_keyword_len, "<pad>")
        
        # convert to ids
        context_keywords = [keyword2id[w] if w in keyword2id else keyword2id["<unk>"] for w in context_keywords] # (2*max_keyword_len, )

    # tokenize
    context_utterances = []
    for sent in history:
        tokens = simp_tokenize(sent)
        tokens = pad_sentence(tokens, max_seq_len, "<pad>")
        
        # convert to ids
        tokens = [word2id[w] if w in word2id else word2id["<unk>"] for w in tokens]
        context_utterances.append(tokens)
    
    # pad conversations
    context_utterances += [[word2id["<pad>"]] * max_seq_len] * (max_context_len - len(context_utterances)) # (max_context_len, max_seq_len)

    return context_utterances, context_keywords


def process_candidate(candidates, word2id, max_sent_len):
    print("processing candidates...")
    candidate_ids = []
    pad_token = "<pad>"
    for cand in tqdm(candidates):
        cand = pad_sentence(cand.split(" "), max_sent_len, pad_token)
        cand_ids = [word2id[w] if w in word2id else word2id["<unk>"] for w in cand]
        candidate_ids.append(cand_ids)
    return candidate_ids


def process_candidate_GNN(candidates, word2id, id2word, node2id, max_sent_len):
    # candidates: a list of utterances
    print("processing candidates...")
    candidate_ids = []
    candidate_concepts = []
    pad_token = "<pad>"
    for cand in tqdm(candidates):
        cand_tokens = pad_sentence(cand.split(" "), max_sent_len, pad_token)
        cand_ids = [word2id[w] if w in word2id else word2id["<unk>"] for w in cand_tokens]
        cand_concepts = extract_concepts(cand_ids, id2word, node2id, max_sent_len)

        candidate_ids.append(cand_ids)
        candidate_concepts.append(cand_concepts)
    return candidate_ids, candidate_concepts