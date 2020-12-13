import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv


class KW_GNN(torch.nn.Module):
    def __init__(self, embed_size, vocab_size, keyword_vocab_size, hidden_size, output_size, n_layers, gnn, aggregation, n_heads=0, dropout=0, bidirectional=False, \
            utterance_encoder="", keywordid2wordid=None, keyword_mask_matrix=None, nodeid2wordid=None, keywordid2nodeid=None, concept_encoder="mean", \
                combine_node_emb="mean"):
        super(KW_GNN, self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.keyword_vocab_size = keyword_vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.gnn = gnn
        self.aggregation = aggregation
        self.n_heads = n_heads
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.utterance_encoder_name = utterance_encoder
        self.keywordid2wordid = keywordid2wordid
        self.keyword_mask_matrix = keyword_mask_matrix
        self.nodeid2wordid = nodeid2wordid
        self.keywordid2nodeid = keywordid2nodeid
        self.concept_encoder = concept_encoder
        self.combine_node_emb = combine_node_emb
        self.num_nodes = nodeid2wordid.shape[0]
        
        self.embedding = nn.Embedding(keyword_vocab_size, embed_size)
        
        # GNN learning
        if gnn == "GatedGraphConv":
            self.conv1 = GatedGraphConv(hidden_size, num_layers=n_layers)
            output_size = hidden_size
        
        if n_layers == 1:
            output_size = hidden_size
        
        # aggregation
        if aggregation in ["mean", "max"]:
            output_size = output_size
        
        # utterance encoder
        if self.utterance_encoder_name == "HierGRU":
            self.utterance_encoder = nn.GRU(embed_size, hidden_size, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
            self.context_encoder = nn.GRU(2*hidden_size if bidirectional else hidden_size, hidden_size, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
            output_size = output_size + 2*hidden_size if bidirectional else output_size + hidden_size

        # final linear layer
        self.mlp = nn.Linear(output_size, keyword_vocab_size)


    def forward_gnn(self, emb, edge_index):
        # emb: (keyword_vocab_size, emb_size)
        # edge_index: (2, num_edges)
        # edge_type: None or (num_edges, )
        # edge_weight: None or (num_edges, )
        if self.gnn in ["GatedGraphConv"]:
            out = self.conv1(emb, edge_index) # (keyword_vocab_size, hidden_size)
        return out

    def forward_aggregation(self, out, x):
        # out: (keyword_vocab_size, output_size)
        # x: (batch_size, seq_len)
        if self.aggregation == "mean":
            x_mask = x.ne(0).float() # (batch_size, seq_len)
            out = out[x] # (batch_size, seq_len, output_size)
            out = (out * x_mask.unsqueeze(-1)).sum(dim=1)/x_mask.sum(dim=-1, keepdim=True).clamp(min=1) # (batch_size, output_size)
        if self.aggregation == "max":
            x_mask = x.ne(0).float() # (batch_size, seq_len)
            out = out[x] # (batch_size, seq_len, output_size)
            out = torch.max(out*x_mask.unsqueeze(-1) + (-5e4)*(1-x_mask.unsqueeze(-1)), dim=1)[0] # (batch_size, output_size)
        return out

    def forward_utterance(self, x):
        # x: None or (batch_size, context_len, seq_len)
        batch_size, context_len, seq_len = x.shape
        # print(x.shape)
        # print(x.max())
        # print(self.embedding.weight.shape)
        if self.utterance_encoder_name == "HierGRU":
            seq_lengths = x.reshape(-1, seq_len).ne(0).sum(dim=-1) # (batch_size*context_len, )
            context_lengths = seq_lengths.reshape(batch_size, -1).ne(0).sum(dim=-1) # (batch_size, )
            out = self.embedding(x) # (batch_size, context_len, seq_len, emb_size)
            out, _ = self.utterance_encoder(out.reshape(batch_size*context_len, seq_len, -1)) # out: (batch_size*context_len, seq_len, num_directions * hidden_size)
            out = out[torch.arange(batch_size*context_len), (seq_lengths-1).clamp(min=0), :] # out: (batch_size*context_len, num_directions * hidden_size)
            out, _ = self.context_encoder(out.reshape(batch_size, context_len, -1)) # out: (batch_size, context_len, num_directions * hidden_size)
            out = out[torch.arange(batch_size), (context_lengths-1).clamp(min=0), :] # out: (batch_size, num_directions * hidden_size)
            return out
        return out

    def forward_concept(self, emb, nodeid2wordid):
        # emb: (vocab_size, emb_size)
        # nodeid2wordid: (num_nodes, 10)
        mask = nodeid2wordid.ne(0).float() # (num_nodes, 10)
        if self.concept_encoder == "mean":
            node_emb = (emb[nodeid2wordid] * mask.unsqueeze(-1)).sum(dim=1)/mask.sum(dim=1, keepdim=True).clamp(min=1) # (num_nodes, emb_size)
        if self.concept_encoder == "max":
            node_emb = (emb[nodeid2wordid] * mask.unsqueeze(-1) + (-5e4) * (1 - mask.unsqueeze(-1))).max(dim=1)[0] # (num_nodes, emb_size)
        return node_emb

    def forward(self, CN_hopk_edge_index, x, x_utter=None, x_concept=None):
        # CN_hopk_edge_index: (2, num_edges)
        # x: (batch_size, seq_len)
        # x_utter: None or (batch_size, context_len, max_sent_len)
        # x_concept: None or (batch_size, max_sent_len)

        # graph convolution
        emb = self.embedding.weight # (keyword_vocab_size, emb_size)
        CN_hopk_out = None
        if CN_hopk_edge_index is not None:
            node_emb = self.forward_concept(emb, self.nodeid2wordid)
            CN_hopk_out, attn = self.forward_gnn(node_emb, CN_hopk_edge_index)

        # aggregation
        if CN_hopk_edge_index is not None:
            x = self.keywordid2nodeid[x] # (batch_size, keyword_seq_len)
            CN_hopk_keyword_out = self.forward_aggregation(CN_hopk_out, x)

        # concept aggregation
        if CN_hopk_edge_index is not None and x_concept is not None:
            CN_hopk_concept_out = self.forward_aggregation(CN_hopk_out, x_concept) # (batch_size, output_size)
            # print("CN_hopk_concept_out: ", CN_hopk_concept_out.shape)
            
            if self.combine_node_emb == "mean":
                CN_hopk_out = (CN_hopk_keyword_out + CN_hopk_concept_out)/2
            if self.combine_node_emb == "max":
                CN_hopk_out = torch.stack([CN_hopk_keyword_out, CN_hopk_concept_out], dim=0).max(dim=0)[0]

        # combine two graphs
        if CN_hopk_edge_index is not None:
            if x_concept is None:
                out = CN_hopk_keyword_out
            else:
                out = CN_hopk_out

        # utterance encoder
        if self.utterance_encoder_name != "":
            utter_out = self.forward_utterance(x_utter)
            out = torch.cat([out, utter_out], dim=-1) # (batch_size, *)

        # final linear layer
        out = self.mlp(out) # out: (batch_size, keyword_vocab_size)
        return out

    def init_embedding(self, embedding, fix_word_embedding):
        print("initializing word embedding layer...")
        self.embedding.weight.data.copy_(embedding)
        if fix_word_embedding:
            self.embedding.weight.requires_grad = False


class CoGraphMatcher(nn.Module):
    def __init__(self, embed_size, vocab_size, gnn_hidden_size, gnn_layers, encoder_hidden_size, encoder_layers, n_heads, gnn, encoder, matching, \
        aggregation, use_keywords, keyword_encoder, keyword_score_weight=1, dropout=0, CN_hopk_edge_matrix_mask=None, nodeid2wordid=None, \
                keywordid2wordid=None, keywordid2nodeid=None, concept_encoder="mean", combine_word_concepts="concat"):
        super(CoGraphMatcher, self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.gnn_hidden_size = gnn_hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.gnn_layers = gnn_layers
        self.encoder_layers = encoder_layers
        self.n_heads = n_heads
        self.gnn = gnn
        self.encoder = encoder
        self.matching = matching
        self.aggregation = aggregation
        self.use_keywords = use_keywords
        self.keyword_score_weight = keyword_score_weight
        self.keyword_encoder = keyword_encoder
        self.dropout = dropout
        self.CN_hopk_edge_matrix_mask = CN_hopk_edge_matrix_mask
        self.nodeid2wordid = nodeid2wordid
        self.keywordid2wordid = keywordid2wordid
        self.keywordid2nodeid = keywordid2nodeid
        self.concept_encoder = concept_encoder
        self.combine_word_concepts = combine_word_concepts
        self.num_nodes = nodeid2wordid.shape[0]

        self.embedding = nn.Embedding(vocab_size, embed_size)
            
        # GNN learning
        encoder_input_size = gnn_hidden_size
        
        if gnn == "GatedGraphConv":
            self.conv1 = GatedGraphConv(gnn_hidden_size, num_layers=gnn_layers)

        if self.encoder == "GRU":
            self.utterance_encoder = nn.GRU(encoder_input_size, encoder_hidden_size, encoder_layers, batch_first=True, dropout=dropout, bidirectional=True)
            self.candidate_encoder = nn.GRU(encoder_input_size, encoder_hidden_size, encoder_layers, batch_first=True, dropout=dropout, bidirectional=True)        
    
    def init_embedding(self, embedding, fix_word_embedding):
        self.embedding.weight.data.copy_(embedding)
        if fix_word_embedding:
            self.embedding.weight.requires_grad = False

    def encode_concept(self, emb, nodeid2wordid):
        # emb: (vocab_size, emb_size)
        # nodeid2wordid: (num_nodes, 10)
        mask = nodeid2wordid.ne(0).float() # (num_nodes, 10)
        # print(emb.device, mask.device, nodeid2wordid.device)
        if self.concept_encoder == "mean":
            node_emb = (emb[nodeid2wordid] * mask.unsqueeze(-1)).sum(dim=1)/mask.sum(dim=1, keepdim=True).clamp(min=1) # (num_nodes, emb_size)
        if self.concept_encoder == "max":
            node_emb = (emb[nodeid2wordid] * mask.unsqueeze(-1) + (-5e4) * (1 - mask.unsqueeze(-1))).max(dim=1)[0] # (num_nodes, emb_size)
        return node_emb

    def encode_gnn(self, emb, edge_index):
        # emb: (num_nodes, emb_size)
        # edge_index: (2, num_edges)
        # edge_type: None or (num_edges, )
        # edge_weight: None or (num_edges, )
        if self.gnn in ["GatedGraphConv"]:
            out = self.conv1(emb, edge_index) # (num_nodes, hidden_size)
        return out


    def encode_context(self, emb, x):
        # x: (batch, context_len, seq_len)
        # print("encode context: ", x.shape)
        batch_size, context_len, seq_len = x.shape

        if self.encoder == "GRU":
            x = x.reshape(batch_size*context_len, -1) # (batch*context_len, seq_len)
            x_out = emb[x] # (batch*context_len, seq_len, embed_size)
            x_out, _ = self.utterance_encoder(x_out) # (batch*context_len, seq_len, 2*hidden_size)
            x_out = x_out.reshape(batch_size, context_len*seq_len, -1) # (batch, context_len*seq_len, 2*hidden_size)
            x_mask = x.reshape(batch_size, -1).ne(0).float() # (batch, context_len*seq_len)
        return x_out, x_mask

    def encode_candidate(self, emb, x):
        # x: (batch, num_candidates, seq_len)
        # print("encode candidate: ", x.shape)
        batch_size, num_candidates, seq_len = x.shape
        
        if self.encoder in ["GRU"]:
            x = x.reshape(batch_size*num_candidates, -1) # (batch*num_candidates, seq_len)
            x_mask = x.ne(0).float() # (batch*num_candidates, seq_len)
            x_out = emb[x] # (batch*num_candidates, seq_len, embed_size)
            x_out, _ = self.candidate_encoder(x_out) # (batch*num_candidates, seq_len, 2*hidden_size)
        return x_out, x_mask

    def encode_keywords(self, emb, x):
        # x: (batch, seq_len)
        # return: (batch, emb_size) or (batch, seq_len, emb_size) for any_max
        assert x.dim() == 2
        if self.keyword_encoder == "mean":
            x_mask = x.ne(0).float() # (batch, seq_len)
            x = emb[x] # (batch, seq_len, emb_size)
            return (x * x_mask.unsqueeze(-1)).sum(dim=1)/x_mask.sum(dim=1, keepdim=True).clamp(min=1)
        if self.keyword_encoder == "max":
            x_mask = x.ne(0).float() # (batch, seq_len)
            x = emb[x] # (batch, seq_len, emb_size)
            return torch.max(x*x_mask.unsqueeze(-1) + (-5e4)*(1-x_mask.unsqueeze(-1)), dim=1)[0]


    def aggregate(self, x, x_mask):
        # x: (batch, seq_len, emb_size)
        # x_mask: (batch, seq_len)
        # return: (batch, emb_size)
        # print("aggregate: ", x.shape, x_mask.shape)
        assert x.dim() == 3 and x_mask.dim() == 2
        batch_size, seq_len, emb_size = x.shape
        if self.aggregation == "mean":
            return (x * x_mask.unsqueeze(-1)).sum(dim=1)/x_mask.sum(dim=1, keepdim=True).clamp(min=1)
        if self.aggregation == "max":
            return torch.max(x*x_mask.unsqueeze(-1) + (-5e4)*(1-x_mask.unsqueeze(-1)), dim=1)[0]
        if self.aggregation == "last":
            x_lens = x_mask.sum(dim=1).long() # (batch, )
            return x[torch.arange(batch_size), (x_lens-1).clamp(min=0)]

    def score(self, x, y):
        # x: (batch, num_candidates, emb)
        # y: (batch, num_candidates, emb)
        # print("score: ", x.shape, y.shape)
        assert x.dim() == 3 and y.dim() == 3
        batch_size, num_candidates, emb_size = x.shape
        return torch.bmm(x.reshape(batch_size*num_candidates, 1, emb_size), y.reshape(batch_size*num_candidates, emb_size, 1)).reshape(batch_size, num_candidates) # (batch, num_candidates)


    def init_gnn_emb(self, CN_hopk_edge_index, edge_type=None):
        node_emb = self.encode_concept(self.embedding.weight, self.nodeid2wordid) # (num_nodes, emb_size)
        CN_hopk_out = self.encode_gnn(node_emb, CN_hopk_edge_index) # (num_nodes, hidden_size)
        self.CN_hopk_out = CN_hopk_out


    def encode_context_offline(self, context, context_concepts, conetxt_keywords, CN_hopk_edge_index):
        """
            context: (batch_size, context_len, seq_len), word ids
            context_concepts: (batch_size, context_len, seq_len), node ids
            conetxt_keywords: (batch_size, 3), keyword ids 
        """
        batch_size, context_len, seq_len = context.shape
        # GNN encoding
        if CN_hopk_edge_index is not None:
            # node_emb = self.encode_concept(self.embedding.weight, self.nodeid2wordid) # (num_nodes, emb_size)
            # CN_hopk_out, attn = self.encode_gnn(node_emb, CN_hopk_edge_index, edge_type, self.CN_hopk_edge_weight) # (num_nodes, hidden_size)

            # encode keywords
            if conetxt_keywords is not None:
                context_keywords_concept_out = self.encode_keywords(self.CN_hopk_out, self.keywordid2nodeid[conetxt_keywords]) # (batch, emb_size)

            # concept encoding
            context_concept_out, context_concept_mask = self.encode_context(self.CN_hopk_out, context_concepts)

        if conetxt_keywords is not None:
            # encode keywords
            context_keywords_out = self.encode_keywords(self.embedding.weight, self.keywordid2wordid[conetxt_keywords]) # (batch, emb_size)
        
        context_out, context_mask = self.encode_context(self.embedding.weight, context) # (batch, m, emb), where m can be context_len*seq_len or context_len

        context_out = torch.cat([context_out, context_concept_out], dim=1) # (batch, 2m, emb)
        context_mask = torch.cat([context_mask, context_concept_mask], dim=1) # (batch, 2m)
        context_out = self.aggregate(context_out, context_mask) # (batch, emb)
        return context_out, context_mask, None, None, context_keywords_concept_out, context_keywords_out
        

    def encode_candidate_offline(self, candidate, candidate_concepts, candidate_keywords, CN_hopk_edge_index):
        """
            candidate: (batch_size, num_candidates, seq_len), word ids
            candidate_concepts: (batch_size, num_candidates, seq_len), node ids 
            candidate_keywords: (batch_size, num_candidates, max_keyword_len), keyword ids
        """
        batch_size, num_candidates, _ = candidate.shape
        # print(len(self.keywordid2nodeid), candidate.max(), candidate_concepts.max(), candidate_keywords.max())
        # GNN encoding
        if CN_hopk_edge_index is not None:
            # node_emb = self.encode_concept(self.embedding.weight, self.nodeid2wordid) # (num_nodes, emb_size)
            # CN_hopk_out, attn = self.encode_gnn(node_emb, CN_hopk_edge_index, edge_type, self.CN_hopk_edge_weight) # (num_nodes, hidden_size)

            # encode keywords
            if candidate_keywords is not None:
                candidate_keywords_concept_out = self.encode_keywords(self.CN_hopk_out, \
                    self.keywordid2nodeid[candidate_keywords].reshape(batch_size*num_candidates, -1)) # (batch*num_candidates, emb_size)

            # concept encoding
            candidate_concept_out, candidate_concept_mask = self.encode_candidate(self.CN_hopk_out, candidate_concepts)
        
        if candidate_keywords is not None:
            # encode keywords
            candidate_keywords_out = self.encode_keywords(self.embedding.weight, \
                self.keywordid2wordid[candidate_keywords].reshape(batch_size*num_candidates, -1)) # (batch*num_candidates, emb_size)

        # encoding
        candidate_out, candidate_mask = self.encode_candidate(self.embedding.weight, candidate) # (batch*num_candidates, n, emb)

        candidate_out = torch.cat([candidate_out, candidate_concept_out], dim=1) # (batch*num_candidates, 2n, emb)
        candidate_mask = torch.cat([candidate_mask, candidate_concept_mask], dim=1) # (batch*num_candidates, 2n)
        candidate_out = self.aggregate(candidate_out, candidate_mask) # (batch*num_candidates, emb)
        return candidate_out, candidate_mask, None, None, candidate_keywords_concept_out, candidate_keywords_out


    def predict(self, context_out, context_mask, context_concept_out, context_concept_mask, context_keywords_concept_out, context_keywords_out, \
        candidate_out, candidate_mask, candidate_concept_out, candidate_concept_mask, candidate_keywords_concept_out, candidate_keywords_out):
        """
            context_out: (batch, emb)
            context_mask: (batch,)
            context_concept_out: None
            context_concept_mask: None
            context_keywords_concept_out: (batch, emb)
            context_keywords_out: (batch, emb)

            candidate_out: (batch*num_candidates, emb)
            candidate_mask: (batch*num_candidates, )
            candidate_concept_out: None
            candidate_concept_mask: None
            candidate_keywords_concept_out: (batch*num_candidates, emb)
            candidate_keywords_out: (batch*num_candidates, emb)
        """
        batch_size = context_out.shape[0]
        num_candidates = candidate_out.shape[0]//batch_size
        
        # keyword matching
        keywords_concept_score = torch.bmm(candidate_keywords_out.reshape(batch_size, num_candidates, -1), \
                        context_keywords_out.unsqueeze(-1)).squeeze(-1) # (batch, num_candidates)

        keywords_score = torch.bmm(candidate_keywords_out.reshape(batch_size, num_candidates, -1), \
                    context_keywords_out.unsqueeze(-1)).squeeze(-1) # (batch, num_candidates)
        keywords_score = (keywords_score + keywords_concept_score)/2 # overall keyword score
        
        # out = self.score(context_out.reshape(batch_size, num_candidates, -1), candidate_out.reshape(batch_size, num_candidates, -1)) # (batch, num_candidates)
        out = torch.bmm(context_out.unsqueeze(1), candidate_out.reshape(batch_size, num_candidates, -1).transpose(1,2)).squeeze(1) # (batch, num_candidates)
        out = out + self.keyword_score_weight * keywords_score
        return out # (batch, num_candidates)


    def forward(self, context, candidate, conetxt_keywords=None, candidate_keywords=None, context_concepts=None, candidate_concepts=None, \
        CN_hopk_edge_index=None):
        """
            context: (batch_size, context_len, seq_len), word ids
            candidate: (batch_size, num_candidates, seq_len), word ids
            conetxt_keywords: (batch_size, 3), keyword ids 
            candidate_keywords: (batch_size, num_candidates, max_keyword_len), keyword ids
            context_concepts: (batch_size, context_len, seq_len), node ids
            candidate_concepts: (batch_size, num_candidates, seq_len), node ids 
        """
        batch_size, num_candidates, _ = candidate.shape

        # GNN encoding
        CN_hopk_out = None
        if CN_hopk_edge_index is not None:
            node_emb = self.encode_concept(self.embedding.weight, self.nodeid2wordid) # (num_nodes, emb_size)
            CN_hopk_out = self.encode_gnn(node_emb, CN_hopk_edge_index) # (num_nodes, hidden_size)

            # encode keywords
            if conetxt_keywords is not None:
                context_keywords_out = self.encode_keywords(CN_hopk_out, self.keywordid2nodeid[conetxt_keywords]) # (batch, emb_size)
                if candidate_keywords is not None:
                    candidate_keywords_out = self.encode_keywords(CN_hopk_out, \
                        self.keywordid2nodeid[candidate_keywords].reshape(batch_size*num_candidates, -1)) # (batch*num_candidates, emb_size)
                else:
                    candidate_keywords_out = self.encode_keywords(CN_hopk_out, \
                        candidate_concepts.reshape(batch_size*num_candidates, -1)) # (batch*num_candidates, emb_size)
                keywords_concept_score = torch.bmm(candidate_keywords_out.reshape(batch_size, num_candidates, -1), \
                        context_keywords_out.unsqueeze(-1)).squeeze(-1) # (batch, num_candidates)

            # concept encoding
            context_concept_out, context_concept_mask = self.encode_context(CN_hopk_out, context_concepts)
            candidate_concept_out, candidate_concept_mask = self.encode_candidate(CN_hopk_out, candidate_concepts)
            
            # concept matching
            context_concept_out = context_concept_out.repeat_interleave(num_candidates, dim=0) # (batch*num_candidates, m, emb)
            context_concept_mask = context_concept_mask.repeat_interleave(num_candidates, dim=0) # (batch*num_candidates, m, emb)

            context_concept_out, candidate_concept_out = self.match(context_concept_out, context_concept_mask, candidate_concept_out, candidate_concept_mask)

            # aggregation
            context_concept_out = self.aggregate(context_concept_out, context_concept_mask) # (batch*num_candidates, emb)
            candidate_concept_out = self.aggregate(candidate_concept_out, candidate_concept_mask) # (batch*num_candidates, emb)

        if conetxt_keywords is not None:
            # encode keywords
            context_keywords_out = self.encode_keywords(self.embedding.weight, self.keywordid2wordid[conetxt_keywords]) # (batch, emb_size)
            if candidate_keywords is not None:
                candidate_keywords_out = self.encode_keywords(self.embedding.weight, \
                    self.keywordid2wordid[candidate_keywords].reshape(batch_size*num_candidates, -1)) # (batch*num_candidates, emb_size)
            else:
                candidate_keywords_out = self.encode_keywords(self.embedding.weight, \
                    candidate.reshape(batch_size*num_candidates, -1)) # (batch*num_candidates, emb_size)
            keywords_score = torch.bmm(candidate_keywords_out.reshape(batch_size, num_candidates, -1), \
                context_keywords_out.unsqueeze(-1)).squeeze(-1) # (batch, num_candidates)
            if CN_hopk_edge_index is not None:
                keywords_score = (keywords_score + keywords_concept_score)/2 # overall keyword score
        
        # encoding
        context_out, context_mask = self.encode_context(self.embedding.weight, context) # (batch, m, emb), where m can be context_len*seq_len or context_len
        candidate_out, candidate_mask = self.encode_candidate(self.embedding.weight, candidate) # (batch*num_candidates, n, emb)

        # matching
        context_out = context_out.repeat_interleave(num_candidates, dim=0) # (batch*num_candidates, m, emb)
        context_mask = context_mask.repeat_interleave(num_candidates, dim=0) # (batch*num_candidates, n, emb)

        # combine
        if CN_hopk_edge_index is not None:
            context_out, candidate_out = self.match(context_out, context_mask, candidate_out, candidate_mask)
        
            # aggregation
            context_out = self.aggregate(context_out, context_mask) # (batch*num_candidates, emb)
            candidate_out = self.aggregate(candidate_out, candidate_mask) # (batch*num_candidates, emb)
            if self.combine_word_concepts == "mean":
                context_out = (context_out + context_concept_out)/2
                candidate_out = (candidate_out + candidate_concept_out)/2
            if self.combine_word_concepts == "max":
                context_out = torch.stack([context_out, context_concept_out], dim=0).max(dim=0)[0]
                candidate_out = torch.stack([candidate_out, candidate_concept_out], dim=0).max(dim=0)[0]
        else:            
            # aggregation
            context_out = self.aggregate(context_out, context_mask) # (batch*num_candidates, emb)
            candidate_out = self.aggregate(candidate_out, candidate_mask) # (batch*num_candidates, emb)
        
        # scoring
        out = self.score(context_out.reshape(batch_size, num_candidates, -1), candidate_out.reshape(batch_size, num_candidates, -1)) # (batch, num_candidates)
        if conetxt_keywords is not None:
            out = out + self.keyword_score_weight * keywords_score
        return out