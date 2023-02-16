"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for ITM model
"""
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from .model import UniterPreTrainedModel, UniterModel
from .ot import optimal_transport_dist


class UniterForImageTextRetrieval(UniterPreTrainedModel):
    """ Finetune UNITER for image text retrieval
    """
    def __init__(self, config, img_dim, margin=0.2):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.itm_output = nn.Linear(config.hidden_size, 2)
        self.rank_output = nn.Linear(config.hidden_size, 1)
        self.margin = margin
        self.apply(self.init_weights)

    def init_output(self):
        """ need to be called after from pretrained """
        self.rank_output.weight.data = self.itm_output.weight.data[1:, :]
        self.rank_output.bias.data = self.itm_output.bias.data[1:]

    def forward(self, batch, compute_loss=True, mode='word', return_attn=False):
    # def forward(self, input_ids, position_ids, img_feat, img_pos_feat,
    #                     attention_mask, gather_index, targets, ot_inputs,
    #                     compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']

        targets = batch['targets']
        ot_inputs = batch['ot_inputs']
        out = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False, return_attn=return_attn)
        if return_attn:
            sequence_output, attn = out
        else:
            sequence_output = out
        pooled_output = self.uniter.pooler(sequence_output)
        rank_scores = self.rank_output(pooled_output)

        # if compute_loss:
        #     # triplet loss
        #     rank_scores_sigmoid = torch.sigmoid(rank_scores)
        #     # sample_size = batch['sample_size']
        #     sample_size = len(input_ids)
        #     scores = rank_scores_sigmoid.contiguous().view(-1, sample_size)
        #     pos = scores[:, :1]
        #     neg = scores[:, 1:]
        #     rank_loss = torch.clamp(self.margin + neg - pos, 0)
        #     return rank_loss
        # else:
        #     return rank_scores

        # OT loss
        pooled_output = self.uniter.pooler(sequence_output)
        itm_scores = self.itm_output(pooled_output)
        if ot_inputs is not None:
            ot_scatter = ot_inputs['ot_scatter']

            b = sequence_output.size(0)
            tl = input_ids.size(1)
            il = img_feat.size(1)
            max_l = max(ot_inputs['scatter_max'] + 1, tl+il)

            ot_scatter = ot_scatter.unsqueeze(-1).expand_as(sequence_output)
            ctx_emb = torch.zeros(b, max_l, self.config.hidden_size,
                                  dtype=sequence_output.dtype,
                                  device=sequence_output.device
                                  ).scatter_(dim=1, index=ot_scatter,
                                             src=sequence_output)
            txt_emb = ctx_emb[:, :tl, :]
            img_emb = ctx_emb[:, tl:tl+il, :]

            txt_pad = ot_inputs['txt_pad']
            img_pad = ot_inputs['img_pad']


            # our method: phrase ot 
            if mode == 'phrase':
                phrases = ot_inputs['phrases']
                phrase_pad = ot_inputs['phrase_pad']

                phrase_num, phrase_len = phrases.shape[1], phrases.shape[2]
                emb_size = txt_emb.shape[-1]
                phrases0 = phrases.reshape(b,-1).unsqueeze(-1).expand(b, phrase_num*phrase_len, emb_size) + 1
                txt_emb_pad = torch.cat([ torch.mean(txt_emb, dim=1).unsqueeze(1).detach(), txt_emb ], dim=1)

                phrase_emb = txt_emb_pad.gather(1,phrases0)
                phrase_emb = phrase_emb.reshape(b, phrase_num, phrase_len, emb_size)
                phrase_emb = torch.mean(phrase_emb, dim=2)

                ot_dist, T = optimal_transport_dist(phrase_emb.float(), img_emb.float(), phrase_pad, img_pad)
                ot_dist = ot_dist.to(phrase_emb)
            else:
                # # NOTE: run in fp32 for stability
                ot_dist, T = optimal_transport_dist(txt_emb.float(), img_emb.float(), txt_pad, img_pad)
                ot_dist = ot_dist.to(txt_emb)

            if compute_loss:
                ot_pos_dist = ot_dist.masked_select(targets == 1)
                ot_neg_dist = ot_dist.masked_select(targets == 0)
                ot_loss = (ot_pos_dist, ot_neg_dist)
        else:
            ot_loss = None

        if compute_loss:
            itm_loss = F.cross_entropy(itm_scores, targets, reduction='none')
            return itm_loss, ot_loss
        else:
            if return_attn:
                return rank_scores, sequence_output, T
            return rank_scores


class UniterForImageTextRetrievalHardNeg(UniterForImageTextRetrieval):
    """ Finetune UNITER for image text retrieval
    """
    def __init__(self, config, img_dim, margin=0.2, hard_size=16):
        super().__init__(config, img_dim, margin)
        self.hard_size = hard_size

    def forward(self, batch, sample_from='t', compute_loss=True):
        # expect same input_ids for all pairs
        batch_size = batch['attn_masks'].size(0)
        input_ids = batch['input_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        if sample_from == 't':
            if input_ids.size(0) == 1:
                batch['input_ids'] = input_ids.expand(batch_size, -1)
        elif sample_from == 'i':
            if img_feat.size(0) == 1:
                batch['img_feat'] = img_feat.expand(batch_size, -1, -1)
            if img_pos_feat.size(0) == 1:
                batch['img_pos_feat'] = img_pos_feat.expand(batch_size, -1, -1)
        else:
            raise ValueError()

        if self.training and compute_loss:
            with torch.no_grad():
                self.eval()
                scores = super().forward(batch, compute_loss=False)
                hard_batch = self._get_hard_batch(batch, scores, sample_from)
                self.train()
            return super().forward(hard_batch, compute_loss=True)
        else:
            return super().forward(batch, compute_loss)

    def _get_hard_batch(self, batch, scores, sample_from='t'):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        hard_batch = {'sample_size': self.hard_size + 1}

        # NOTE first example is positive
        hard_indices = scores.squeeze(-1)[1:].topk(
            self.hard_size, sorted=False)[1] + 1
        indices = torch.cat([torch.zeros(1, dtype=torch.long,
                                         device=hard_indices.device),
                             hard_indices])

        attention_mask = attention_mask.index_select(0, indices)
        gather_index = gather_index.index_select(0, indices)
        if position_ids.size(0) != 1:
            position_ids = position_ids[:self.hard_size+1]

        if sample_from == 't':
            # cut to minimum padding
            max_len = attention_mask.sum(dim=1).max().item()
            max_i = max_len - input_ids.size(1)
            attention_mask = attention_mask[:, :max_len]
            gather_index = gather_index[:, :max_len]
            img_feat = img_feat.index_select(0, indices)[:, :max_i, :]
            img_pos_feat = img_pos_feat.index_select(0, indices)[:, :max_i, :]
            # expect same input_ids for all pairs
            input_ids = input_ids[:self.hard_size+1]
        elif sample_from == 'i':
            input_ids = input_ids.index_select(0, indices)
            # expect same image features for all pairs
            img_feat = img_feat[:self.hard_size+1]
            img_pos_feat = img_pos_feat[:self.hard_size+1]
        else:
            raise ValueError()

        hard_batch['input_ids'] = input_ids
        hard_batch['position_ids'] = position_ids
        hard_batch['img_feat'] = img_feat
        hard_batch['img_pos_feat'] = img_pos_feat
        hard_batch['attn_masks'] = attention_mask
        hard_batch['gather_index'] = gather_index

        return hard_batch
