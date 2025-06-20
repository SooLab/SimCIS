# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import copy
import json
import logging
import math
import os
import random
from typing import Optional

import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
import torch.distributed as dist
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from .position_encoding import PositionEmbeddingSine
from .utils import box_ops
from .utils.utils import gen_encoder_output_proposals_p, inverse_sigmoid

def generate_random_bbox(n, min_wh=0.1, max_wh=0.5):
    random_wh = torch.rand(n, 2) * (max_wh - min_wh) + min_wh  # (n, 2) -> w 和 h

    random_cxcy = torch.rand(n, 2) * (1 - random_wh) + random_wh / 2  # (n, 2) -> cx 和 cy

    random_bbox = torch.cat([random_cxcy, random_wh], dim=-1)  # (n, 4)

    return random_bbox   

def sigmoid_to_logit(x):
    x = x.clamp(0.001, 0.999)
    return torch.log(x / (1-x))

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        n_cls_in_tasks: list,
        text_path: str,
        use_text_embedding: False,
        clip_embedding_dim: int,
        output_dir: str,
        collect_query_mode: bool,
        weighted_sample: bool,
        vq_number: int,
        freeze_label: bool=False,
        add_pos_to_vq: bool=False,
        distribution_alpha: float=0.5,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for i in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        # self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        self.use_text_embedding = use_text_embedding
        if self.mask_classification:
            if use_text_embedding: # we donot use text embedding in this work(SimCIS)
                # learn form https://github.com/bytedance/fc-clip/blob/main/fcclip/modeling/transformer_decoder/fcclip_transformer_decoder.py#L385
                self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
                self.dim_adaptor = MLP(hidden_dim, hidden_dim, clip_embedding_dim, 3)
                text_embedding = np.load(text_path)
                self.text_embedding = []
                for i, n_cls in enumerate(n_cls_in_tasks):
                    old_cls = np.int32(np.sum(n_cls_in_tasks[:i]))
                    self.text_embedding.append(torch.from_numpy(text_embedding[old_cls:old_cls+n_cls]))
                self.text_embedding = torch.cat(self.text_embedding, dim=0).to(torch.device('cuda'))
                self.text_embedding.requires_grad = False
            else:
                # Use static class head
                self.class_embed = nn.Linear(hidden_dim, 150)
                if freeze_label:
                    last_step_cls = sum(n_cls_in_tasks[:-1]) if len(n_cls_in_tasks) > 1 else 0
                    with torch.no_grad():
                        self.class_embed.weight[:last_step_cls].requires_grad = False
                        self.class_embed.bias[:last_step_cls].requires_grad = False
                    print(f"freeze the first {last_step_cls} classes in the class head")
                
                # Use incremental class head
                # self.class_embeds = nn.ModuleList()
                # for i, n_cls in enumerate(n_cls_in_tasks):
                #     self.class_embeds.append(nn.Linear(hidden_dim, n_cls))
                # Initialize the new linear weights
                # if len(n_cls_in_tasks) > 1:
                #     # meanHead = torch.mean(self.class_embeds[0].weight.data, dim=0, keepdim=True)
                #     selectedBysimilarity = [43, 54, 83, 76, 47, 96, 86, 50, 94, 37]
                #     for i in range(1, len(n_cls_in_tasks)):
                #         self.class_embeds[i].weight.data.copy_(self.class_embeds[0].weight.data[selectedBysimilarity])

        self.n_cls_in_tasks = torch.as_tensor(n_cls_in_tasks)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # maskdino like query_pos
        self.ref_point_head = MLP(hidden_dim * 2, hidden_dim, hidden_dim, 2)
        self.enc_output = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        # self.ref_point_head = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        self.query_scale = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        self._bbox_embed = _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        box_embed_layerlist = [_bbox_embed for _ in range(self.num_layers)]  # share box prediction each layer
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.output_dir = output_dir
        self.collect_query_mode = collect_query_mode
        self.weighted_sample = weighted_sample
        self.vq_number = vq_number

        if len(self.n_cls_in_tasks)>1 and not self.collect_query_mode[0] and self.vq_number > 0:
            print("Use PSD distribution ...")
            try:
                with open(os.path.join(output_dir, 'psd_distribution.json'), 'r') as f:
                    psd_dis = json.load(f)
                    psd_dis = torch.tensor(psd_dis[:int(self.n_cls_in_tasks.sum().item())-int(self.n_cls_in_tasks[-1].item())]) + 1
                # self.psd_dis = torch.sqrt(psd_dis.sum()/psd_dis)
                self.psd_dis = torch.pow(psd_dis.sum() / psd_dis, distribution_alpha)
            except:
                print("No PSD distribution file found, use uniform distribution")
                self.psd_dis = torch.ones(100)
        else:
            print("No PSD distribution", self.n_cls_in_tasks, self.collect_query_mode[0], type(self.collect_query_mode))
            self.psd_dis = torch.ones(100)

        self.task = len(n_cls_in_tasks)
        if self.task < 10:
            query_root = self.output_dir[:-1] + f"{self.task-1}"
        else:
            query_root = self.output_dir[:-2] + f"{self.task-1}"

        self.add_pos_to_vq = add_pos_to_vq
        # if self.task > 1:
        #     # self.query_lib = torch.load(f"{query_root}/fake_query.pkl", map_location='cpu')  # 加载到CPU
        #     with open(f"{query_root}/fake_query.pkl", 'rb') as f:
        #         self.query_lib = pickle.load(f)    
        #     # self.query_lib = torch.load(f"{query_root}/fake_query.pkl", map_location='cuda:{}'.format(dist.get_rank()))
        #     print([len(self.query_lib[q]) for q in self.query_lib])
        # else:
        #     self.query_lib = None
    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        tot_cls = cfg.CONT.TOT_CLS
        base_cls = cfg.CONT.BASE_CLS
        inc_cls = cfg.CONT.INC_CLS
        task = cfg.CONT.TASK

        num_tasks = 1 + (tot_cls - base_cls) // inc_cls
        n_cls_in_tasks = [base_cls] + [inc_cls] * (num_tasks - 1)

        ret["n_cls_in_tasks"] = n_cls_in_tasks[:task]
        ret['use_text_embedding'] = cfg.MODEL.MASK_FORMER.USE_TEXT_EMBEDDING
        ret['text_path'] = cfg.MODEL.MASK_FORMER.TEXT_PATH
        ret['clip_embedding_dim'] = cfg.MODEL.MASK_FORMER.CLIP_DIM
        ret['output_dir'] = cfg.OUTPUT_DIR
        ret["collect_query_mode"] = cfg.CONT.COLLECT_QUERY_MODE,
        ret['weighted_sample'] = cfg.CONT.WEIGHTED_SAMPLE
        ret['vq_number'] = cfg.CONT.VQ_NUMBER
        ret['freeze_label'] = cfg.CONT.FREEZE_LABEL
        ret['distribution_alpha'] = cfg.CONT.DISTRIBUTION_ALPHA
        ret['add_pos_to_vq'] = cfg.CONT.ADD_POS
        # print(f"collect_query_mode: {cfg.CONT.COLLECT_QUERY_MODE}")
        return ret

    def forward(self, x, mask_features, mask = None, distill_position = None, query_lib = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # get fake query with bs
        # fake_target = random.sample(self.fake_query.keys(), bs)
        # bad_cat = [6, 95, 8, 12, 64, 82, 36, 41, 3, 24, 63, 0, 43, 15, 84, 44, 11, 56, 89, 29, 19, 98, 32, 66, 57, 23, 16, 81, 48, 73, 39, 87, 25, 74, 38, 30, 46, 49, 13, 52, 37, 92, 69, 78, 97, 94, 34, 50, 99, 80]
        # fake_targets = random.sample([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 27, 28, 30, 31, 32, 36, 38, 39, 40, 41, 42, 43, 47, 53, 57, 66, 67, 69, 82, 85, 89, 93, 98])
        if query_lib is not None and self.vq_number > 0:
            if not self.weighted_sample:
                sampleWeight = torch.ones_like(self.psd_dis)
            else:
                sampleWeight = self.psd_dis
            fake_targets = torch.multinomial(sampleWeight, self.vq_number*bs, replacement=True)
            fake_query = []
            for i in fake_targets:
                info = random.sample(query_lib[int(i)], 1)[0]
                # fake_query.append(torch.as_tensor(info['med_feats'], device=src[0].device, dtype=torch.float32))  # 去掉外面的 []
                fake_query.append(torch.as_tensor(info, device=src[0].device, dtype=torch.float32))

            # 将列表中的张量拼接
            fake_query = torch.stack(fake_query, dim=0).reshape(bs, self.vq_number, -1)
            fake_query = fake_query.detach()
            fake_targets = fake_targets.reshape(bs, -1)
        else:
            fake_query = None
            fake_targets = None
        feats = torch.cat(src, dim=0)
        # Use anchor points
        topk = self.num_queries
        hid_dim = src[-1].shape[-1]
        output_memory, reference_point = gen_encoder_output_proposals_p(feats.transpose(0, 1), size_list, None)
        output_memory = self.encoder_norm(self.enc_output(output_memory))
        enc_outputs_coord_unselected = self._bbox_embed(
            output_memory) + reference_point  # (bs, \sum{hw}, 4) unsigmoid
        if self.use_text_embedding:
            logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
            output_memory_cls = self.dim_adaptor(self.decoder_norm(output_memory))
            # output_memory_cls = self.dim_adaptor(output_memory)
            output_memory_cls = output_memory_cls / (output_memory_cls.norm(dim=-1, keepdim=True) + 1e-7)
            enc_outputs_class_unselected = logit_scale * output_memory_cls @ self.text_embedding.T # (bs, \sum{hw}, 100)
            # enc_outputs_class_unselected[reference_point.sum(-1).isinf()] = float("-inf")
        else:
            # enc_outputs_class_unselected =torch.cat([class_embed(self.decoder_norm(output_memory)) for class_embed in self.class_embeds], dim=-1) # (bs, \sum{hw}, num_classes)
            enc_outputs_class_unselected =self.class_embed(self.decoder_norm(output_memory)) # (bs, \sum{hw}, num_classes)
            # if self.training:
            #     print("new:",enc_outputs_class_unselected[0][0])
            # else:
            #     print("old:",enc_outputs_class_unselected[0][0])
                
            # if distill_position is not None:
            #     interm_distill_logits = torch.gather(enc_outputs_class_unselected, 1, distill_position.unsqueeze(-1).repeat(1, 1, enc_outputs_class_unselected.shape[-1]))
            # else:
            #     interm_distill_logits = None
        topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]
        tgt_undetach = torch.gather(output_memory, 1,
                                  topk_proposals.unsqueeze(-1).repeat(1, 1, hid_dim))
        distill_tgt_undetach = torch.gather(output_memory, 1,
                                  distill_position.unsqueeze(-1).repeat(1, 1, hid_dim)) \
                                      if distill_position is not None else None
        # concat with fake query
        if self.training and self.task >1 and query_lib and self.vq_number > 0:
            tgt_undetach = torch.cat([tgt_undetach, fake_query], dim=1)

        refpoint_embed_unsig_undetach = torch.gather(enc_outputs_coord_unselected, 1,
                                                topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  # unsigmoid


        # Get the feats
        output = tgt_undetach.permute(1, 0, 2).detach()
        distill_output = distill_tgt_undetach.permute(1, 0, 2).detach() if distill_position is not None else None

        enc_output_class, enc_outputs_mask, attn_mask = self.forward_prediction_heads(
            tgt_undetach.transpose(0,1), mask_features, attn_mask_target_size=size_list[0]) 
        # For KD
        if distill_position is not None:
            interm_distill_logits, _, distill_attn_mask = self.forward_prediction_heads(
                distill_tgt_undetach.transpose(0,1), mask_features, attn_mask_target_size=size_list[0]) 
            distill_refpoint_embed = torch.gather(enc_outputs_coord_unselected, 1,
                                                topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).\
                                                    sigmoid().transpose(0, 1).detach()  

        refpoint_embed = refpoint_embed_unsig_undetach.sigmoid().transpose(0, 1).detach() # bs, topk, 4
        if self.training and self.task >1 and self.vq_number > 0 and query_lib:
            refpoint_embed = F.pad(refpoint_embed, (0, 0, 0, 0, 0, self.vq_number))
            # random_bboxes = generate_random_bbox(self.vq_number).unsqueeze(1).repeat(1, refpoint_embed.shape[1], 1)  # (bs, self.vq_number, 4)
            # random_bboxes = random_bboxes.to(refpoint_embed.device)
            # Concatenate randomly generated bbox with refpoint_embed
            # refpoint_embed = torch.cat([refpoint_embed, random_bboxes], dim=0)

        interm_outputs=dict()
        interm_outputs['pred_logits'] = enc_output_class
        interm_outputs['pred_masks'] = enc_outputs_mask
        interm_outputs['pred_boxes'] = F.pad(refpoint_embed_unsig_undetach.sigmoid(), (0,0,0,self.vq_number,0,0)) \
            if self.task > 1 and self.vq_number > 0 else refpoint_embed_unsig_undetach.sigmoid()
        results = self.forward_decoder(
                output=output, 
                mask_features=mask_features, 
                refpoint_embed=refpoint_embed, 
                pos=pos, src=src, 
                attn_mask=attn_mask, 
                size_list=size_list, 
                query_lib=query_lib
            )
        predictions_class, predictions_mask, predictions_box = results[0], results[1], results[2]
        if distill_position is not None:
            distill_results = self.forward_decoder(
                    output=distill_output, 
                    mask_features=mask_features, 
                    refpoint_embed=distill_refpoint_embed, 
                    pos=pos, src=src, 
                    attn_mask=distill_attn_mask, 
                    size_list=size_list, 
                    query_lib=None,
                )
            distill_info = {
                'pred_logits': distill_results[0][-1],
                'aux_outputs': self._set_aux_loss(
                    distill_results[0], distill_results[1], distill_results[2]
                ),
                'interm_outputs': {'pred_logits' : interm_distill_logits}
            }

        if not self.training:
            out = {
                'pred_logits': predictions_class[-1],
                'pred_masks': predictions_mask[-1],
                'pred_boxes': predictions_box[-1],
                'aux_outputs': self._set_aux_loss(
                    predictions_class if self.mask_classification else None, predictions_mask, predictions_box
                ),
                'interm_outputs': interm_outputs,
                'topk_feats_info': {'topk_proposals':topk_proposals, 'med_feats':tgt_undetach, 'class_logits':enc_output_class},
                'med_feats_info': {'flatten_feats':feats.transpose(0, 1), 'feats_logits':enc_outputs_class_unselected },
            }
            return out
        else:
            out = {
                'pred_logits': predictions_class[-1],
                'pred_masks': predictions_mask[-1],
                'pred_boxes': predictions_box[-1],
                'aux_outputs': self._set_aux_loss(
                    predictions_class if self.mask_classification else None, predictions_mask, predictions_box
                ),
                'interm_outputs': interm_outputs,
                'topk_feats_info': {'topk_proposals':topk_proposals, 'med_feats':tgt_undetach, 'class_logits':enc_output_class},
                'med_feats_info': {'flatten_feats':feats.transpose(0, 1), 'feats_logits':enc_outputs_class_unselected },
                'distill_info': distill_info if distill_position is not None else None,
            }
            
            return out, fake_targets

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        bs,_, _ = decoder_output.shape
        if self.use_text_embedding:
            cls_decoder = self.dim_adaptor(decoder_output)
            norm_decoder = cls_decoder / (cls_decoder.norm(dim=-1, keepdim=True) + 1e-7)
            logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
            outputs_class = logit_scale*norm_decoder @ self.text_embedding.T
        else:
            # outputs_class = torch.cat([class_embed(decoder_output) for class_embed in self.class_embeds], dim=-1) # (bs, \sum{hw}, num_classes)
            outputs_class = self.class_embed(decoder_output)
            # outputs_class = torch.cat((-torch.ones((bs, 100,100), device=outputs_class.device)*100, outputs_class), dim=-1)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask
    def _gen_sineembed_for_position(self, pos_tensor):
        # n_query, bs, _ = pos_tensor.size()
        # sineembed_tensor = torch.zeros(n_query, bs, 256)
        scale = 2 * math.pi
        dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
        dim_t = 10000 ** (2 * torch.div(dim_t, 2,rounding_mode='trunc') / 128)
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        if pos_tensor.size(-1) == 2:
            pos = torch.cat((pos_y, pos_x), dim=2)
        elif pos_tensor.size(-1) == 4:
            w_embed = pos_tensor[:, :, 2] * scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

            h_embed = pos_tensor[:, :, 3] * scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        else:
            raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
        return pos
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, out_boxes=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            if out_boxes is None:
                return [
                    {"pred_logits": a, "pred_masks": b}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
            else:
                return [
                    {"pred_logits": a, "pred_masks": b, "pred_boxes": c}
                    for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1])
                ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
    
    def check_logits(self, logits):
        score, label = logits.max(-1)
        outrange_mask = (label >= self.n_cls_in_tasks.sum()) | (label < 0)
        outNum = torch.sum(outrange_mask)
        return outNum, outrange_mask

    def forward_decoder(self, output, mask_features, refpoint_embed, pos, src, attn_mask, size_list, query_lib):
        predictions_class = []
        predictions_mask = []
        predictions_box = []
        # save = None
        for i in range(self.num_layers):
            query_sine_embed = self._gen_sineembed_for_position(refpoint_embed) # nq, bs, 256*2

            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            query_embed = query_pos
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            if self.training and self.task >1 and query_lib:
                fake_query_embed = output[-self.vq_number:,:] #.unsqueeze(0) # x * bs * dim
                if self.add_pos_to_vq and i == 0:
                    fake_query_embed = output[-self.vq_number:, :] + query_embed[-self.vq_number:, :]

                output = self.transformer_self_attention_layers[i](
                    output[:-self.vq_number], tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed[:-self.vq_number]
                )
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask[:,:-self.vq_number,:],
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed[:-self.vq_number]
                )
                
                output = torch.cat([output, fake_query_embed], dim=0)
            else:
                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(refpoint_embed)
                delta_unsig = self.bbox_embed[i](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                refpoint_embed = new_reference_points.detach()
                # if layer_id != self.num_layers - 1:
                predictions_box.append(new_reference_points.transpose(0, 1))
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            
            # test
        #     temp = torch.max(outputs_mask.flatten(2), dim=-1)[0] # [B, 100]
        #     if i == 0:
        #         save = temp
        #     else:
        #         save = torch.cat((save, temp), dim=0)

        # try:
        #     matrix = np.load('save.npy')
        #     save = save.cpu().numpy()
        #     matrix = np.concatenate((matrix, save[None,...]), axis=0)
        #     np.save('save.npy', matrix)
        # except:
        #     np.save('save.npy', save.cpu().numpy()[None,...])
            
            
        # print(f"save shape: {save.shape}")

        return (predictions_class, predictions_mask, predictions_box)

