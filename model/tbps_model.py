import random
import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import copy

from misc import utils
from misc.utils import is_using_distributed
from text_utils.tokenizer import tokenize
from .loss import compute_simclr_loss
from .visual_transformer import visual_transformer
from .text_transformer import text_transformers
from .eda import EDA
from .base_transformer import Transformer, LayerNorm, QuickGELU

from .shared_modules import AllGather
from collections import OrderedDict


class CLIP(nn.Module):
    def __init__(self, config, image_encode, text_encode, num_classes=11003, eps=1e-2):
        super().__init__()
        self.visual = image_encode
        self.encode_text = text_encode
        self.embed_dim = config.model.embed_dim

        self.use_gather = config.model.use_gather
        self.logit_scale = nn.Parameter(torch.ones([]))
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        self.config = config
        self.eda = EDA()
        self.eps = eps

        if config.experiment.ss:
            structure = config.experiment.simclr_mlp
            self.simclr_mlp = self._build_mlp(*structure)

        if config.experiment.id:
            self.classifier = nn.Linear(self.embed_dim, num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if config.experiment.mlm:
            self.vocab_size = config.model.vocab_size
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=config.experiment.cmt_depth,
                                                       heads=self.embed_dim // 64)
            scale = self.cross_modal_transformer.width ** -0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                             ('gelu', QuickGELU()),
                             ('ln', LayerNorm(self.embed_dim)),
                             ('fc', nn.Linear(self.embed_dim, self.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def forward(self, input, alpha):
        ret = dict()

        images = input['image'].to(self.config.device)
        images_1 = input['aug1'].to(self.config.device)
        texts = input['caption']
        texts_bt = input['caption_bt']

        # back translation
        if self.config.experiment.back_trans:
            for i in range(len(texts)):
                if random.random() < self.config.experiment.backtrans_p:
                    texts[i] = texts_bt[i]

        # random deletion
        cap_new = []
        for text in texts:
            eda_alpha = self.config.experiment.eda_alpha
            cap_new.append(self.eda.random_deletion(text, eda_alpha))
        texts = cap_new

        # MLM
        if self.config.experiment.mlm:
            text_tokens, mlm_labels = tokenize(texts, context_length=self.config.experiment.text_length,
                                               mask_type='MLM')
            text_tokens = text_tokens.to(self.config.device)
            mlm_labels = mlm_labels.to(self.config.device)
        else:
            text_tokens = tokenize(texts, context_length=self.config.experiment.text_length).to(self.config.device)
        ids = input['id'].to(self.config.device)

        image_features, image_seq_embeddings = self.encode_image(images, return_dense=True)
        text_features, text_seq_embeddings = self.encode_text(text_tokens, return_dense=True)
        image_features_norm = F.normalize(image_features)
        text_features_norm = F.normalize(text_features)
        image_features_norm_gathered = self.all_gather(image_features_norm)
        text_features_norm_gathered = self.all_gather(text_features_norm)

        # image ss
        if self.config.experiment.ss:
            aug1_embed = self.simclr_mlp(self.encode_image(input['aug_ss_1'].to(self.config.device)))
            aug2_embed = self.simclr_mlp(self.encode_image(input['aug_ss_2'].to(self.config.device)))
            q_a = F.normalize(aug1_embed, dim=-1, p=2)
            q_b = F.normalize(aug2_embed, dim=-1, p=2)
            local_batch_size = q_a.size(0)
            labels = local_batch_size * utils.get_rank() + torch.arange(local_batch_size, device=q_a.device)
            k_a = self.all_gather(q_a)
            k_b = self.all_gather(q_b)
            ss_loss = compute_simclr_loss(q_a, q_b, k_a, k_b, labels, self.config.experiment.simclr_temperature)
            ret['ss_loss'] = ss_loss * self.config.experiment.ss_ratio

        logit_scale = self.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)

        idx = ids.view(-1, 1)
        gathered_ids = self.all_gather(ids)
        idx_all = gathered_ids.view(1, -1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        with torch.no_grad():
            image_features_s = self.encode_image(images).detach()
            text_features_s = self.encode_text(text_tokens).detach()
            image_features_s_norm = F.normalize(image_features_s)
            text_features_s_norm = F.normalize(text_features_s)
            image_features_s_norm_gathered = self.all_gather(image_features_s_norm)
            text_features_s_norm_gathered = self.all_gather(text_features_s_norm)
        nitc_loss = self.calc_contrastive(image_features_norm, text_features_norm, image_features_s_norm,
                                          text_features_s_norm,
                                          image_features_norm_gathered, text_features_norm_gathered,
                                          image_features_s_norm_gathered, text_features_s_norm_gathered,
                                          sim_targets, alpha, logit_scale)

        if self.config.experiment.mvs_image:
            image_1_features = self.encode_image(images_1)
            image_1_features_norm = F.normalize(image_1_features)
            image_1_features_norm_gathered = self.all_gather(image_1_features_norm)
            with torch.no_grad():
                image_1_features_s = self.encode_image(images_1).detach()
                image_1_features_s_norm = F.normalize(image_1_features_s)
                image_1_features_s_norm_gathered = self.all_gather(image_1_features_s_norm)
            loss_img1_txt0 = self.calc_contrastive(image_1_features_norm, text_features_norm, image_1_features_s_norm,
                                                   text_features_s_norm,
                                                   image_1_features_norm_gathered, text_features_norm_gathered,
                                                   image_1_features_s_norm_gathered, text_features_s_norm_gathered,
                                                   sim_targets, alpha, logit_scale)
            nitc_loss = (nitc_loss + loss_img1_txt0) / 2

        ret['nitc_loss'] = nitc_loss * self.config.experiment.nitc_ratio

        if self.config.experiment.citc:
            logits_image_per_image = logit_scale * image_features_norm_gathered @ image_features_norm_gathered.t()
            logits_text_per_text = logit_scale * text_features_norm_gathered @ text_features_norm_gathered.t()
            inmodal_cyclic_loss = (logits_image_per_image - logits_text_per_text).square().mean() / (
                    logit_scale * logit_scale)
            logits_text_per_image = logit_scale * image_features_norm_gathered @ text_features_norm_gathered.t()
            logits_image_per_text = logit_scale * text_features_norm_gathered @ image_features_norm_gathered.t()
            crossmodal_cyclic_loss = (logits_text_per_image - logits_image_per_text).square().mean() / (
                    logit_scale * logit_scale)
            citc_loss = self.config.experiment.citc_lambda1 * inmodal_cyclic_loss + self.config.experiment.citc_lambda2 * crossmodal_cyclic_loss
            ret['citc_loss'] = citc_loss * self.config.experiment.citc_ratio

        if self.config.experiment.ritc:
            logits_per_image_1 = logit_scale * image_features_norm @ text_features_norm_gathered.t()
            logits_per_text_1 = logit_scale * text_features_norm @ image_features_norm_gathered.t()
            img_log = F.log_softmax(logits_per_image_1, dim=1)
            txt_log = F.log_softmax(logits_per_text_1, dim=1)
            target_log = (sim_targets + self.eps).log()
            kl_img = F.kl_div(target_log, img_log, log_target=True, reduction='batchmean')
            kl_txt = F.kl_div(target_log, txt_log, log_target=True, reduction='batchmean')
            ritc_loss = 0.5 * (kl_img + kl_txt)
            ret['ritc_loss'] = ritc_loss * self.config.experiment.ritc_ratio

        if self.config.experiment.mlm:
            x = self.cross_former(text_seq_embeddings, image_seq_embeddings, image_seq_embeddings)
            x = self.mlm_head(x)
            scores = x.float().reshape(-1, self.vocab_size)
            mlm_labels = mlm_labels.reshape(-1)
            mlm_loss = F.cross_entropy(scores, mlm_labels)
            ret['mlm_loss'] = mlm_loss * self.config.experiment.mlm_ratio

        if self.config.experiment.id:
            image_logits = self.classifier(image_features)
            text_logits = self.classifier(text_features)
            id_loss = (F.cross_entropy(image_logits, ids) + F.cross_entropy(text_logits, ids)) / 2
            ret['id_loss'] = id_loss * self.config.experiment.id_ratio

        return ret

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    # input features are normed
    def calc_contrastive(self, image_features, text_features, image_features_s, text_features_s,
                         image_features_gathered, text_features_gathered, image_features_s_gathered,
                         text_features_s_gathered,
                         sim_targets, alpha, logit_scale):
        with torch.no_grad():
            sim_i2t_s = logit_scale * image_features_s @ text_features_s_gathered.t()
            sim_t2i_s = logit_scale * text_features_s @ image_features_s_gathered.t()
            sim_i2t_targets = alpha * F.softmax(sim_i2t_s, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_s, dim=1) + (1 - alpha) * sim_targets  # soft + hard
        sim_i2t = logit_scale * image_features @ text_features_gathered.t()
        sim_t2i = logit_scale * text_features @ image_features_gathered.t()
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        loss_ita = (loss_i2t + loss_t2i) / 2
        return loss_ita

    def compute_simclr_loss(self, logits_a, logits_b, logits_a_gathered, logits_b_gathered, labels, temperature):
        sim_aa = logits_a @ logits_a_gathered.t() / temperature
        sim_ab = logits_a @ logits_b_gathered.t() / temperature
        sim_ba = logits_b @ logits_a_gathered.t() / temperature
        sim_bb = logits_b @ logits_b_gathered.t() / temperature
        masks = torch.where(F.one_hot(labels, logits_a_gathered.size(0)) == 0, 0, float('-inf'))
        sim_aa += masks
        sim_bb += masks
        sim_a = torch.cat([sim_ab, sim_aa], 1)
        sim_b = torch.cat([sim_ba, sim_bb], 1)
        loss_a = F.cross_entropy(sim_a, labels)
        loss_b = F.cross_entropy(sim_b, labels)
        return (loss_a + loss_b) * 0.5

    def _build_mlp(self, in_dim=512, mlp_dim=512, out_dim=512):
        return nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, out_dim)
        )

    @property
    def dtype(self):
        try:
            return self.visual.conv1.weight.dtype
        except:
            try:
                return self.visual.head.weight.dtype
            except:
                try:
                    return self.visual.stem[0].weight.dtype
                except:
                    return self.encode_text.text_projection.weight.dtype

    def encode_image(self, image, return_dense=False):
        if return_dense:
            output = self.visual(image.type(self.dtype), return_dense=return_dense)
            return output
        output = self.visual(image.type(self.dtype))
        return output

    def all_gather(self, input):
        if not self.use_gather or not is_using_distributed():
            return input
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output


def clip_vitb(config, num_classes=11003):
    image_encode = visual_transformer(config)
    text_encode = text_transformers(config)
    model = CLIP(config, image_encode, text_encode, num_classes, config.experiment.ritc_eps)
    return model
