import torch
import numpy as np
from misc.lr_scheduler import LRSchedulerWithWarmup


def interpolate_text(pos_embed_checkpoint, target_dim=77):
    # (n_ctx, n_feat) for pos_embed_checkpoint, including SOT and EOT
    if pos_embed_checkpoint.size(0) == target_dim:
        return pos_embed_checkpoint
    start_token = pos_embed_checkpoint[:1, :]
    end_token = pos_embed_checkpoint[-1:, :]
    pos_tokens = pos_embed_checkpoint[1:-1, :].unsqueeze(0).permute(0, 2, 1)
    pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=target_dim-2, mode='linear')
    pos_tokens = pos_tokens.squeeze(0).t()
    pos_tokens = torch.cat([start_token, pos_tokens, end_token], dim=0)
    return pos_tokens


def load_checkpoint(model, config):
    if config.model.ckpt_type == 'original_clip':
        with open(config.model.checkpoint, 'rb') as opened_file:
            model_tmp = torch.jit.load(opened_file, map_location="cpu")
            state = model_tmp.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in state:
                del state[key]

        # 2 towers in new_state: visual, encode_text
        new_state = {}
        for name, params in state.items():
            if name == 'positional_embedding':
                new_state['encode_text.' + name] = interpolate_text(params, config.experiment.text_length)
            elif name.startswith('transformer') or name in ['positional_embedding', 'token_embedding.weight',
                                                    'ln_final.weight', 'ln_final.bias', 'text_projection']:
                new_state['encode_text.' + name] = params
            else:
                new_state[name] = params
    elif config.model.ckpt_type == 'saved':
        ckpt = torch.load(config.model.saved_path, map_location='cpu')
        new_state = ckpt['model']
    else:
        raise KeyError

    load_result = model.load_state_dict(new_state, strict=False)
    return model, load_result


def cosine_scheduler(config):
    schedule_config = config.schedule
    base_value = schedule_config.lr
    start_warmup_value = schedule_config.lr_start
    final_value = schedule_config.lr_end
    epochs = schedule_config.epoch
    warmup_epochs = schedule_config.epoch_warmup
    niter_per_ep = schedule_config.niter_per_ep

    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


# def build_optimizer(config, model):
#     p_wd, p_non_wd = [], []
#     for n, p in model.named_parameters():
#         if not p.requires_grad:
#             continue  # frozen weights
#         if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
#             p_non_wd.append(p)
#         else:
#             p_wd.append(p)
#
#     schedule_config = config.schedule
#     optim_params = [{"params": p_wd, "weight_decay": schedule_config.weight_decay, "ratio": 1.},
#                     {"params": p_non_wd, "weight_decay": 0, "ratio": 1.}]
#
#     optimizer = torch.optim.AdamW(optim_params, lr=schedule_config.lr, betas=schedule_config.betas,
#                                   eps=schedule_config.eps, weight_decay=schedule_config.weight_decay)
#     return optimizer


def build_optimizer(config, model):
    params = []
    schedule_config = config.schedule
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        weight_decay = schedule_config.weight_decay
        ratio = 1.

        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            weight_decay = 0.
        if "cross" in n or "classifier" in n or "mlm_head" in n:
            ratio = ratio * schedule_config.ratio_factor  # default 5.0

        params += [{"params": [p], "weight_decay": weight_decay, "ratio": ratio}]

    optimizer = torch.optim.AdamW(params, lr=schedule_config.lr, betas=schedule_config.betas,
                                  eps=schedule_config.eps, weight_decay=schedule_config.weight_decay)
    return optimizer
