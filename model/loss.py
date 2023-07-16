import torch
import torch.nn.functional as F


def compute_simclr_loss(logits_a, logits_b, logits_a_gathered, logits_b_gathered, labels, temperature):
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
