import os

import torch
from torch import nn
import torch.distributed as dist


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.rank = int(os.environ['RANK'])
        ctx.world_size = int(os.environ['WORLD_SIZE'])

        #         y = tensor.new(ctx.world_size, *tensor.size())

        y = [tensor.new(*tensor.size()) for _ in range(ctx.world_size)]

        dist.all_gather(y, tensor.contiguous())

        y = torch.cat(y, 0).view(-1, *tensor.size())

        return y

    @staticmethod
    def backward(ctx, grad_output):
        in_grad = torch.zeros_like(grad_output)
        in_grad.copy_(grad_output)
        # sum grad for gathered tensor
        dist.all_reduce(in_grad.contiguous())
        # split
        return in_grad[ctx.rank]

