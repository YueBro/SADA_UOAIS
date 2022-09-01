import torch
from numpy import prod


__all__ = [
    'kl_loss',
    'kl_loss_single_batch',
]


DELTA = 1e-7


def kl_loss(x: torch.Tensor, y: torch.Tensor):

    dim1_1, dim1_2 = x.shape[0], y.shape[0]
    if dim1_1 == 0 or dim1_2 == 0:
        return torch.tensor(0.0).to(x.device)
    else:
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        
        depth = prod(x.shape[1:])
        x = x.view(dim1_1, 1, depth)
        y = y.view(1, dim1_2, depth)
        x = x.expand(dim1_1, dim1_2, depth)
        y = y.expand(dim1_1, dim1_2, depth)

        x = x.sigmoid().clip(DELTA, 1-DELTA)
        y = y.sigmoid().clip(DELTA, 1-DELTA)
        # x = x.sigmoid()*(1-2*DELTA)+DELTA
        # y = y.sigmoid()*(1-2*DELTA)+DELTA
        x_flip = 1 - x
        y_flip = 1 - y

        kl_div_cls0 = y * (y/x).log()
        kl_div_cls1 = y_flip * (y_flip/x_flip).log()
        kl_div = (kl_div_cls0 + kl_div_cls1)
        # kl_div /= 2
        loss = torch.mean(kl_div)

        return loss


def kl_loss_single_batch(x: torch.Tensor, y: torch.Tensor):
    assert x.shape==y.shape

    dim1_1, dim1_2 = x.shape[0], y.shape[0]
    if dim1_1 == 0 or dim1_2 == 0:
        return torch.tensor(0.0).to(x.device)
    else:
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        
        x = x.sigmoid().clip(DELTA, 1-DELTA)
        y = y.sigmoid().clip(DELTA, 1-DELTA)
        x_flip = 1 - x
        y_flip = 1 - y

        kl_div_cls0 = y * (y/x).log()
        kl_div_cls1 = y_flip * (y_flip/x_flip).log()
        kl_div = (kl_div_cls0 + kl_div_cls1)
        # kl_div /= 2
        loss = torch.mean(kl_div)

        return loss
