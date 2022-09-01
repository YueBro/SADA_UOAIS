from .mmd import MMD, MMD_single_batch
from .kl_loss import kl_loss, kl_loss_single_batch


__all__ = [
    'build_domain_loss_fn',
    'build_combined_batch_loss',
    'build_combined_batch_and_single_loss',
]


def build_domain_loss_fn(cfg):
    loss_type = cfg.DOMAIN_ADAPTATION.LOSS_TYPE
    if loss_type == 'MMD':
        return MMD
    elif loss_type == 'KL_LOSS':
        return kl_loss
    elif loss_type == 'both':
        return build_combined_batch_loss()
    else:
        raise ValueError(f'Unrecognized domain loss type "{cfg.DOMAIN_ADAPTATION.LOSS_TYPE}". Only the following losses are allowed: MMD, KL_LOSS')


def build_combined_batch_loss(mmd_sclae=1.0, kll_scale=1.0):
    return lambda x, y: mmd_sclae*MMD(x,y) + kll_scale*kl_loss(x,y)


def build_combined_single_loss(mmd_sclae=1.0, kll_scale=1.0):
    return lambda x, y: mmd_sclae*MMD_single_batch(x,y) + kll_scale*kl_loss_single_batch(x,y)


def build_combined_batch_and_single_loss(mmd_sclae=1.0, kll_scale=1.0, batch_scale=1.0, single_batch_scale=0.3):
    return lambda x, y: \
        (mmd_sclae*MMD(x,y) + kll_scale*kl_loss(x,y)) * batch_scale + \
        (mmd_sclae*MMD_single_batch(x,y) + kll_scale*kl_loss_single_batch(x,y)) * single_batch_scale
