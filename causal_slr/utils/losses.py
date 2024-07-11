import torch
import math
_LOG_2PI = math.log(2 * math.pi)


def gaussian_nll_loss(pred, target, with_logvar=False,
                      fixed_variance=None, detach_mean=False,
                      detach_var=False):
    mean = pred[0]
    if detach_mean:
        mean = mean.detach()

    if with_logvar:
        logvar = pred[1]
        if detach_var:
            logvar = logvar.detach()

        if fixed_variance is not None:
            logvar = torch.ones_like(mean) * math.log(fixed_variance)
        ll = -0.5 * ((target - mean)**2 * (-logvar).exp() + logvar + _LOG_2PI)
    else:
        var = pred[1]
        if detach_var:
            var = var.detach()

        if fixed_variance is not None:
            var = torch.ones_like(mean) * fixed_variance
        ll = -0.5 * ((target - mean)**2 / var + torch.log(var) +
                     _LOG_2PI)  # gaussian log likelihood (to be max)

    return -torch.sum(ll, axis=-1)  # we minimize negative loglikehood


def beta_nll_loss(pred, target, beta=0.5):
    """Compute beta-NLL loss

    :param mean: Predicted mean of shape B x D
    :param variance: Predicted variance of shape B x D
    :param target: Target of shape B x D
    :param beta: Parameter from range [0, 1] controlling relative 
        weighting between data points, where `0` corresponds to 
        high weight on low error points and `1` to an equal weighting.
    :returns: Loss per batch element of shape B
    """

    mean = pred[0]
    variance = pred[1]

    loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())

    if beta > 0:
        loss = loss * (variance.detach() ** beta)

    return loss.sum(axis=-1)
