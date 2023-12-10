import torch.nn as nn
from metrics import MMD

import torch
def lr_scheduler(loss, ideal_loss, x_min=0.1, x_max=0.1, h_min=0.1, f_max=2.0):
    """
    Gap-aware Learning Rate Scheduler for Adversarial Networks.

    Args:
        loss (torch.Tensor): the loss of the discriminator D on the training data.
        ideal_loss (float): the ideal loss of D.
        x_min (float): the value of x at which the scheduler achieves its minimum allowed value h_min.
        x_max (float): the value of x at which the scheduler achieves its maximum allowed value f_max.
        h_min (float, optional): a scalar in (0, 1] denoting the minimum allowed value of the scheduling function.
                                Defaults to 0.1.
        f_max (float, optional): a scalar (>= 1) denoting the maximum allowed value of the scheduling function.
                                Defaults to 2.0.

    Returns:
        torch.Tensor: A scalar in [h_min, f_max], which can be used as a multiplier for the learning rate of D.
    """

    x = torch.abs(torch.tensor(loss - ideal_loss))
    f_x = torch.clamp(torch.pow(f_max, x / x_max), 1.0, f_max)
    h_x = torch.clamp(torch.pow(h_min, x / x_min), h_min, 1.0)

    # Use a conditional statement to handle the loss comparison
    if loss >= ideal_loss:
        return f_x
    else:
        return h_x


def compute_ind_disc_R1_loss(model, x, y):
    x = x.detach().clone()
    x.requires_grad_()
    pred_real = model(x).sum()
    grad_real, = torch.autograd.grad(
        outputs=pred_real,
        inputs=[x],
        create_graph=True,
        retain_graph=True,
    )
    grad_real2 = grad_real.pow(2)
    dims = list(range(1, grad_real2.ndim))
    grad_penalty = grad_real2.sum(dims) * 0.5

    return grad_penalty.sum()


ind_criterion = nn.BCELoss()


def indep_loss(logits, y, should_be_dependent=True):
    logits = logits.squeeze()
    y = y.squeeze()
    y = torch.clamp(logits, min=1e-5, max=1 - 1e-5)
    logits = torch.clamp(logits, min=1e-5, max=1 - 1e-5)
    if should_be_dependent:
        return ind_criterion(logits, y)
    else:
        return -ind_criterion(logits, y)


import torch


def gradient_penalty(real, fake, f):
    def interpolate(a, b):
        shape = [a.size(0)] + [1] * (a.dim() - 1)
        alpha = torch.rand(shape)
        b_resized = torch.nn.functional.interpolate(b.unsqueeze(0).unsqueeze(0), size=a.shape, mode='bilinear',
                                                    align_corners=False).squeeze()
        inter = a + alpha * (b_resized - a)
        return inter

    x = interpolate(real, fake)
    pred = f(x)
    gradients = torch.autograd.grad(outputs=pred, inputs=x,
                                    grad_outputs=torch.ones_like(pred),
                                    create_graph=True, retain_graph=True)[0]
    slopes = torch.sqrt(torch.sum(gradients ** 2, dim=1))
    gp = torch.mean((slopes - 1.) ** 2)
    return gp



def eval_mmd(source, target, num_pts=500):
    return MMD(source, target).cost()


def get_cdca_term(src_feature, tgt_feature, src_label, tgt_label):
    attention_s_t = torch.nn.functional.softmax(src_feature @ tgt_feature.t()/2, dim=-1)
    attention_t_s = torch.nn.functional.softmax(tgt_feature @ src_feature.t()/2, dim=-1)
    attention_s_s = torch.nn.functional.softmax(src_feature @ src_feature.t()/2, dim=-1)
    attention_t_t = torch.nn.functional.softmax(tgt_feature @ tgt_feature.t()/2, dim=-1)


    labels_t_s = attention_t_s @ src_label
    labels_s_t = attention_s_t @ tgt_label
    labels_s_s = attention_s_s @ src_label
    labels_t_t = attention_t_t @ tgt_label

    labels_t_s = torch.clip(labels_t_s,1e-3,1-1e-3)
    labels_s_t = torch.clip(labels_s_t,1e-3,1-1e-3)
    labels_s_s = torch.clip(labels_s_s,1e-3,1-1e-3)
    labels_t_t = torch.clip_(labels_t_t,1e-3,1-1e-3)


    return labels_t_s ,labels_s_t,labels_s_s,labels_t_t
