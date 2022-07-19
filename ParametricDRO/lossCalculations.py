import numpy as np
import tensorflow as tf
import torch as th
from src.configuration import ArgumentGroup
from src.running_average import LogRunningAverage
from typing import Optional


def compute_model_loss(
    losses: th.Tensor,
    log_q: th.Tensor,
    log_p: th.Tensor,
    alpha: float,
    log_Z_model: LogRunningAverage,
) -> th.Tensor:
    """Computes the loss of the model given the model's los and the
    adversary's weights on each sample
    Args:
        losses: Loss of each sample (of shape [B])
        log_q: Log probability of each sample under the adversary
            (of shape [B])
        log_p: Log probability of each sample under the MLE model
            (of shape [B])
        alpha: Interpolation coefficient
        log_Z_model: Log normalizer for the model's weights
    Returns:
        Loss for the model on this batch (a scalar tensor)
    """
    # Compute the log ratio
    log_ratios = log_q-log_p
    # Renormalize weights
    # log_ratios = log_ratios - log_Z_model.value
    log_ratios -= log_Z_model.value
    # Importance weights
    weights = th.exp(log_ratios)
    # Loss
    model_loss = (weights.detach()*losses).sum()
    # Interpolate between the adversarial loss and the ERM objective
    # 1 means we are only training on the adversarial objective
    # 0 means we are only training on the ERM objective
    if alpha < 1:
        erm_loss = losses.mean()
        model_loss = model_loss*alpha + (1-alpha)*erm_loss
    return model_loss


def compute_adv_loss(
    losses: th.Tensor,
    log_q: th.Tensor,
    log_p: th.Tensor,
    adv_obj: str,
    adv_on_acc: bool,
    beta: float,
    tau: float,
    self_norm_lambda: float,
    log_Z_adv: LogRunningAverage,
    log_Z_model: LogRunningAverage,
    errors: Optional[th.Tensor],
) -> th.Tensor:
    """Compute the adversary's loss given the model's loss on a batch of
    examples and the weights produced by the adversary
    Args:
        losses: A tensor containing the losses of the model on a
            minibatch
        log_q: A tensor containing the probability of each example
            in the mininbatch
        log_p: A tensor containing the baseline probability for
            each example in the batch
        adv_obj: type of objective for the adversary
        adv_on_acc: Train adversary to maximize error
        beta: Interpolation coefficient
        tau: Temperature for the adversary's loss
        self_norm_lambda: self normalization penalty
        log_Z_adv: Running average of the weights used in
            computing the adversary's loss
        errors: Tensor containing the errors of the model on the
            minibatch (these can be non-differentiable, as opposed as the
            losses)
        log_Z_model: This is the log normalizer of the
            weights used to compute the model's loss. Here this is used to
            recompute the model loss in the `zero_sum` setting (where the
            adversary is trained to maximize the model's loss)
    """
    # Interpolate with the regular nll
    if adv_obj == "zero_sum":
        # LM NLL in log space:
        weights = (log_q - log_p) - log_Z_model.value
        adv_loss = -(th.exp(weights)*losses.detach()).mean()
    elif adv_obj == "fwd_kl":
        # Log likelihood ratio
        log_weights = (log_q - log_p) - log_Z_model.value
        # weights
        weights = th.exp(log_weights)
        # "KL penalty" component
        kl_loss = (weights*log_weights).mean()
        # "zero sum" component
        zero_sum_loss = (-weights*losses.detach()).mean()
        adv_loss = zero_sum_loss + tau*kl_loss
    elif adv_obj == "log_zero_sum":
        # LM NLL in log space:
        log_losses = log_q - log_p + th.log(losses).detach()
        adv_loss = -th.logsumexp(log_losses, 0)
    elif adv_obj.startswith("exp"):
        if adv_on_acc:
            log_q_star = errors / tau
        else:
            # q*(x, y) \propto \ell(x, y)/temp * p
            log_q_star = losses.detach() / tau
        if adv_obj == "exp":
            # Reweight by log_p
            log_lm_weights = log_q_star-log_p
        elif adv_obj == "exp_kl":
            # Reweight by log_p
            log_lm_weights = log_q_star
        # Actual weights are normalized across minibatch
        log_normalizer = th.logsumexp(log_lm_weights, 0).item()
        # Running average
        log_Z_adv += log_normalizer
        # print(log_Z_adv.value, flush=True)
        # log_lm_weights += np.log(batch.size)
        lm_weights = th.exp(log_lm_weights-log_Z_adv.value)
        # Loss for the lm
        adv_loss = -(lm_weights*log_q).sum()
    # # lm_loss = -(th.exp(log_q-log_p)*nlls.detach()).mean()
    if self_norm_lambda > 0:
        log_expected_ratio = th.logsumexp(log_q-np.log(len(log_q)), dim=0)
        adv_loss += self_norm_lambda*log_expected_ratio**2
    # Interpolate with the likelihood of the data
    # (this pulls back the adversary towards the nominal distribution)
    if beta < 1:
        adv_loss = beta*adv_loss + (1-beta) * (-log_q).mean()
    return adv_loss
