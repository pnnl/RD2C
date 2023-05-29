import numpy as np
import tensorflow as tf
import torch as th
import torch.nn.functional as F
import tqdm
import os
from lossCalculations import compute_model_loss, compute_adv_loss

# Source File Functions (Go Through)
from src.tasks import task_list, prepare_task
from src.data.language_modeling import to_lm_batch
from src.optim import get_optimizer, get_lr_scheduler
from src.utils import cacheable, get_loader, get_group_dro_loader
from src.tasks import LanguageModelingTask, CCLanguageModelingTask, Task
from src.configuration import Experiment, ArgumentGroup
from src.running_average import get_log_running_average, LogRunningAverage
from src.stopping import AverageStopping, GreedyMinMaxStopping, GroupRobustStopping
from src.logging import NpzLogger

""" RP-DRO Reproduction
Reference:
[1] Michel, Paul, Tatsunori Hashimoto, and Graham Neubig. 
    "Distributionally Robust Models with Parametric Likelihood Ratios." 
    arXiv preprint arXiv:2204.06340 (2022).
Courtesy of https://github.com/pmichel31415/P-DRO
"""

# Rebuild train function for just RP-DRO (ratio version of P-DRO)
def train(
    model: th.nn.Module,
    adv: th.nn.Module,
    task: Task,
    model_args: ArgumentGroup,
    adv_args: ArgumentGroup,
    optim_args: ArgumentGroup,
    dro_args: ArgumentGroup,
    train_log_interval: int = 1,
    device="cuda:0",
    exp_name: str = "",
    figure_prefix: str = "precisions",
    results_prefix: str = "results/",
    eval_domain_filters=None,
    train_domain_filters=None,
    valid_pseudo_domain_filters=None,
    save_name: str = "",
):

    # MARCO EDIT: Hyperparameters (using what was used in code runs)
    model_optimizer = "adamw"
    lr = 2e-5
    weight_decay = 0
    adv_mom = 0
    norm_k = 0
    clip_grad = 10
    batch_size = 64
    max_tokens_per_batch = 2500
    num_workers = 1
    num_epochs = 50
    num_steps = 50
    lr_scheduler = "constant"
    valid_interval = 250
    update_every = 1
    joint = True
    class_conditional = True
    renorm_ratios = True
    adv_obj = "fwd_kl"
    adv_on_acc = True  # Train adversary to maximize error
    self_norm_lambda = 0
    alpha = 1.0
    beta = 1.0
    tau = 0.01


    adv_task = task

    # Save files
    model_file = os.path.join(results_prefix, f"{save_name}_model.pt")
    lm_file = os.path.join(results_prefix, f"{save_name}_lm.pt")
    adv_model_file = os.path.join(results_prefix, f"{save_name}_adv_model.pt")
    adv_lm_file = os.path.join(results_prefix, f"{save_name}_adv_lm.pt")
    robust_model_file = os.path.join(
        results_prefix, f"{save_name}_robust_model.pt")
    robust_lm_file = os.path.join(results_prefix, f"{save_name}_robust_lm.pt")

    # Optimizer for this task (can be rewritten)
    opt = get_optimizer(
        model_optimizer,
        list(model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Optimizer for the adversary
    # Default to the model's optimizer and lr
    adv_optimizer = model_optimizer
    adv_opt = get_optimizer(
        adv_optimizer,
        list(adv.parameters()),
        lr=lr,
        mom=adv_mom,
        weight_decay=weight_decay,
    )

    # Log normalizers
    log_Z_model = get_log_running_average(norm_k)
    log_Z_adv = get_log_running_average(norm_k)
    # Indices for each pseudo domain on the training set (if available)
    q = np.zeros(1)

    # Stopping
    avg_stop = AverageStopping(lower_is_better=False)
    # Minmax stopping
    adv_stop = GreedyMinMaxStopping(len(task.valid_data), lower_is_better=True)
    # Add ERM weights
    adv_stop.add_adv_log_weights(np.zeros(len(task.valid_data)))

    # Compute the initial log probabilities of the adversary on the dev set
    dev_log_q0 = np.zeros(len(task.valid_data))
    dev_log_q = np.zeros(len(task.valid_data))
    # set the model's mode to training mode.
    model.train()
    # No dropout for the adversary (otherwise the likelihood ratios can become
    # bad). In an ideal scenario we would dropout the same weights both for
    # adversary and MLE model, but since the MLE log probabilities are
    # pre-computed with the full model we are out of luck
    adv.eval()
    sampler, loader = get_loader(
        task.train_data,
        batch_size,
        max_tokens_per_batch=max_tokens_per_batch,
        shuffle=True,
        collate_fn=task.collate_fn,
        num_workers=num_workers,
    )

    # Number of steps and epochs
    steps_per_epochs = len(sampler)
    if num_epochs is not None:
        num_steps = steps_per_epochs * num_epochs
        # Don't stop based on step
        stop_by_step = False
    else:
        # Make sure we run as many epochs as necessary to reach
        # the appropriate number of steps
        stop_by_step = True
        optim_args.n_epochs = int(np.ceil(num_steps / steps_per_epochs))

    # Validate by epoch maybe?
    if valid_interval == "epoch":
        valid_interval = None
    else:
        valid_interval = int(valid_interval)

    # Get lr scheduler
    lr_schedule = get_lr_scheduler(lr_scheduler, opt, lr, num_steps)
    # Logging
    log_tracker = NpzLogger(
        filename=f"{results_prefix}{save_name}.npz",
        static_fields={"exp_name": exp_name,
                       "name": task.name,
                       "dev_log_q0": dev_log_q0},
        overwrite=True,
    )

    # Step tracker
    step = 0
    # Training loop
    for epoch in range(1, optim_args.n_epochs + 1):
        # Data source
        itr = tqdm.tqdm(loader)
        for step_in_epoch, batch in enumerate(itr, 1):
            # Total step
            step += 1
            # Reset gradients
            if (step - 1) % update_every == 0:
                opt.zero_grad()
            # if (step-1) % lm_update_every == 0:
            adv_opt.zero_grad()
            # Get data on device
            batch = batch.to(device)
            # Model forward pass to get the losses and predictions
            nlls, _, y_hat = task.nll(
                model,
                batch,
                reduction="none",
                predict=True,
            )
            # Model errors
            errors = (batch.outputs != y_hat).float().detach()
            # Adversary forward pass
            # Transform the minibatch for processing by the adversary
            lm_batch = batch
            if not (joint or class_conditional):
                lm_batch = to_lm_batch(lm_batch)
            # Get log prob of each sample under the adversary
            logits = adv_task.logits(adv, batch)
            y = batch.outputs.to(logits.device)
            log_q = - F.nll_loss(logits, y, reduction="none")
            if renorm_ratios:
                log_q = th.log_softmax(
                    log_q, dim=0) + np.log(len(log_q))
            # log prob under the MLE LM
            log_p = th.tensor(batch.attributes["log_p"]).to(log_q.device)
            # Keep track of the log normalizer for the weights used to
            # compute the model's loss
            log_Z_model += th.logsumexp(log_q - log_p, 0).item()
            model_loss = compute_model_loss(nlls, log_q, log_p, adv_args,
                                            log_Z_adv, log_Z_model, errors)
            # Compute the adversary's loss
            adv_loss = compute_adv_loss(nlls, log_q, log_p, adv_obj, adv_on_acc, beta, tau, self_norm_lambda,
                                        log_Z_adv, log_Z_model, errors)
            # Backward pass
            adv_loss.backward()
            # L2 regularization for the model
            if optim_args.l2_reg > 0:
                param_vec = th.cat([p.view(-1) for p in model.parameters()])
                model_loss += optim_args.l2_reg * th.sum(param_vec ** 2)
            # Model backward pass
            model_loss.backward()
            # Take a step
            if step % optim_args.update_every == 0:
                # Clip model gradient
                if optim_args.clip_grad > 0:
                    th.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        optim_args.clip_grad,
                    )
                # Update params and LR
                opt.step()
                lr_schedule.step()
            if step % adv_args.adv_update_every == 0:
                # Clip adv gradient
                if adv_args.clip_grad_adv > 0:
                    th.nn.utils.clip_grad_norm_(
                        adv.parameters(),
                        adv_args.clip_grad_adv,
                    )
                # Update adversary
                adv_opt.step()

