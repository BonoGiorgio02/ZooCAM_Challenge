# coding: utf-8

# Standard imports
import os

# External imports
import torch
import torch.nn
import tqdm


def generate_unique_logpath(logdir, raw_run_name):
    """
    Generate a unique directory name
    Argument:
        logdir: the prefix directory
        raw_run_name(str): the base name
    Returns:
        log_path: a non-existent path like logdir/raw_run_name_xxxx
                  where xxxx is an int
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


class ModelCheckpoint(object):
    """
    Early stopping callback that saves a full training checkpoint.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        savepath,
        min_is_best: bool = True,
        optimizer=None,
        scheduler=None,
    ) -> None:
        self.model = model
        self.savepath = savepath
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score):
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score):
        return self.best_score is None or score > self.best_score

    def update(self, score, epoch: int):
        """
        Save checkpoint if score improved.
        """
        if self.is_better(score):
            ckpt = {
                "epoch": epoch,
                "best_score": score,
                "model": self.model.state_dict(),
            }
            if self.optimizer is not None:
                ckpt["optimizer"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                ckpt["scheduler"] = self.scheduler.state_dict()

            torch.save(ckpt, self.savepath)
            self.best_score = score
            return True
        return False


def train(model, loader, f_loss, optimizer, device, dynamic_display=True, wandb_log=None):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    optimizer -- A torch.optim.Optimzer object
    device    -- A torch.device
    Returns :
    The averaged train metrics computed over a sliding window
    """

    # We enter train mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.train()

    total_loss = 0
    num_samples = 0
    
    pbar = tqdm.tqdm(enumerate(loader), desc="Train", leave=False)
    
    for i, (inputs, targets) in pbar:

        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward propagation
        outputs = model(inputs)

        loss = f_loss(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the metrics
        # We here consider the loss is batch normalized
        total_loss += inputs.shape[0] * loss.item()
        num_samples += inputs.shape[0]
        
        # pbar.set_description(f"Train loss : {total_loss/num_samples:.2f}")
        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{total_loss/num_samples:.4f}", lr=f"{lr:.2e}")
        
        if wandb_log is not None and (i % 100) == 0:
            wandb_log({"train_loss_batch": float(loss.item()), "batch": i})

    return total_loss / num_samples


def test(model, loader, f_loss, device):
    """
    Test a model over the loader
    using the f_loss as metrics
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    device    -- A torch.device
    Returns :
    """

    # We enter eval mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.eval()

    total_loss = 0
    num_samples = 0
    
    pbar = tqdm.tqdm(enumerate(loader), desc="Val", leave=False)
    
    for i, (inputs, targets) in pbar:

        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward propagation
        outputs = model(inputs)

        loss = f_loss(outputs, targets)

        # Update the metrics
        # We here consider the loss is batch normalized
        total_loss += inputs.shape[0] * loss.item()
        num_samples += inputs.shape[0]
        
        pbar.set_postfix(loss=f"{total_loss/num_samples:.4f}")

    return total_loss / num_samples
