# coding: utf-8

# Standard imports
import contextlib
import copy
import os

# External imports
import torch
import torch.nn
import tqdm
from sklearn.metrics import f1_score


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


def macro_f1_from_confusion(confusion):
    conf = confusion.to(torch.float32)
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    denom = 2 * tp + fp + fn
    f1 = torch.where(denom > 0, (2 * tp) / denom, torch.zeros_like(tp))
    return float(f1.mean().item())


def unpack_supervised_batch(batch):
    if not isinstance(batch, (tuple, list)):
        raise ValueError("Expected training/validation batch as tuple/list.")

    if len(batch) == 2:
        inputs, targets = batch
        metadata = None
    elif len(batch) == 3:
        inputs, metadata, targets = batch
    else:
        raise ValueError(
            "Expected supervised batch format (inputs, targets) or (inputs, metadata, targets)."
        )

    return inputs, metadata, targets


def unpack_inference_batch(batch):
    if not isinstance(batch, (tuple, list)):
        raise ValueError("Expected inference batch as tuple/list.")

    if len(batch) == 2:
        inputs, names = batch
        metadata = None
    elif len(batch) == 3:
        inputs, metadata, names = batch
    else:
        raise ValueError(
            "Expected inference batch format (inputs, names) or (inputs, metadata, names)."
        )

    return inputs, metadata, names


def model_forward(model, inputs, metadata=None):
    expects_metadata = bool(getattr(model, "expects_metadata", False))

    if expects_metadata:
        if metadata is None:
            raise ValueError("Model expects metadata but batch did not provide it.")
        return model(inputs, metadata)

    if metadata is None:
        return model(inputs)

    return model(inputs)


def apply_tta(inputs, mode):
    mode = str(mode).lower()
    if mode in {"orig", "none"}:
        return inputs
    if mode == "hflip":
        return torch.flip(inputs, dims=[3])
    if mode == "vflip":
        return torch.flip(inputs, dims=[2])
    if mode == "rot180":
        return torch.rot90(inputs, k=2, dims=[2, 3])
    raise ValueError(f"Unsupported TTA mode '{mode}'")


class ModelEMA:
    def __init__(self, model, decay=0.9998):
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()

        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)

    def state_dict(self):
        return {
            "decay": self.decay,
            "ema_state_dict": self.ema.state_dict(),
        }

    def load_state_dict(self, state):
        if state is None:
            return

        if "ema_state_dict" in state:
            self.decay = float(state.get("decay", self.decay))
            self.ema.load_state_dict(state["ema_state_dict"])
            return

        self.ema.load_state_dict(state)


def train(
    model,
    loader,
    f_loss,
    optimizer,
    device,
    *,
    scaler=None,
    amp_enabled=False,
    scheduler=None,
    scheduler_step="epoch",
    grad_clip_norm=None,
    ema=None,
):
    """
    Train a model for one epoch.
    Returns:
        Averaged train loss.
    """
    model.train()

    use_amp = bool(amp_enabled and device.type == "cuda")
    total_loss = 0.0
    num_samples = 0
    
    pbar = tqdm.tqdm(enumerate(loader), desc="Train")
    
    for i, (inputs, targets) in pbar:

    for _, batch in (pbar := tqdm.tqdm(enumerate(loader), total=len(loader))):
        inputs, metadata, targets = unpack_supervised_batch(batch)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if metadata is not None:
            metadata = metadata.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else contextlib.nullcontext()
        )
        with amp_ctx:
            outputs = model_forward(model, inputs, metadata)
            loss = f_loss(outputs, targets)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        if scheduler is not None and scheduler_step == "batch":
            scheduler.step()

        total_loss += inputs.shape[0] * loss.item()
        num_samples += inputs.shape[0]
        
        # pbar.set_description(f"Train loss : {total_loss/num_samples:.2f}")
        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{total_loss/num_samples:.4f}", lr=f"{lr:.2e}")
        
        if wandb_log is not None and (i % 100) == 0:
            wandb_log({"train_loss_batch": float(loss.item()), "batch": i})

    return total_loss / max(num_samples, 1)


def evaluate(model, loader, f_loss, device, *, amp_enabled=False, num_classes=None):
    """
    Evaluate model on loader and return (loss, macro_f1).
    """
    model.eval()
    use_amp = bool(amp_enabled and device.type == "cuda")

    total_loss = 0.0
    num_samples = 0

    confusion = None

    with torch.inference_mode():
        for batch in loader:
            inputs, metadata, targets = unpack_supervised_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if metadata is not None:
                metadata = metadata.to(device, non_blocking=True)

            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else contextlib.nullcontext()
            )
            with amp_ctx:
                outputs = model_forward(model, inputs, metadata)
                loss = f_loss(outputs, targets)

            preds = torch.argmax(outputs, dim=1)

            if confusion is None:
                current_num_classes = outputs.shape[1] if num_classes is None else num_classes
                confusion = torch.zeros(
                    (current_num_classes, current_num_classes),
                    dtype=torch.long,
                )
            _update_confusion_matrix(confusion, preds.detach().cpu(), targets.detach().cpu(), confusion.shape[0])

            total_loss += inputs.shape[0] * loss.item()
            num_samples += inputs.shape[0]

    if confusion is None:
        return 0.0, 0.0

    macro_f1 = macro_f1_from_confusion(confusion)
    avg_loss = total_loss / max(num_samples, 1)
    return avg_loss, macro_f1


def test(model, loader, f_loss, device):
    """
    Backward-compatible test helper returning only loss.
    """

    # We enter eval mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.eval()

    total_loss = 0.0
    num_samples = 0

    all_targets = []
    all_preds = []

    pbar = tqdm.tqdm(enumerate(loader), total=len(loader), desc="Val")

    with torch.no_grad():
        for i, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = f_loss(outputs, targets)

            total_loss += inputs.shape[0] * loss.item()
            num_samples += inputs.shape[0]

            preds = torch.argmax(outputs, dim=1)

            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

            current_loss = total_loss / num_samples
            pbar.set_postfix(loss=f"{current_loss:.4f}")

    val_loss = total_loss / num_samples
    macro_f1 = f1_score(all_targets, all_preds, average="macro")

    return val_loss, macro_f1


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()

    all_probs = []

    pbar = tqdm.tqdm(loader, desc="Predict")

    for batch in pbar:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs = batch[0]
        else:
            inputs = batch

        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu())

    return torch.cat(all_probs, dim=0)


@torch.no_grad()
def predict_proba_tta(model, loaders, device, weights=None):
    model.eval()

    if weights is None:
        weights = [1.0 / len(loaders)] * len(loaders)

    assert len(weights) == len(loaders), "weights and loaders must have same length"

    final_probs = None

    for w, loader in zip(weights, loaders):
        probs = predict_proba(model, loader, device)

        if final_probs is None:
            final_probs = w * probs
        else:
            final_probs += w * probs

    return final_probs
