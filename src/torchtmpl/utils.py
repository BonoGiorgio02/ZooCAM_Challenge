# coding: utf-8

# Standard imports
import contextlib
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
    Model checkpoint callback.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        savepath,
        min_is_best: bool = True,
    ) -> None:
        self.model = model
        self.savepath = savepath
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score):
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score):
        return self.best_score is None or score > self.best_score

    def update(self, score):
        if self.is_better(score):
            torch.save(self.model.state_dict(), self.savepath)
            self.best_score = score
            return True
        return False


def _update_confusion_matrix(confusion, preds, targets, num_classes):
    valid = (targets >= 0) & (targets < num_classes)
    idx = num_classes * targets[valid].to(torch.int64) + preds[valid].to(torch.int64)
    bins = torch.bincount(idx, minlength=num_classes * num_classes)
    confusion += bins.reshape(num_classes, num_classes)


def macro_f1_from_confusion(confusion):
    conf = confusion.to(torch.float32)
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    denom = 2 * tp + fp + fn
    f1 = torch.where(denom > 0, (2 * tp) / denom, torch.zeros_like(tp))
    return float(f1.mean().item())


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

    for _, (inputs, targets) in (pbar := tqdm.tqdm(enumerate(loader), total=len(loader))):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else contextlib.nullcontext()
        )
        with amp_ctx:
            outputs = model(inputs)
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

        if scheduler is not None and scheduler_step == "batch":
            scheduler.step()

        total_loss += inputs.shape[0] * loss.item()
        num_samples += inputs.shape[0]
        pbar.set_description(f"Train loss: {total_loss / max(num_samples, 1):.4f}")

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
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else contextlib.nullcontext()
            )
            with amp_ctx:
                outputs = model(inputs)
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
    loss, _ = evaluate(model, loader, f_loss, device)
    return loss
