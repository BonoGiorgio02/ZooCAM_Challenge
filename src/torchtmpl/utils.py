# coding: utf-8

# Standard imports
import contextlib
import copy
import os

# External imports
import torch
import torch.nn
import torch.nn.functional as F
import tqdm

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


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


def _expand_norm_values(values, channels):
    if values is None:
        return None
    if isinstance(values, (int, float)):
        return [float(values)] * int(channels)
    if isinstance(values, (list, tuple)):
        parsed = [float(v) for v in values]
        if len(parsed) == channels:
            return parsed
        if len(parsed) == 1:
            return parsed * int(channels)
    raise ValueError(
        f"Invalid normalization stats for TTA (expected scalar or list of length 1/{channels})."
    )


def _apply_color_jitter_on_normalized(
    inputs,
    brightness=1.0,
    contrast=1.0,
    norm_mean=None,
    norm_std=None,
):
    if inputs.ndim != 4:
        raise ValueError(f"Expected BCHW tensor for TTA, got shape {tuple(inputs.shape)}")

    channels = int(inputs.shape[1])
    resolved_mean = _expand_norm_values(norm_mean, channels)
    resolved_std = _expand_norm_values(norm_std, channels)
    if resolved_mean is None:
        resolved_mean = _expand_norm_values(_IMAGENET_MEAN, channels)
    if resolved_std is None:
        resolved_std = _expand_norm_values(_IMAGENET_STD, channels)

    mean = inputs.new_tensor(resolved_mean).view(1, channels, 1, 1)
    std = inputs.new_tensor(resolved_std).view(1, channels, 1, 1)

    x = inputs * std + mean
    x = x.clamp(0.0, 1.0)

    if float(brightness) != 1.0:
        x = x * float(brightness)
    if float(contrast) != 1.0:
        x_mean = x.mean(dim=(2, 3), keepdim=True)
        x = (x - x_mean) * float(contrast) + x_mean

    x = x.clamp(0.0, 1.0)
    return (x - mean) / std


def _build_gaussian_kernel2d(kernel_size, sigma, device, dtype):
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-0.5 * (coords / float(sigma)) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d


def _apply_gaussian_blur_on_normalized(
    inputs,
    sigma=0.5,
    kernel_size=3,
    norm_mean=None,
    norm_std=None,
):
    if inputs.ndim != 4:
        raise ValueError(f"Expected BCHW tensor for TTA, got shape {tuple(inputs.shape)}")
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("Gaussian blur kernel_size must be an odd integer >= 3.")

    channels = int(inputs.shape[1])
    resolved_mean = _expand_norm_values(norm_mean, channels)
    resolved_std = _expand_norm_values(norm_std, channels)
    if resolved_mean is None:
        resolved_mean = _expand_norm_values(_IMAGENET_MEAN, channels)
    if resolved_std is None:
        resolved_std = _expand_norm_values(_IMAGENET_STD, channels)

    mean = inputs.new_tensor(resolved_mean).view(1, channels, 1, 1)
    std = inputs.new_tensor(resolved_std).view(1, channels, 1, 1)

    x = inputs * std + mean
    x = x.clamp(0.0, 1.0)

    kernel_2d = _build_gaussian_kernel2d(
        kernel_size=kernel_size,
        sigma=max(float(sigma), 1e-3),
        device=x.device,
        dtype=x.dtype,
    )
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    x = F.conv2d(x, weight=kernel, bias=None, stride=1, padding=kernel_size // 2, groups=channels)
    x = x.clamp(0.0, 1.0)
    return (x - mean) / std


def _apply_single_tta_token(inputs, token, norm_mean=None, norm_std=None):
    if token in {"orig", "none"}:
        return inputs
    if token == "hflip":
        return torch.flip(inputs, dims=[3])
    if token == "vflip":
        return torch.flip(inputs, dims=[2])
    if token == "rot90":
        return torch.rot90(inputs, k=1, dims=[2, 3])
    if token == "rot180":
        return torch.rot90(inputs, k=2, dims=[2, 3])
    if token == "rot270":
        return torch.rot90(inputs, k=3, dims=[2, 3])
    if token.startswith("cj"):
        parts = token.split("_")
        if parts[0] != "cj":
            raise ValueError(f"Unsupported TTA token '{token}'")

        brightness = 1.0
        contrast = 1.0
        if len(parts) == 1:
            raise ValueError(
                "Color jitter TTA must provide at least one factor, e.g. "
                "'cj_b1.05' or 'cj_b0.95_c1.05'."
            )

        for part in parts[1:]:
            if part.startswith("b") and len(part) > 1:
                brightness = float(part[1:])
            elif part.startswith("c") and len(part) > 1:
                contrast = float(part[1:])
            else:
                raise ValueError(f"Unsupported color-jitter TTA token part '{part}'")

        return _apply_color_jitter_on_normalized(
            inputs,
            brightness=brightness,
            contrast=contrast,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )
    if token.startswith("gb"):
        parts = token.split("_")
        if parts[0] != "gb":
            raise ValueError(f"Unsupported TTA token '{token}'")

        kernel_size = 3
        sigma = 0.5
        for part in parts[1:]:
            if part.startswith("k") and len(part) > 1:
                kernel_size = int(part[1:])
            elif part.startswith("s") and len(part) > 1:
                sigma = float(part[1:])
            else:
                raise ValueError(f"Unsupported gaussian-blur TTA token part '{part}'")

        return _apply_gaussian_blur_on_normalized(
            inputs,
            sigma=sigma,
            kernel_size=kernel_size,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

    raise ValueError(f"Unsupported TTA token '{token}'")


def apply_tta(inputs, mode, norm_mean=None, norm_std=None):
    mode = str(mode).lower().strip()
    if mode in {"", "orig", "none"}:
        return inputs

    tokens = [tok.strip() for tok in mode.split("+") if tok.strip()]
    if not tokens:
        return inputs

    out = inputs
    for token in tokens:
        out = _apply_single_tta_token(
            out,
            token,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )
    return out


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
    wandb_log = None
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

    for i, batch in (pbar := tqdm.tqdm(enumerate(loader), total=len(loader))):
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
        pbar.set_description(f"Train loss: {total_loss / max(num_samples, 1):.4f}")

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
    loss, _ = evaluate(model, loader, f_loss, device)
    return loss
