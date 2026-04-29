import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import average_precision_score
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Minimal config matching official default.yaml."""
    # Data
    batch_size: int = 64
    num_workers: int = 4
    
    # Model
    dobs: int = 1          # input channels (grayscale)
    henc: int = 32         # encoder hidden dim
    hpre: int = 32         # predictor hidden dim
    dstc: int = 8          # representation dim
    steps: int = 4         # prediction steps during training
    
    # Loss coefficients
    std_coeff: float = 10.0
    cov_coeff: float = 100.0
    
    # Optimization
    lr: float = 1e-3
    epochs: int = 100
    
    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1
    log_every: int = 1
    save_every: int = 10
    
    # Dataset
    map_size: int = 8      # detection map size


# =============================================================================
# UTILITIES
# =============================================================================

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def init_module_weights(m, std: float = 0.02):
    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)):
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# =============================================================================
# TEMPORAL BATCH MIXIN
# =============================================================================

class TemporalBatchMixin:
    """Handles 4D [B,C,H,W] and 5D [B,C,T,H,W] inputs."""

    def _forward(self, x):
        raise NotImplementedError

    def forward(self, x):
        assert x.ndim in [4, 5], "Only 4D or 5D tensors supported"
        if x.ndim == 5:
            b = x.shape[0]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            out = self._forward(x)
            out = rearrange(out, "(b t) c h w -> b c t h w", b=b)
            return out
        return self._forward(x)


# =============================================================================
# DATASET
# =============================================================================

FILENAME = Path("datasets/mnist_test_seq.npy")
URL = "https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"


def load_or_download(filename: str, url: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.exists(filename):
        print(f"Downloading {url}...")
        import urllib.request
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
    return filename


def return_bbox(img):
    import cv2
    thres = (img.min() + img.max()) / 2
    contours, _ = cv2.findContours(
        (img > thres).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, x + w, y + h))
    return boxes


class MovingMNIST(Dataset):
    def __init__(self, split: str = "train"):
        load_or_download(str(FILENAME), URL)
        dataset = np.load(str(FILENAME))  # (T, N, H, W) [0-255]
        dataset = np.swapaxes(dataset, 0, 1)  # (N, T, H, W)
        
        rs = np.random.RandomState(2025)
        dataset = rs.permutation(dataset)
        val_data, train_data = dataset[:1000], dataset[1000:]
        data = val_data if split == "val" else train_data
        
        # Flatten temporally by factor of 2
        self.data = np.reshape(
            data, [data.shape[0] * 2, data.shape[1] // 2, data.shape[2], data.shape[3]]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames = torch.from_numpy(self.data[idx]).unsqueeze(0).float() / 255.0
        return {"video": frames}


class MovingMNISTDet(MovingMNIST):
    def __init__(self, split: str = "train", map_size: int = 8):
        super().__init__(split)
        self.map_size = map_size
        N, T = self.data.shape[:2]
        self.digit_locations = torch.zeros((N, T, map_size, map_size))
        for idx, frames in enumerate(self.data):
            for t in range(T):
                boxes = return_bbox(frames[t])
                for x1, y1, x2, y2 in boxes:
                    x, y = (x1 + x2) / 2, (y1 + y2) / 2
                    px = int(x / frames.shape[-1] * map_size)
                    py = int(y / frames.shape[-2] * map_size)
                    self.digit_locations[idx, t, py, px] = 1

    def __getitem__(self, idx):
        instance = super().__getitem__(idx)
        instance["digit_location"] = self.digit_locations[idx]
        return instance


# =============================================================================
# ARCHITECTURES
# =============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet5(TemporalBatchMixin, nn.Module):
    def __init__(self, in_d, h_d, out_d, s1=1, s2=1, s3=1, avg_pool=False):
        super().__init__()
        self.avg_pool = avg_pool
        self.conv1 = nn.Conv2d(in_d, h_d, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(h_d)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = ResidualBlock(h_d, h_d, stride=s1)
        self.layer2 = ResidualBlock(h_d, h_d * 2, stride=s2)
        self.layer3 = ResidualBlock(h_d * 2, out_d, stride=s3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) if avg_pool else nn.Identity()

    def _forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        if self.avg_pool:
            out = out.flatten(1)
        return out


class ResUNet(TemporalBatchMixin, nn.Module):
    def __init__(self, in_d, h_d, out_d, is_rnn=False):
        super().__init__()
        self.is_rnn = is_rnn
        self.conv1 = nn.Conv2d(in_d, h_d, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(h_d)
        self.relu = nn.ReLU(inplace=True)
        
        self.enc1 = ResidualBlock(h_d, h_d, stride=1)
        self.enc2 = ResidualBlock(h_d, 2 * h_d, stride=2)
        self.enc3 = ResidualBlock(2 * h_d, 4 * h_d, stride=2)
        self.bott = ResidualBlock(4 * h_d, 8 * h_d, stride=2)
        
        self.up3 = nn.ConvTranspose2d(8 * h_d, 4 * h_d, 2, 2)
        self.dec3 = ResidualBlock(8 * h_d, 4 * h_d, stride=1)
        self.up2 = nn.ConvTranspose2d(4 * h_d, 2 * h_d, 2, 2)
        self.dec2 = ResidualBlock(4 * h_d, 2 * h_d, stride=1)
        self.up1 = nn.ConvTranspose2d(2 * h_d, 1 * h_d, 2, 2)
        self.dec1 = ResidualBlock(2 * h_d, 1 * h_d, stride=1)
        self.head = nn.Conv2d(h_d, out_d, 1)

    @staticmethod
    def _match_size(x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def _forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        s1 = self.enc1(x0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b = self.bott(s3)
        
        d3 = self._match_size(self.up3(b), s3)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))
        d2 = self._match_size(self.up2(d3), s2)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))
        d1 = self._match_size(self.up1(d2), s1)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))
        return self.head(d1)


class StateOnlyPredictor(nn.Module):
    """Wrapper: concatenates prev and next state channels, ignores actions."""
    def __init__(self, predictor, context_length=2):
        super().__init__()
        self.predictor = predictor
        self.is_rnn = predictor.is_rnn
        self.context_length = context_length

    def forward(self, x, a=None):
        prev_state = x[:, :, :-1]   # [B, C, T-1, H, W]
        next_state = x[:, :, 1:]    # [B, C, T-1, H, W]
        combined = torch.cat((prev_state, next_state), dim=1)
        return self.predictor(combined)


class Projector(nn.Module):
    """MLP projector from spec string like '256-512-128'."""
    def __init__(self, mlp_spec):
        super().__init__()
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.extend([nn.Linear(f[i], f[i + 1]), nn.BatchNorm1d(f[i + 1]), nn.ReLU(True)])
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        self.net = nn.Sequential(*layers)
        self.out_dim = f[-1]

    def forward(self, x):
        return self.net(x)


class ImageDecoder(TemporalBatchMixin, nn.Module):
    def __init__(self, in_dim, out_dim=1, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_dim, 3, 1, 1),
        )
        self.apply(init_module_weights)

    def _forward(self, x):
        return self.net(x)


class conv3d2(nn.Sequential):
    def __init__(self, in_d, h_d, out_d, tk, ts, sk, ss, pad):
        super().__init__(
            nn.Conv3d(in_d, h_d, (tk, sk, sk), (1, 1, 1), pad),
            nn.ReLU(),
            nn.Conv3d(h_d, out_d, (tk, sk, sk), (ts, ss, ss), pad),
        )
        self.apply(init_module_weights)
        if pad == "valid":
            self.t_shift = 2 * tk - 1
        elif pad == "same":
            self.t_shift = 2 * (tk - 1)


class DetHead(nn.Module):
    def __init__(self, in_d, h_d, out_d):
        super().__init__()
        self.head = nn.Sequential(conv3d2(in_d, h_d, out_d, 1, 1, 3, 1, "same"))
        self.apply(init_module_weights)

    def forward(self, x):
        # x: [B, C, T, H, W]
        x = [F.adaptive_avg_pool2d(x[:, :, t], (8, 8)) for t in range(x.shape[2])]
        x = torch.stack(x, 2)
        x = self.head(x).squeeze(1)
        return torch.sigmoid(x)

    @torch.no_grad()
    def score(self, preds, targets):
        scores = []
        for T in range(len(preds) - 1):
            x = preds[T]
            x = [F.adaptive_avg_pool2d(x[:, :, t], (8, 8)) for t in range(x.shape[2])]
            x = torch.stack(x, 2)
            x = self.head(x).squeeze(1)
            y = targets[:, T:]
            x = x[:, T:]
            ap = average_precision_score(
                y.flatten().detach().long().cpu().numpy(),
                x.flatten().detach().cpu().numpy(),
                average="weighted",
            )
            scores.append(ap)
        return scores


# =============================================================================
# LOSSES
# =============================================================================

class HingeStdLoss(nn.Module):
    def __init__(self, std_margin: float = 1.0):
        super().__init__()
        self.std_margin = std_margin

    def forward(self, x):
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0) + 0.0001)
        return torch.mean(F.relu(self.std_margin - std))


class CovarianceLoss(nn.Module):
    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x):
        batch_size = x.shape[0]
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (batch_size - 1)
        return self.off_diagonal(cov).pow(2).mean()


class VCLoss(nn.Module):
    def __init__(self, std_coeff, cov_coeff, proj=None):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.proj = nn.Identity() if proj is None else proj
        self.std_loss_fn = HingeStdLoss(std_margin=1.0)
        self.cov_loss_fn = CovarianceLoss()

    def forward(self, x, actions=None):
        x = x.transpose(0, 1).flatten(1).transpose(0, 1)  # [B*T*H*W, C]
        fx = self.proj(x)
        std_loss = self.std_loss_fn(fx)
        cov_loss = self.cov_loss_fn(fx)
        loss = self.std_coeff * std_loss + self.cov_coeff * cov_loss
        total_unweighted = std_loss + cov_loss
        loss_dict = {"std_loss": std_loss.item(), "cov_loss": cov_loss.item()}
        return loss, total_unweighted, loss_dict


class SquareLossSeq(nn.Module):
    def __init__(self, proj=None):
        super().__init__()
        self.proj = nn.Identity() if proj is None else proj

    def forward(self, state, predi):
        state = self.proj(state.transpose(0, 1).flatten(1).transpose(0, 1))
        predi = self.proj(predi.transpose(0, 1).flatten(1).transpose(0, 1))
        return F.mse_loss(state, predi)


# =============================================================================
# JEPA
# =============================================================================

class JEPA(nn.Module):
    def __init__(self, encoder, aencoder, predictor, regularizer, predcost):
        super().__init__()
        self.encoder = encoder
        self.action_encoder = aencoder
        self.predictor = predictor
        self.regularizer = regularizer
        self.predcost = predcost
        self.single_unroll = getattr(self.predictor, "is_rnn", False)

    def unroll(self, observations, actions=None, nsteps=1, unroll_mode="parallel",
               ctxt_window_time=1, compute_loss=True, return_all_steps=False):
        state = self.encoder(observations)
        context_length = getattr(self.predictor, "context_length", 0)

        if compute_loss:
            rloss, rloss_unweight, rloss_dict = self.regularizer(state, actions)
            ploss = 0.0
        else:
            rloss = rloss_unweight = rloss_dict = ploss = None

        actions_encoded = self.action_encoder(actions) if actions is not None else None
        all_steps = [] if return_all_steps else None

        if unroll_mode == "parallel":
            predicted_states = state
            for _ in range(nsteps):
                predicted_states = self.predictor(predicted_states, actions_encoded)[:, :, :-1]
                if return_all_steps:
                    all_steps.append(predicted_states)
                predicted_states = torch.cat((state[:, :, :context_length], predicted_states), dim=2)
                if compute_loss:
                    ploss += self.predcost(state, predicted_states) / nsteps

        elif unroll_mode == "autoregressive":
            if actions is not None and nsteps > actions.size(2):
                raise ValueError(f"nsteps ({nsteps}) > action length ({actions.size(2)})")
            effective_ctxt = 1 if self.single_unroll else ctxt_window_time
            predicted_states = state[:, :, :effective_ctxt]
            for i in range(nsteps):
                context_states = predicted_states[:, :, -effective_ctxt:]
                context_actions = (actions_encoded[:, :, max(0, i + 1 - effective_ctxt):i + 1]
                                   if actions_encoded is not None else None)
                pred_step = self.predictor(context_states, context_actions)[:, :, -1:]
                predicted_states = torch.cat([predicted_states, pred_step], dim=2)
                if return_all_steps:
                    all_steps.append(predicted_states.clone())
                if compute_loss:
                    ploss += self.predcost(pred_step, state[:, :, i + 1:i + 2]) / nsteps
        else:
            raise ValueError(f"Unknown unroll_mode: {unroll_mode}")

        if compute_loss:
            loss = rloss + ploss
            losses = (loss, rloss, rloss_unweight, rloss_dict, ploss)
        else:
            losses = None

        return (all_steps if return_all_steps else predicted_states), losses


class JEPAProbe(nn.Module):
    """Frozen JEPA encoder + trainable head."""
    def __init__(self, jepa, head, hcost):
        super().__init__()
        self.jepa = jepa
        self.head = head
        self.hcost = hcost

    def forward(self, observations, targets):
        with torch.no_grad():
            state = self.jepa.encoder(observations)
        output = self.head(state.detach())
        return self.hcost(output, targets)


# =============================================================================
# EVALUATION
# =============================================================================

@torch.inference_mode()
def validation_loop(val_loader, jepa, detection_head, pixel_decoder, steps, device):
    jepa.eval()
    detection_head.eval()
    pixel_decoder.eval()

    metrics = {k: [] for k in ["val/recon_loss", "val/det_loss"]}
    for batch in tqdm(val_loader, desc="Val", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        x = batch["video"]
        loc_map = batch["digit_location"]

        recon_loss = pixel_decoder(x, x)
        det_loss = detection_head(x, loc_map)
        metrics["val/recon_loss"].append(float(recon_loss.item()))
        metrics["val/det_loss"].append(float(det_loss.item()))

        T = x.shape[2]
        preds, _ = jepa.unroll(x, None, nsteps=T - 2, unroll_mode="parallel",
                               compute_loss=False, return_all_steps=True)
        scores = detection_head.head.score(preds, loc_map[:, 2:])
        for s, score in enumerate(scores):
            metrics.setdefault(f"AP_{s}", []).append(float(score))

    jepa.train()
    detection_head.train()
    pixel_decoder.train()
    return {k: float(np.mean(v)) for k, v in metrics.items()}


# =============================================================================
# TRAINING
# =============================================================================

def train(cfg: Config):
    device = torch.device(cfg.device)
    setup_seed(cfg.seed)

    # Data
    train_set = MovingMNISTDet(split="train", map_size=cfg.map_size)
    val_set = MovingMNISTDet(split="val", map_size=cfg.map_size)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Device: {device}")

    # Model
    encoder = ResNet5(cfg.dobs, cfg.henc, cfg.dstc)
    predictor_model = ResUNet(2 * cfg.dstc, cfg.hpre, cfg.dstc)
    predictor = StateOnlyPredictor(predictor_model, context_length=2)
    projector = Projector(f"{cfg.dstc}-{cfg.dstc*4}-{cfg.dstc*4}")
    regularizer = VCLoss(cfg.std_coeff, cfg.cov_coeff, proj=projector)
    ploss = SquareLossSeq(projector)
    jepa = JEPA(encoder, encoder, predictor, regularizer, ploss).to(device)

    decoder = ImageDecoder(cfg.dstc, cfg.dobs, hidden_dim=16)
    dethead = DetHead(cfg.dstc, cfg.hpre, cfg.dobs)
    pixel_decoder = JEPAProbe(jepa, decoder, nn.MSELoss()).to(device)
    detection_head = JEPAProbe(jepa, dethead, nn.BCELoss()).to(device)

    enc_params = sum(p.numel() for p in encoder.parameters())
    pre_params = sum(p.numel() for p in predictor.parameters())
    print(f"Encoder params: {enc_params:,} | Predictor params: {pre_params:,}")

    jepa.train()
    detection_head.train()
    pixel_decoder.train()

    optimizer = Adam([
        {"params": jepa.parameters(), "lr": cfg.lr},
        {"params": pixel_decoder.head.parameters(), "lr": cfg.lr / 10},
        {"params": detection_head.head.parameters(), "lr": cfg.lr},
    ])

    # Training loop
    global_step = 0
    for epoch in range(cfg.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            x = batch["video"]
            loc_map = batch["digit_location"]

            optimizer.zero_grad()
            _, (jepa_loss, regl, _, regldict, pl) = jepa.unroll(
                x, None, nsteps=cfg.steps, unroll_mode="parallel",
                compute_loss=True, return_all_steps=False,
            )
            recon_loss = pixel_decoder(x, x)
            det_loss = detection_head(x, loc_map)
            total_loss = jepa_loss + recon_loss + det_loss

            total_loss.backward()
            optimizer.step()

            pbar.set_postfix({
                "loss": f"{jepa_loss.item():.4f}",
                "vc": f"{regl.item():.4f}",
                "pred": f"{pl.item():.4f}",
            })
            global_step += 1

        if epoch % cfg.log_every == 0:
            val_logs = validation_loop(val_loader, jepa, detection_head,
                                       pixel_decoder, cfg.steps, device)
            print(f"\\nEpoch {epoch}: val_recon={val_logs.get('val/recon_loss', 0):.4f} "
                  f"val_det={val_logs.get('val/det_loss', 0):.4f}")
            for k, v in val_logs.items():
                if k.startswith("AP_"):
                    print(f"  {k}={v:.4f}")

    print("Training complete!")
    return jepa, pixel_decoder, detection_head


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    cfg = Config()
    train(cfg)
