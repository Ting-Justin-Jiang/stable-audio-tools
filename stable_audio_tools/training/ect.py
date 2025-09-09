import pytorch_lightning as pl
import random
import torch
import typing as tp

from ema_pytorch import EMA
from safetensors.torch import save_file
from time import time

from ..models.diffusion import ConditionedDiffusionModelWrapper
from ..inference.sampling import get_alphas_sigmas
from .losses import MSELoss, MultiLoss
from .utils import create_optimizer_from_config, create_scheduler_from_config

DEBUG = False


class Profiler:
    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed * 1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep


class ECTTrainingWrapper(pl.LightningModule):
    """
    Easy Consistency Tuning (ECT) training loop for conditional audio diffusion.
    
    Supports optional SRA (Self-Representation Alignment) for enhanced training.
    The combined objective is: L = L_ECT + lambda * L_SRA
    """

    def __init__(
            self,
            model: ConditionedDiffusionModelWrapper,
            lr: tp.Optional[float] = None,
            mask_padding: bool = False,
            mask_padding_dropout: float = 0.0,
            use_ema: bool = True,
            log_loss_info: bool = False,
            optimizer_configs: tp.Optional[dict] = None,
            pre_encoded: bool = False,
            cfg_dropout_prob=0.1,

            # ECT parameters
            mapping_q: float = 8.0,
            mapping_k: float = 8.0,
            mapping_b: float = 1.0,
            mapping_d: int = 15000,
            min_sigma: float = 1e-4,
            max_sigma: float = 30,
            timestep_dist: tp.Literal["lognormal", "uniform"] = "lognormal",
            t_log_mean: float = -1.1,
            t_log_std: float = 2.0,
            
            # SRA parameters
            use_sra: bool = False,
            sra_lambda: float = 0.1,
            sra_student_layer: int = 8,
            sra_teacher_layer: int = 20,
            sra_max_time_interval: float = 0.2,
    ):
        super().__init__()
        self.diffusion = model

        if use_ema:
            self.diffusion_ema = EMA(
                self.diffusion.model,
                beta=0.9999,
                power=3 / 4,
                update_every=1,
                update_after_step=1,
                include_online_model=False
            )
        else:
            self.diffusion_ema = None

        self.mask_padding = mask_padding
        self.mask_padding_dropout = mask_padding_dropout
        self.cfg_dropout_prob = cfg_dropout_prob
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        # timestep sampling
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.timestep_dist = timestep_dist
        self.t_log_mean = t_log_mean
        self.t_log_std = t_log_std

        # mapping function p(r|t,iters) hyper‑params (Eq. 15)
        self.q = mapping_q
        self.k = mapping_k
        self.b = mapping_b
        self.d = mapping_d

        # iteration counter
        self.register_buffer("iter_counter", torch.tensor(0, dtype=torch.long), persistent=False)

        # loss
        self.diffusion_objective = model.diffusion_objective
        self.log_loss_info = log_loss_info

        # SRA parameters
        self.use_sra = use_sra
        self.sra_lambda = sra_lambda
        self.sra_student_layer = sra_student_layer
        self.sra_teacher_layer = sra_teacher_layer
        self.sra_max_time_interval = sra_max_time_interval

        # SRA validation
        if self.use_sra:
            if self.sra_student_layer >= self.sra_teacher_layer:
                raise ValueError(f"SRA requires student_layer ({self.sra_student_layer}) < teacher_layer ({self.sra_teacher_layer}). "
                               f"Student should be at earlier layer with higher noise, teacher at later layer with lower noise.")
            
            if not use_ema:
                print("WARNING: SRA is enabled but EMA is disabled. SRA works best with EMA teacher model. "
                      "Consider setting use_ema=True for optimal SRA performance.")
            
            print(f"SRA enabled: student_layer={self.sra_student_layer}, teacher_layer={self.sra_teacher_layer}, "
                  f"lambda={self.sra_lambda}, max_time_interval={self.sra_max_time_interval}")

        # optimizer
        assert lr is not None or optimizer_configs is not None, "Must specify either lr or optimizer_configs in training config"
        if optimizer_configs is None:
            self.optimizer_configs = {
                "diffusion": {
                    "optimizer": {"type": "Adam", "config": {"lr": lr}},
                }
            }
        else:
            self.optimizer_configs = optimizer_configs
        self.pre_encoded = pre_encoded

    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs['diffusion']
        opt_diff = create_optimizer_from_config(diffusion_opt_config['optimizer'], self.diffusion.parameters())
        if "scheduler" in diffusion_opt_config:
            sched_diff = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt_diff)
            sched_diff_config = {
                "scheduler": sched_diff,
                "interval": "step"
            }
            return [opt_diff], [sched_diff_config]
        return [opt_diff]

    def sample_t(self, batch: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps t (~ noise levels σ)."""
        if self.timestep_dist == "lognormal":
            ln_t = torch.randn(batch, device=device) * self.t_log_std + self.t_log_mean
            t = ln_t.exp().clamp(min=self.min_sigma)
        else:  # uniform
            t = torch.rand(batch, device=device) * (self.max_sigma - self.min_sigma) + self.min_sigma
        if DEBUG: print(f"sample t: {t} at iteration {self.iter_counter}")
        return t

    def sample_r(self, t: torch.Tensor) -> torch.Tensor:
        """Compute r given t and current iteration (Eq. 15 in paper)."""
        a = float(self.iter_counter // self.d)  # floor division; increases every d iterations
        n_t = 1.0 + self.k * torch.sigmoid(-self.b * t)
        ratio = 1.0 - torch.pow(self.q, -a * n_t)
        r = ratio * t
        r = r.clamp(min=self.min_sigma)

        # If SRA is enabled,
        # r >= tau_to_sigma( sigma_to_tau(t) - sra_max_time_interval )
        if self.use_sra:
            tau_t = self.sigma_to_tau(t)
            tau_min = torch.clamp(tau_t - self.sra_max_time_interval, min=0.0)
            r_min = self.tau_to_sigma(tau_min)
            r = torch.maximum(r, r_min)
        if DEBUG: print(f"sample r: {r} at iteration {self.iter_counter}")

        return r

    def sigma_to_tau(self, sigma: torch.Tensor) -> torch.Tensor:
        return (2 / torch.pi) * torch.atan(sigma)

    def tau_to_sigma(self, tau: torch.Tensor) -> torch.Tensor:
        return torch.tan(0.5 * torch.pi * tau)

    def training_step(self, batch, batch_idx):
        # setup
        p = Profiler()
        self.iter_counter += 1

        reals, metadata = batch
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]
        x0 = reals

        p.tick("setup")

        # pre-transform
        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)
            if not self.pre_encoded:
                with torch.cuda.amp.autocast(), \
                        torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)
                    x0 = self.diffusion.pretransform.encode(x0)
                    # align padding masks if you use them
            elif getattr(self.diffusion.pretransform, "scale", 1.0) != 1.0:
                x0 = x0 / self.diffusion.pretransform.scale

        p.tick("pre-transform")

        # conditioning
        cond = self.diffusion.conditioner(metadata, self.device)
        use_padding_mask = self.mask_padding and random.random() > self.mask_padding_dropout
        if use_padding_mask:
            pad_masks = torch.stack([md["padding_mask"] for md in metadata], dim=0).to(self.device)
            extra_args = {"mask": pad_masks}
        else:
            extra_args = {}

        p.tick("conditioning")

        # sample t & r
        t_sigma = self.sample_t(x0.shape[0], self.device)
        r_sigma = self.sample_r(t_sigma) # ratio is r/t
        
        t, r = self.sigma_to_tau(t_sigma), self.sigma_to_tau(r_sigma) # snr -> angle

        if DEBUG:
            print(f"convert t to angle time tau_t: {t}")
            print(f"convert r to angle time tau_r: {r}")

        # add noise
        eps = torch.randn_like(x0)
        if self.diffusion_objective in ["v"]: # note that t means sigma in ECT
            alphas_t, sigmas_t = get_alphas_sigmas(t)
            alphas_t, sigmas_t = alphas_t[:, None, None], sigmas_t[:, None, None]
            alphas_r, sigmas_r = get_alphas_sigmas(r)
            alphas_r, sigmas_r = alphas_r[:, None, None], sigmas_r[:, None, None]

        elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            t_b, r_b = t[:, None, None], r[:, None, None]
            alphas_t, sigmas_t = 1-t_b, t_b
            alphas_r, sigmas_r = 1-r_b, r_b

        else: raise NotImplementedError

        # interpolate for noisy input
        x_t = x0 * alphas_t + eps * sigmas_t
        x_r = x0 * alphas_r + eps * sigmas_r

        if self.use_sra:
            # forward Pass 1: Get ECT output and STUDENT representation from ONLINE model
            f_xt, sra_student_repr = self.diffusion(
                x_t, t, cond=cond, cfg_dropout_prob=self.cfg_dropout_prob, 
                sra_extract_layer=self.sra_student_layer, **extra_args
            )
            
            # forward Pass 2: Generate ECT output and TEACHER representation from the same model
            with torch.no_grad():
                if self.diffusion_ema is not None and self.diffusion_ema.ema_model is not None:
                    original_model = self.diffusion.model
                    self.diffusion.model = self.diffusion_ema.ema_model
                    try:
                        f_xr, sra_teacher_repr = self.diffusion(
                            x_r, r, cond=cond, cfg_dropout_prob=self.cfg_dropout_prob,
                            sra_extract_layer=self.sra_teacher_layer, **extra_args
                        )
                    finally:
                        self.diffusion.model = original_model
                else:
                    # Fallback: use online model if EMA not available (not recommended)
                    f_xr, sra_teacher_repr = self.diffusion(
                        x_r, r, cond=cond, cfg_dropout_prob=self.cfg_dropout_prob,
                        sra_extract_layer=self.sra_teacher_layer, **extra_args
                    )
        else:
            # Standard ECT: two forward passes without SRA
            f_xt = self.diffusion(x_t, t, cond=cond, cfg_dropout_prob=self.cfg_dropout_prob, **extra_args)
            with torch.no_grad():
                f_xr = self.diffusion(x_r, r, cond=cond, cfg_dropout_prob=self.cfg_dropout_prob, **extra_args)

        if self.diffusion_objective == "v":
            x0_pred_t = alphas_t * x_t - sigmas_t * f_xt
            x0_pred_r = alphas_r * x_r - sigmas_r * f_xr
        elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            x0_pred_t = x_t - sigmas_t * f_xt
            x0_pred_r = x_r - sigmas_r * f_xr
        else:
            raise NotImplementedError

        if DEBUG:
            with torch.no_grad():
                mse_r = torch.mean((x0_pred_r - x0) ** 2)
                print(f"Boundary condition mapping error: {mse_r}")
        p.tick("forward")

        # ECT loss
        delta = x0_pred_t - x0_pred_r

        # Charbonnier‑style adaptive weight to stabilise gradients (Eq. 16)
        adaptive = 1.0 / torch.sqrt(delta.pow(2).mean(dim=tuple(range(1, delta.ndim))) + 1e-5)
        time_gap = torch.clamp(t - r, min=1e-2)
        timestep_w = 1.0 / time_gap
        w = (adaptive * timestep_w)[:, None, None]  # broadcast to match delta dims

        ect_loss = (w * delta.pow(2)).mean()

        # SRA loss
        if self.use_sra:
            # L2 loss between student and teacher representations
            sra_loss = torch.nn.functional.mse_loss(sra_student_repr, sra_teacher_repr)

            total_loss = ect_loss + self.sra_lambda * sra_loss
            
            # Track EMA usage for SRA
            using_ema_teacher = self.diffusion_ema is not None and self.diffusion_ema.ema_model is not None
        else:
            sra_loss = torch.tensor(0.0, device=self.device)
            total_loss = ect_loss
            using_ema_teacher = False

        if DEBUG:
            print(f"ECT Loss: {ect_loss.item():.6f}")
            if self.use_sra:
                print(f"SRA Loss: {sra_loss.item():.6f}")
                print(f"SRA using EMA teacher: {using_ema_teacher}")
            print(f"Total Loss: {total_loss.item():.6f}")

        p.tick("loss")

        log_dict = {
            'train/ect_loss': ect_loss.detach(),
            'train/total_loss': total_loss.detach(),
            'train/std_data': x0.std(),
            'train/lr': self.trainer.optimizers[0].param_groups[0]['lr']
        }
        
        if self.use_sra:
            log_dict['train/sra_loss'] = sra_loss.detach()
            log_dict['train/sra_lambda'] = self.sra_lambda
            log_dict['train/sra_using_ema_teacher'] = float(using_ema_teacher)

            # Log representation statistics for monitoring
            log_dict['train/sra_student_repr_std'] = sra_student_repr.std().detach()
            log_dict['train/sra_teacher_repr_std'] = sra_teacher_repr.std().detach()
            log_dict['train/sra_time_interval'] = (t_sigma - r_sigma).mean().detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)

        p.tick("log")
        # print(f"Profiler: {p}")
        return total_loss

    def on_before_zero_grad(self, *args, **kwargs):
        if self.diffusion_ema is not None:
            self.diffusion_ema.update()

    def validation_step(self, batch, batch_idx):
        ...

    def on_validation_epoch_end(self):
        ...

    def export_model(self, path, use_safetensors=False):
        if self.diffusion_ema is not None and self.diffusion_ema.ema_model is not None:
            # Temporarily replace the model with EMA model for export
            original_model = self.diffusion.model
            self.diffusion.model = self.diffusion_ema.ema_model
            
            try:
                if use_safetensors:
                    save_file(self.diffusion.state_dict(), path)
                else:
                    torch.save({"state_dict": self.diffusion.state_dict()}, path)
            finally:
                # Restore original model
                self.diffusion.model = original_model
        else:
            if use_safetensors:
                save_file(self.diffusion.state_dict(), path)
            else:
                torch.save({"state_dict": self.diffusion.state_dict()}, path)

