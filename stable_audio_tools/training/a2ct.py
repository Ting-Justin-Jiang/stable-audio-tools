import copy
import math
import random
import typing as tp

import pytorch_lightning as pl
import torch
from ema_pytorch import EMA
from torch import Tensor

from ..models.diffusion import ConditionedDiffusionModelWrapper
from ..models.reconstruction import predict_x0
from ..inference.sampling import get_alphas_sigmas
from .losses.losses import charbonnier_weight
from .utils import create_optimizer_from_config, create_scheduler_from_config, regex_any_match, sample_pair_indices


class A2CTTrainingWrapper(pl.LightningModule):
    """
    A²CT: Anchored Adaptive Cross-Time Tuning for domain-adaptive fine-tuning of
    distilled, time-conditioned audio diffusion models.
    """

    def __init__(
        self,
        model: ConditionedDiffusionModelWrapper,
        lr: tp.Optional[float] = None,
        optimizer_configs: tp.Optional[dict] = None,
        discrete_tau: tp.Sequence[float] = (0.1, 0.6),
        tau_grid_from_solver: tp.Optional[dict] = None,
        pair_sampler: tp.Optional[dict] = None,
        gate: tp.Optional[dict] = None,
        charbonnier: tp.Optional[dict] = None,
        sra: tp.Optional[dict] = None,
        prox: tp.Optional[dict] = None,
        ema: tp.Optional[dict] = None,
        cfg_dropout_prob: float = 0.1,
        mask_padding: bool = False,
        mask_padding_dropout: float = 0.0,
        pre_encoded: bool = False,
    ):
        super().__init__()
        self.diffusion = model
        self.diffusion_objective = self.diffusion.diffusion_objective

        # EMA options
        ema = {} if ema is None else ema
        self.use_ema = bool(ema.get("use", True))
        self.ema_beta = float(ema.get("beta", 0.9999))
        self.ema_start_step = int(ema.get("start_step", 0))

        if self.use_ema:
            self.diffusion_ema = EMA(
                self.diffusion.model,
                beta=self.ema_beta,
                power=3 / 4,
                update_every=1,
                update_after_step=max(1, self.ema_start_step),
                include_online_model=False,
            )
        else:
            self.diffusion_ema = None

        # Frozen prior
        self.prior_model = copy.deepcopy(self.diffusion.model).eval().requires_grad_(False)

        if tau_grid_from_solver is not None:
            tau_grid = self._build_tau_grid_from_solver(tau_grid_from_solver)
        else:
            tau_grid = torch.tensor([float(x) for x in discrete_tau], dtype=torch.float32)
        self.register_buffer("discrete_tau", tau_grid, persistent=True)
        assert torch.all(self.discrete_tau[1:] > self.discrete_tau[:-1]), "discrete_tau must be strictly increasing"
        assert self.discrete_tau.ndim == 1 and self.discrete_tau.numel() in (2, 3, 5, 9), "discrete_tau length must be 1, 2, 4 or 8"

        # scheduler options
        pair_sampler = {} if pair_sampler is None else pair_sampler
        self.neighbor_prob = float(pair_sampler.get("neighbor_prob", 0.7))
        self.skip_max = int(pair_sampler.get("skip_max", 3))

        # gate options
        gate = {} if gate is None else gate
        self.tau_c = float(gate.get("tau_c", 0.6))
        self.tau_delta = float(gate.get("tau_delta", 0.1))
        self.use_confidence = bool(gate.get("use_confidence", False))
        self.confidence_prev_step = int(gate.get("confidence_prev_step", 1))
        self.diffusion_loss_tau_gate = gate.get("diffusion_tau_gate", None)
        if self.diffusion_loss_tau_gate is not None:
            self.diffusion_loss_tau_gate = float(self.diffusion_loss_tau_gate)
        self.diffusion_loss_weight = float(gate.get("diffusion_loss_weight", 0.0))

        # Charbonnier
        charbonnier = {} if charbonnier is None else charbonnier
        self.char_c0 = float(charbonnier.get("c0", 1.0))
        self.char_c = float(charbonnier.get("c", 0.02))

        # SRA options
        sra = {} if sra is None else sra
        self.use_sra = bool(sra.get("use", False))
        self.sra_lambda = float(sra.get("lambda", 0.1))
        self.sra_student_layer = int(sra.get("student_layer", 8))
        self.sra_ema_layer = int(sra.get("ema_layer", 20))
        self.sra_prior_layer = int(sra.get("prior_layer", 20))

        # TODO: check if these are necessary
        # Prox options
        prox = {} if prox is None else prox
        self.use_prox = bool(prox.get("use", False))
        self.prox_mu = float(prox.get("mu", 3e-4))
        self.prox_name_regex = list(prox.get("param_name_regex", ["time_embed", "pos_embed", "block0", "block1"]))

        self._prox_anchors: tp.Optional[tp.Dict[str, torch.Tensor]] = None
        self._prox_anchors_device: tp.Optional[torch.device] = None

        self.mask_padding = mask_padding
        self.mask_padding_dropout = mask_padding_dropout
        self.pre_encoded = pre_encoded
        self.cfg_dropout_prob = cfg_dropout_prob

        self.register_buffer("iter_counter", torch.tensor(0, dtype=torch.long), persistent=False)

        assert lr is not None or optimizer_configs is not None, "Must specify either lr or optimizer_configs in training config"
        if optimizer_configs is None:
            self.optimizer_configs = {
                "diffusion": {
                    "optimizer": {"type": "Adam", "config": {"lr": float(lr)}},
                }
            }
        else:
            self.optimizer_configs = optimizer_configs

    @staticmethod
    def _expand_time_to_batch(tau: Tensor, x: Tensor) -> Tensor:
        assert tau.ndim == 1 and tau.shape[0] == x.shape[0]
        return tau

    def _build_tau_grid_from_solver(self, cfg: dict) -> torch.Tensor:
        """
        Build a monotone increasing τ-grid using a solver-style schedule of length K.
        """
        K = int(cfg.get("K", 8))
        sigma_max = float(cfg.get("sigma_max", 1.0))
        assert K >= 1
        steps = K

        if self.diffusion_objective in ("rectified_flow", "rf_denoiser", "eps"):
            if sigma_max > 1:
                sigma_max = 1.0
            logsnr_max = math.log(((1 - sigma_max) / sigma_max) + 1e-6) if sigma_max < 1 else -6.0
            logsnr = torch.linspace(logsnr_max, 2.0, steps + 1)
            t_desc = torch.sigmoid(-logsnr)
            t_desc[0] = sigma_max
            t_desc[-1] = 0.0
            tau_inc = torch.flip(t_desc, dims=[0])  # increasing 0..sigma_max
            print(f"Using tau grid from solver: {tau_inc}")
            return tau_inc

        # v-param: linear in sigma
        print(f"Using tau grid from solver: {torch.linspace(0.0, sigma_max, steps + 1)}")
        return torch.linspace(0.0, sigma_max, steps + 1)

    def _mix(self, x0: Tensor, eps: Tensor, tau: Tensor) -> Tensor:
        """
        Form noisy mixture x_tau given clean x0 and noise eps at time tau.
        For v-param EDM grid we use x_tau = alpha * x0 + sigma * eps.
        For rectified_flow, we interpret tau as noise-level in [0,1] and use alphas=1-tau, sigmas=tau.
        """
        objective = self.diffusion.diffusion_objective
        if objective == "v":
            alpha, sigma = get_alphas_sigmas(tau)
            alpha = alpha[:, None, None]
            sigma = sigma[:, None, None]
        elif objective in ("rectified_flow", "rf_denoiser", "eps"):
            t_b = tau[:, None, None]
            alpha = 1.0 - t_b
            sigma = t_b
        else:
            raise NotImplementedError(f"Unsupported diffusion_objective: {objective}")
        return x0 * alpha + eps * sigma

    def _predict_x0_with_model(
        self,
        model: torch.nn.Module,
        x: Tensor,
        tau: Tensor,
        cond: tp.Dict[str, tp.Any],
        extra_args: tp.Dict[str, tp.Any],
        sra_extract_layer: tp.Optional[int] = None,
    ) -> tp.Tuple[Tensor, tp.Optional[Tensor]]:
        """
        Forward through model possibly extracting a feature for SRA and reconstruct x0.
        Returns (x0_hat, feature_or_None).
        """
        kwargs = dict(**extra_args)
        if "cfg_dropout_prob" not in kwargs:
            kwargs["cfg_dropout_prob"] = self.cfg_dropout_prob
        if sra_extract_layer is not None:
            out, feat = self._forward_with_feature(model, x, tau, cond, sra_extract_layer, **kwargs)
            x0_hat = predict_x0(self.diffusion.diffusion_objective, x, tau, out)
            return x0_hat, feat
        else:
            out = self._forward_only(model, x, tau, cond, **kwargs)
            x0_hat = predict_x0(self.diffusion.diffusion_objective, x, tau, out)
            return x0_hat, None

    def _forward_only(self, model: torch.nn.Module, x: Tensor, tau: Tensor, cond: tp.Dict[str, tp.Any], **kwargs) -> Tensor:
        original = self.diffusion.model
        self.diffusion.model = model
        try:
            return self.diffusion(x, tau, cond=cond, **kwargs)
        finally:
            self.diffusion.model = original

    def _build_prox_anchors(self) -> None:
        if not self.use_prox:
            return
        anchors: tp.Dict[str, torch.Tensor] = {}
        for name, p in self.prior_model.named_parameters():
            if p.requires_grad is False and self._sensitive_param_mask(name):
                anchors[name] = p.detach()
        # Move anchors to device and dtype of the student
        device = self.device
        dtype_map: tp.Dict[str, torch.dtype] = {n: p.dtype for n, p in self.diffusion.model.named_parameters()}
        for name in list(anchors.keys()):
            target_dtype = dtype_map.get(name, anchors[name].dtype)
            anchors[name] = anchors[name].to(device=device, dtype=target_dtype)
        self._prox_anchors = anchors
        self._prox_anchors_device = device

    def _forward_with_feature(
        self,
        model: torch.nn.Module,
        x: Tensor,
        tau: Tensor,
        cond: tp.Dict[str, tp.Any],
        layer_index: int,
        **kwargs,
    ) -> tp.Tuple[Tensor, Tensor]:
        """
        Call model with feature extraction.
        """
        original = self.diffusion.model
        self.diffusion.model = model
        try:
            out, feat = self.diffusion(
                x,
                tau,
                cond=cond,
                sra_extract_layer=layer_index,
                **kwargs,
            )
            return out, feat
        finally:
            self.diffusion.model = original

    def _compute_gate(
        self,
        tau_r: Tensor,
        conf_prior: tp.Optional[Tensor],
        conf_ema: tp.Optional[Tensor],
    ) -> Tensor:
        # lambda_noise = 1 - sigmoid( (tau_r - tau_c) / tau_delta )
        lambda_noise = 1.0 - torch.sigmoid((tau_r - self.tau_c) / self.tau_delta)
        if self.use_confidence and conf_prior is not None and conf_ema is not None:
            denom = (conf_prior + conf_ema + 1e-8)
            conf = conf_prior / denom
            lam = 0.5 * lambda_noise + 0.5 * conf
        else:
            lam = lambda_noise
        return lam.clamp(0.0, 1.0)

    def _sensitive_param_mask(self, name: str) -> bool:
        return regex_any_match(name, self.prox_name_regex)

    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs["diffusion"]
        opt_diff = create_optimizer_from_config(diffusion_opt_config["optimizer"], self.diffusion.parameters())
        if "scheduler" in diffusion_opt_config:
            sched_diff = create_scheduler_from_config(diffusion_opt_config["scheduler"], opt_diff)
            sched_diff_config = {"scheduler": sched_diff, "interval": "step"}
            return [opt_diff], [sched_diff_config]
        return [opt_diff]

    def training_step(self, batch, batch_idx):
        self.iter_counter += 1

        reals, metadata = batch
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]
        x0 = reals

        # Pre-transform
        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)
            encoded_ch = getattr(self.diffusion.pretransform, "encoded_channels", None)
            already_encoded = encoded_ch is not None and x0.shape[1] == encoded_ch
            if not self.pre_encoded and not already_encoded:
                with torch.cuda.amp.autocast(), torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)
                    x0 = self.diffusion.pretransform.encode(x0)
            elif getattr(self.diffusion.pretransform, "scale", 1.0) != 1.0:
                x0 = x0 / self.diffusion.pretransform.scale

        # Conditioning
        cond = self.diffusion.conditioner(metadata, self.device)
        use_padding_mask = self.mask_padding and random.random() > self.mask_padding_dropout
        if use_padding_mask:
            pad_masks = torch.stack([md["padding_mask"] for md in metadata], dim=0).to(self.device)
            extra_args: tp.Dict[str, tp.Any] = {"mask": pad_masks}
        else:
            extra_args = {}

        # use deterministic teacher targets (no cfg dropout)
        extra_args_student = dict(extra_args)
        extra_args_teacher = dict(extra_args)
        extra_args_teacher["cfg_dropout_prob"] = 0.0

        # time pair sampling
        # TODO: maybe we should change this sampling strategy...according to the conosistency model by Yang Song
        K = int(self.discrete_tau.numel())
        B = int(x0.shape[0])
        t_idx, r_idx = sample_pair_indices(K, B, neighbor_prob=self.neighbor_prob, skip_max=self.skip_max)
        t_idx = t_idx.to(self.device)
        r_idx = r_idx.to(self.device)
        if self.use_confidence:
            s_idx = torch.clamp(r_idx - self.confidence_prev_step, min=0)
        else:
            s_idx = r_idx

        tau_t = self.discrete_tau[t_idx]
        tau_r = self.discrete_tau[r_idx]
        tau_s = self.discrete_tau[s_idx]

        eps = torch.randn_like(x0)
        x_t = self._mix(x0, eps, tau_t)
        x_r = self._mix(x0, eps, tau_r)
        x_s = self._mix(x0, eps, tau_s)

        # Student forward at t
        if self.use_sra:
            g_t, y_student = self._predict_x0_with_model(self.diffusion.model, x_t, tau_t, cond, extra_args_student, sra_extract_layer=self.sra_student_layer)
        else:
            g_t, y_student = self._predict_x0_with_model(self.diffusion.model, x_t, tau_t, cond, extra_args_student, sra_extract_layer=None)

        # Teachers at r
        with torch.no_grad():
            if self.diffusion_ema is not None and self.diffusion_ema.ema_model is not None:
                ema_teacher = self.diffusion_ema.ema_model
                ema_prev_train = ema_teacher.training
                ema_teacher.eval()
            else:
                ema_teacher = self.diffusion.model
                ema_prev_train = None

            # TODO: SRA could only use subset of the timestep interval, instead of clamping to sra_max_time_interval
            if self.use_sra:
                g_r_ema, y_ema = self._predict_x0_with_model(ema_teacher, x_r, tau_r, cond, extra_args_teacher, sra_extract_layer=self.sra_ema_layer)
                g_r_prior, y_prior = self._predict_x0_with_model(self.prior_model, x_r, tau_r, cond, extra_args_teacher, sra_extract_layer=self.sra_prior_layer)
            else:
                g_r_ema, _ = self._predict_x0_with_model(ema_teacher, x_r, tau_r, cond, extra_args_teacher, sra_extract_layer=None)
                g_r_prior, _ = self._predict_x0_with_model(self.prior_model, x_r, tau_r, cond, extra_args_teacher, sra_extract_layer=None)

            # confidence via 1-step cycle residuals near r
            if self.use_confidence:
                g_s_ema, _ = self._predict_x0_with_model(ema_teacher, x_s, tau_s, cond, extra_args_teacher, sra_extract_layer=None)
                g_s_prior, _ = self._predict_x0_with_model(self.prior_model, x_s, tau_s, cond, extra_args_teacher, sra_extract_layer=None)
                cyc_ema = (g_r_ema - g_s_ema).pow(2).mean(dim=(1, 2))
                cyc_prior = (g_r_prior - g_s_prior).pow(2).mean(dim=(1, 2))
            else:
                cyc_ema = cyc_prior = None

            if ema_prev_train is not None and ema_prev_train:
                ema_teacher.train()

        # Gate lambda per sample
        lam = self._compute_gate(tau_r, conf_prior=cyc_prior, conf_ema=cyc_ema)
        lam_bc = lam[:, None, None]

        g_target = lam_bc * g_r_ema + (1.0 - lam_bc) * g_r_prior

        # Cross-time loss
        # Stabilize deltas by clipping extreme values to avoid exploding early steps
        delta = (g_t - g_target).clamp(min=-20.0, max=20.0)
        w = charbonnier_weight(delta, c0=self.char_c0, c=self.char_c)[:, None, None]
        Lx = (w * delta.pow(2)).mean()

        # SRA loss
        if self.use_sra and y_student is not None:
            if self.diffusion_ema is not None and self.diffusion_ema.ema_model is not None:
                # y_ema and y_prior were computed above in no-grad
                pass
            # Compute per-sample SRA
            lam_s = lam[:, None]
            sra_loss = 0.0
            if 'sum' in dir(torch):
                # avoid linter complaint; using torch ops only
                pass
            if self.use_sra:
                assert y_ema is not None and y_prior is not None
                mse_ema = (y_student - y_ema).pow(2).mean(dim=tuple(range(1, y_student.ndim)))
                mse_prior = (y_student - y_prior).pow(2).mean(dim=tuple(range(1, y_student.ndim)))
                Lsra = (lam * mse_ema + (1.0 - lam) * mse_prior).mean()
            else:
                Lsra = torch.tensor(0.0, device=self.device)
        else:
            Lsra = torch.tensor(0.0, device=self.device)

        if self.diffusion_loss_tau_gate is not None and self.diffusion_loss_weight > 0.0:
            per_sample_mse = (g_t - x0).pow(2).mean(dim=(1, 2))
            gate_mask = (tau_t <= self.diffusion_loss_tau_gate).float()
            Ld = (gate_mask * per_sample_mse).mean()
        else:
            Ld = torch.tensor(0.0, device=self.device)

        # Proximal regularizer
        if self.use_prox:
            if self._prox_anchors is None or self._prox_anchors_device != self.device:
                self._build_prox_anchors()
            Lprox_val = torch.tensor(0.0, device=self.device)
            for name, p in self.diffusion.model.named_parameters():
                if p.requires_grad and self._sensitive_param_mask(name) and self._prox_anchors is not None and name in self._prox_anchors:
                    anchor = self._prox_anchors[name]
                    if anchor.device != p.device or anchor.dtype != p.dtype:
                        anchor = anchor.to(device=p.device, dtype=p.dtype)
                    Lprox_val = Lprox_val + (p - anchor).pow(2).sum()
        else:
            Lprox_val = torch.tensor(0.0, device=self.device)

        L_total = Lx + self.sra_lambda * Lsra + self.prox_mu * Lprox_val + self.diffusion_loss_weight * Ld

        # Logging
        log_dict = {
            "train/Lx": Lx.detach(),
            "train/Ld": Ld.detach(),
            "train/Lsra": Lsra.detach() if isinstance(Lsra, Tensor) else torch.tensor(float(Lsra)),
            "train/Lprox": Lprox_val.detach(),
            "train/Ltotal": L_total.detach(),
            "gate/lambda_mean": lam.mean().detach(),
            "gate/lambda_std": lam.std(unbiased=False).detach(),
            "time/gap_idx_mean": (t_idx - r_idx).float().mean().detach(),
            "time/t_idx_mean": t_idx.float().mean().detach(),
            "time/r_idx_mean": r_idx.float().mean().detach(),
            "train/lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }
        self.log_dict(log_dict, prog_bar=True, on_step=True)

        return L_total

    def on_before_zero_grad(self, *args, **kwargs):
        if self.diffusion_ema is not None:
            self.diffusion_ema.update()

    def on_fit_start(self):
        self.prior_model.to(self.device)
        if self.use_prox:
            self._build_prox_anchors()

    def validation_step(self, batch, batch_idx):
        reals, metadata = batch
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]
        x0 = reals

        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)
            with torch.no_grad():
                encoded_ch = getattr(self.diffusion.pretransform, "encoded_channels", None)
                already_encoded = encoded_ch is not None and x0.shape[1] == encoded_ch
                if not self.pre_encoded and not already_encoded:
                    x0 = self.diffusion.pretransform.encode(x0)
                elif getattr(self.diffusion.pretransform, "scale", 1.0) != 1.0:
                    x0 = x0 / self.diffusion.pretransform.scale

        cond = self.diffusion.conditioner(metadata, self.device)
        extra_args: tp.Dict[str, tp.Any] = {}

        taus = [self.discrete_tau[0], self.discrete_tau[len(self.discrete_tau) // 2], self.discrete_tau[-1]]
        taus = torch.stack(taus) if isinstance(taus, list) else taus
        taus = taus.to(self.device)

        mse_vals = []
        with torch.no_grad():
            for tau in taus:
                tau_b = tau.repeat(x0.shape[0]).to(self.device)
                eps = torch.randn_like(x0)
                x_tau = self._mix(x0, eps, tau_b)
                out = self.diffusion(x_tau, tau_b, cond=cond, cfg_dropout_prob=self.cfg_dropout_prob, **extra_args)
                x0_hat = predict_x0(self.diffusion.diffusion_objective, x_tau, tau_b, out)
                mse = (x0_hat - x0).pow(2).mean()
                mse_vals.append(mse)

        self.log("val/recon_mse_mean", torch.stack(mse_vals).mean(), prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        pass

    def export_model(self, path: str, use_safetensors: bool = False):
        model = self.diffusion
        if self.diffusion_ema is not None and self.diffusion_ema.ema_model is not None:
            original_model = model.model
            model.model = self.diffusion_ema.ema_model
            try:
                if use_safetensors:
                    from safetensors.torch import save_file
                    save_file(model.state_dict(), path)
                else:
                    torch.save({"state_dict": model.state_dict()}, path)
            finally:
                model.model = original_model
        else:
            if use_safetensors:
                from safetensors.torch import save_file
                save_file(model.state_dict(), path)
            else:
                torch.save({"state_dict": model.state_dict()}, path)

