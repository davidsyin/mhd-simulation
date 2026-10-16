from math import sqrt
from types import SimpleNamespace
from typing import Optional, Union, Dict, Any
import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange, reduce

from ..utils import (
    exists,
    default,
    log,
)
from .diffusion_factory import register_diffusion_model


class ElucidatedDiffusion(nn.Module):
    """
    Elucidated Diffusion model implementation.

    This model uses a UNet backbone for the denoising process and implements
    the sampling and training process as described in the Elucidated Diffusion paper.
    It provides both regular sampling and DPM++ sampling methods for generating high-quality
    outputs from noise.

    References:
        - Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models"
    """

    def __init__(
        self,
        net: nn.Module,
        *,
        image_size: int,
        channels: int = 3,
        num_sample_steps: int = 32,
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        sigma_data: float = 0.5,
        rho: float = 7,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        S_churn: float = 80,
        S_tmin: float = 0.05,
        S_tmax: float = 50,
        S_noise: float = 1.003,
        gaot_input_size: int = 9,
    ):
        """
        Initialize the Elucidated Diffusion model.

        Args:
            net: Neural network model (typically UNet) for denoising
            image_size: Size of the input/output images (assumes square images)
            channels: Number of channels in the input/output images
            num_sample_steps: Number of discretized steps used during sampling
            sigma_min: Minimum noise level σ_min
            sigma_max: Maximum noise level σ_max
            sigma_data: Standard deviation of (the normalized) data distribution
            rho: Controls the sampling schedule shape
            P_mean: Mean of log-normal noise distribution for training
            P_std: Standard deviation of log-normal noise distribution for training
            S_churn: Stochastic sampling parameter (Table 5 in paper)
            S_tmin: Minimum threshold for applying stochasticity
            S_tmax: Maximum threshold for applying stochasticity
            S_noise: Noise scale for stochastic sampling
        """
        super().__init__()
        self.self_condition = True
        self.net = net

        self.channels = channels
        self.image_size = image_size
        self.gaot_input_size = gaot_input_size

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sample_steps = num_sample_steps
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

        self.register_buffer("mesh", self._make_mesh(), persistent=False)

    @property
    def device(self) -> torch.device:
        """Get the device on which the model parameters are stored."""
        return next(self.net.parameters()).device

    @property
    def num_points(self) -> int:
        """Number of flattened spatial points in one square image array."""
        return self.image_size * self.image_size

    def _is_array_image(self, images: torch.Tensor) -> bool:
        """Return True for the public array representation: (batch, points, channels)."""
        return images.ndim == 3

    def _make_mesh(self) -> torch.Tensor:
        """Create the DINO output mesh as a flattened 128x128 unit-square array."""
        coords = torch.linspace(-1.0, 1.0, self.image_size, dtype=torch.float32)
        xx, yy = torch.meshgrid(coords, coords, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(self.num_points, 2)

    def _assert_valid_image_array(self, images: torch.Tensor) -> None:
        # print("images", images.shape)
        """Validate an image represented as (batch, 128*128, channels)."""
        assert images.ndim == 3, (
            "images must be represented as arrays with shape "
            f"(batch, {self.num_points}, {self.channels})"
        )

        batch_size, num_points, channels = images.shape
        assert batch_size > 0, "batch dimension must be non-empty"
        assert num_points == self.num_points, (
            f"point-array image must have {self.num_points} points, got {num_points}"
        )
        assert channels == self.channels, "mismatch of image channels"

    def _mesh_for_batch(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Expand the stored mesh to (batch, points, 2) for GAOT query points."""
        return (
            self.mesh.to(device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

    def _prepare_condition_features(
        self, self_cond: torch.Tensor, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Validate condition features containing initial-condition and PINO output."""
        if self_cond is None:
            raise ValueError(
                "GAOT conditioning is required. Pass initial-condition and PINO "
                "features shaped "
                f"(batch, {self.num_points}, features)."
            )

        if self_cond.device != device:
            self_cond = self_cond.to(device)

        if self_cond.ndim != 3:
            raise ValueError(
                "GAOT conditioning must have shape "
                f"(batch, {self.num_points}, features), "
                f"got {tuple(self_cond.shape)}"
            )

        if self_cond.shape[0] != batch_size:
            raise ValueError(
                f"GAOT conditioning batch {self_cond.shape[0]} does not match "
                f"image batch {batch_size}."
            )
        if self_cond.shape[1] != self.num_points:
            raise ValueError(
                f"GAOT conditioning must have {self.num_points} points, "
                f"got {self_cond.shape[1]}."
            )

        return self_cond

    def _prepare_gaot_input_array(
        self,
        self_cond: torch.Tensor,
        noised_images: torch.Tensor,
        sigma: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build GAOT input_array with 5 per-point features."""
        condition = self._prepare_condition_features(self_cond, batch_size, device)
        scaled_noised_images = self.c_in(sigma.view(batch_size, 1, 1)) * noised_images
        noise_level = self.c_noise(sigma).view(batch_size, 1, 1).expand(
            -1, self.num_points, -1
        )

        input_array = torch.cat([condition, scaled_noised_images, noise_level], dim=2)
        if input_array.shape[-1] != self.gaot_input_size:
            raise ValueError(
                "GAOT diffusion input_array must contain "
                f"{self.gaot_input_size} features: initial condition, PINO output, "
                f"noised images, and noise level. Got {input_array.shape[-1]} "
                f"features from condition={condition.shape[-1]}, "
                f"noised_images={noised_images.shape[-1]}, noise_level=1."
            )

        return input_array

    def c_skip(self, sigma: Union[torch.Tensor, float]) -> torch.Tensor:
        """
        Calculate the skip connection strength c_skip(σ).

        From equation in Table 1 of the paper.

        Args:
            sigma: Noise level

        Returns:
            The skip connection coefficient tensor
        """
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma: Union[torch.Tensor, float]) -> torch.Tensor:
        """
        Calculate the output scaling factor c_out(σ).

        From equation in Table 1 of the paper.

        Args:
            sigma: Noise level

        Returns:
            The output scaling coefficient tensor
        """
        return sigma * self.sigma_data * (self.sigma_data**2 + sigma**2) ** -0.5

    def c_in(self, sigma: Union[torch.Tensor, float]) -> torch.Tensor:
        """
        Calculate the input scaling factor c_in(σ).

        From equation in Table 1 of the paper.

        Args:
            sigma: Noise level

        Returns:
            The input scaling coefficient tensor
        """
        return 1 * (sigma**2 + self.sigma_data**2) ** -0.5

    def c_noise(self, sigma: Union[torch.Tensor, float]) -> torch.Tensor:
        """
        Calculate the noise level embedding c_noise(σ).

        From equation in Table 1 of the paper.

        Args:
            sigma: Noise level

        Returns:
            The noise level embedding tensor
        """
        return log(sigma) * 0.25

    def preconditioned_network_forward(
        self,
        noised_images: torch.Tensor,
        sigma: Union[torch.Tensor, float],
        self_cond: Optional[torch.Tensor] = None,
        clamp: bool = False,
    ) -> torch.Tensor:
        """
        Apply the preconditioned network forward pass.

        Public tensors are arrays shaped (batch, 128*128, channels). This method
        calls GAOT with the concatenated per-point features as input_array and
        the mesh as query_points.

        Implements equation (7) from the paper: D_θ(x, σ) = c_skip(σ)·x + c_out(σ)·F_θ(c_in(σ)·x, c_noise(σ))

        Args:
            noised_images: Input noisy images
            sigma: Noise level(s)
            self_cond: Optional self-conditioning input
            clamp: Whether to clamp outputs to [-1, 1]

        Returns:
            Denoised image prediction
        """
        self._assert_valid_image_array(noised_images)

        batch, device = noised_images.shape[0], noised_images.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        # print(self_cond.shape, noised_images.shape, sigma.shape)
        input_array = self._prepare_gaot_input_array(
            self_cond, noised_images, sigma, batch, device
        )
        query_points = self._mesh_for_batch(batch, device, noised_images.dtype)

        net_out = self.net.forward_points(
            query_points=query_points,
            input_array=input_array,
        )

        out = (
            self.c_skip(padded_sigma) * noised_images
            + self.c_out(padded_sigma) * net_out
        )

        if clamp:
            out = out.clamp(-1.0, 1.0)

        return out

    def sample_schedule(self, num_sample_steps: Optional[int] = None) -> torch.Tensor:
        """
        Generate the noise level schedule for sampling.

        Implements equation (5) from the paper for discretized noise levels.

        Args:
            num_sample_steps: Number of sampling steps (default: self.num_sample_steps)

        Returns:
            Tensor of sigma values for each sampling step
        """
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device=self.device, dtype=torch.float32)
        sigmas = (
            self.sigma_max**inv_rho
            + steps / (N - 1) * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value=0.0)
        return sigmas

    @torch.no_grad()
    def sample(
        self,
        self_cond: torch.Tensor,
        batch_size: Optional[int] = None,
        num_sample_steps: Optional[int] = None,
        clamp: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples using the standard sampling procedure.

        This copy represents generated images as point arrays, so `self_cond`
        should be shaped (batch, 128*128, channels), and the returned samples
        have the same array shape. Internally, sampling noise stays in that
        array layout and is passed to GAOT through `preconditioned_network_forward`.

        Args:
            self_cond: GAOT condition features containing the initial condition
                and PINO output
            batch_size: Number of samples to generate (default: derived from self_cond)
            num_sample_steps: Number of sampling steps (default: self.num_sample_steps)
            clamp: Whether to clamp final outputs to [-1, 1]

        Returns:
            Generated image samples
        """
        batch_size = self_cond.shape[0]
        self._prepare_condition_features(self_cond, batch_size, self_cond.device)
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        shape = (batch_size, self.num_points, self.channels)

        sigmas = self.sample_schedule(num_sample_steps)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, sqrt(2) - 1),
            0.0,
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        init_sigma = sigmas[0]

        images = init_sigma * torch.randn(shape, device=self.device)

        for sigma, sigma_next, gamma in tqdm(
            sigmas_and_gammas, desc="sampling time step"
        ):
            sigma, sigma_next, gamma = map(
                lambda t: t.item(), (sigma, sigma_next, gamma)
            )

            eps = self.S_noise * torch.randn(shape, device=self.device)

            sigma_hat = sigma + gamma * sigma
            images_hat = images + sqrt(sigma_hat**2 - sigma**2) * eps

            model_output = self.preconditioned_network_forward(
                images_hat, sigma_hat, self_cond, clamp=clamp
            )
            denoised_over_sigma = (images_hat - model_output) / sigma_hat

            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            if sigma_next != 0:
                model_output_next = self.preconditioned_network_forward(
                    images_next, sigma_next, self_cond, clamp=clamp
                )
                denoised_prime_over_sigma = (
                    images_next - model_output_next
                ) / sigma_next
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (
                    denoised_over_sigma + denoised_prime_over_sigma
                )

            images = images_next

        if clamp:
            images = images.clamp(-1.0, 1.0)
        return images

    @torch.no_grad()
    def sample_using_dpmpp(
        self,
        self_cond: torch.Tensor,
        batch_size: Optional[int] = None,
        num_sample_steps: Optional[int] = None,
        clamp: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples using the DPM++ solver.

        This method now accepts and returns flattened image arrays shaped
        (batch, 128*128, channels). Each solver update operates on that array
        representation; the denoising network conversion is isolated to
        `preconditioned_network_forward`.

        Implements the DPM++ solver from "DPM-Solver: A Fast ODE Solver for Diffusion
        Probabilistic Model Sampling in Around 10 Steps" for improved sampling quality.

        Thanks to Katherine Crowson (https://github.com/crowsonkb) for figuring it all out!
        https://arxiv.org/abs/2211.01095

        Args:
            self_cond: GAOT condition features containing the initial condition
                and PINO output
            batch_size: Number of samples to generate (default: derived from self_cond)
            num_sample_steps: Number of sampling steps (default: self.num_sample_steps)

        Returns:
            Generated image samples
        """
        batch_size = self_cond.shape[0]
        self._prepare_condition_features(self_cond, batch_size, self_cond.device)
        device, num_sample_steps = (
            self.device,
            default(num_sample_steps, self.num_sample_steps),
        )

        sigmas = self.sample_schedule(num_sample_steps)

        shape = (batch_size, self.num_points, self.channels)
        images = sigmas[0] * torch.randn(shape, device=device)

        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        old_denoised = None
        for i in tqdm(range(len(sigmas) - 1)):
            denoised = self.preconditioned_network_forward(
                images, sigmas[i].item(), self_cond, clamp=clamp
            )
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t

            if not exists(old_denoised) or sigmas[i + 1] == 0:
                denoised_d = denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                gamma = -1 / (2 * r)
                denoised_d = (1 - gamma) * denoised + gamma * old_denoised

            images = (sigma_fn(t_next) / sigma_fn(t)) * images - (
                -h
            ).expm1() * denoised_d
            old_denoised = denoised

        if clamp:
            images = images.clamp(-1.0, 1.0)
        return images

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Calculate the weighting factor for the loss function.

        This implements the λ(σ) weighting from the paper.

        Args:
            sigma: Noise level

        Returns:
            Loss weighting factor
        """
        return (sigma**2 + self.sigma_data**2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size: int) -> torch.Tensor:
        """
        Sample noise levels from the training noise distribution.

        Samples from a log-normal distribution as described in the paper.

        Args:
            batch_size: Number of noise levels to sample

        Returns:
            Batch of noise level values
        """
        return (
            self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)
        ).exp()

    def forward(
        self, images: torch.Tensor, self_cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Training inputs are now expected as arrays shaped
        (batch, 128*128, channels). Noise is added in this point-array layout,
        and GAOT uses the DINO mesh plus the concatenated condition/noisy/noise
        feature array.

        Args:
            images: Clean input images
            self_cond: Required GAOT condition features from the converter

        Returns:
            Mean loss value

        Raises:
            AssertionError: If image dimensions don't match expected size
        """
        self._assert_valid_image_array(images)

        batch_size = images.shape[0]

        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = rearrange(sigmas, "b -> b 1 1")

        noise = torch.randn_like(images)

        noised_images = images + padded_sigmas * noise

        denoised = self.preconditioned_network_forward(noised_images, sigmas, self_cond)

        losses = F.mse_loss(denoised, images, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")

        losses = losses * self.loss_weight(sigmas)

        return losses.mean()


@register_diffusion_model(model_type="elucidated")
def elucidated_diffusion_default(model_params: Dict[str, Any]) -> ElucidatedDiffusion:
    """
    Create a default ElucidatedDiffusion model with a GAOT backbone.

    This function constructs a GAOT model that consumes DINO-style point arrays:
    the mesh is passed as query_points, and the initial condition, PINO output,
    noised images, and noise level are concatenated into input_array.

    Args:
        model_params: Dictionary of model parameters

    Returns:
        Initialized ElucidatedDiffusion model
    """
    from GAOT.src.model.gaot import GAOT
    from GAOT.src.model.layers.attn import AttentionConfig, TransformerConfig
    from GAOT.src.model.layers.magno import MAGNOConfig

    channels = model_params.get("channels", 2)
    gaot_input_size = model_params.get("gaot_input_size", 9)
    gaot_config = _make_gaot_config(
        model_params, MAGNOConfig, AttentionConfig, TransformerConfig
    )

    gaot = GAOT(
        input_size=gaot_input_size,
        output_size=model_params.get("output_size", channels),
        config=gaot_config,
    )

    model = ElucidatedDiffusion(
        gaot,
        image_size=model_params.get("image_size", 128),
        channels=channels,
        num_sample_steps=model_params.get("num_sample_steps", 32),
        sigma_min=model_params.get("sigma_min", 0.002),
        sigma_max=model_params.get("sigma_max", 80),
        sigma_data=model_params.get("sigma_data", 0.5),
        rho=model_params.get("rho", 7),
        P_mean=model_params.get("P_mean", -1.2),
        P_std=model_params.get("P_std", 1.2),
        S_churn=model_params.get("S_churn", 80),
        S_tmin=model_params.get("S_tmin", 0.05),
        S_tmax=model_params.get("S_tmax", 50),
        S_noise=model_params.get("S_noise", 1.003),
        gaot_input_size=gaot_input_size,
    )

    return model


def _make_gaot_config(
    model_params: Dict[str, Any],
    magno_config_cls,
    attention_config_cls,
    transformer_config_cls,
) -> SimpleNamespace:
    """Build the GAOT config namespace expected by GAOT.__init__."""
    magno_params = model_params.get("magno", {})
    transformer_params = model_params.get("transformer", {})
    attn_params = transformer_params.get("attn_config", {})

    magno_config = magno_config_cls(
        coord_dim=model_params.get("coord_dim", magno_params.get("coord_dim", 2)),
        radius=model_params.get("radius", magno_params.get("radius", 0.033)),
        hidden_size=model_params.get(
            "magno_hidden_size",
            magno_params.get("hidden_size", model_params.get("n_hidden", 64)),
        ),
        mlp_layers=model_params.get(
            "magno_mlp_layers",
            magno_params.get("mlp_layers", model_params.get("mlp_layers", 3)),
        ),
        lifting_channels=model_params.get(
            "lifting_channels", magno_params.get("lifting_channels", 64)
        ),
        scales=magno_params.get("scales", model_params.get("scales", [1.0])),
        use_scale_weights=magno_params.get(
            "use_scale_weights", model_params.get("use_scale_weights", False)
        ),
        use_attention=magno_params.get(
            "use_attention", model_params.get("use_attention", True)
        ),
        attention_type=magno_params.get(
            "attention_type", model_params.get("attention_type", "cosine")
        ),
        use_geoembed=magno_params.get(
            "use_geoembed", model_params.get("use_geoembed", True)
        ),
        embedding_method=magno_params.get(
            "embedding_method", model_params.get("embedding_method", "statistical")
        ),
        pooling=magno_params.get("pooling", model_params.get("pooling", "max")),
        transform_type=magno_params.get(
            "transform_type", model_params.get("transform_type", "linear")
        ),
        sampling_strategy=magno_params.get(
            "sampling_strategy", model_params.get("sampling_strategy")
        ),
        max_neighbors=magno_params.get(
            "max_neighbors", model_params.get("max_neighbors")
        ),
        sample_ratio=magno_params.get("sample_ratio", model_params.get("sample_ratio")),
        node_embedding=magno_params.get(
            "node_embedding", model_params.get("node_embedding", False)
        ),
        neighbor_search_method=magno_params.get(
            "neighbor_search_method",
            model_params.get("neighbor_search_method", "auto"),
        ),
        use_torch_scatter=magno_params.get(
            "use_torch_scatter", model_params.get("use_torch_scatter", True)
        ),
        neighbor_strategy=magno_params.get(
            "neighbor_strategy", model_params.get("neighbor_strategy", "radius")
        ),
        precompute_edges=magno_params.get(
            "precompute_edges", model_params.get("precompute_edges", False)
        ),
    )

    n_heads = model_params.get("n_head", model_params.get("n_heads", 8))
    attn_config = attention_config_cls(
        num_heads=attn_params.get(
            "num_heads", n_heads
        ),
        num_kv_heads=attn_params.get(
            "num_kv_heads",
            model_params.get("num_kv_heads", n_heads),
        ),
        use_conditional_norm=attn_params.get("use_conditional_norm", False),
        cond_norm_hidden_size=attn_params.get("cond_norm_hidden_size", 4),
        atten_dropout=attn_params.get(
            "atten_dropout", model_params.get("attn_dropout", 0.0)
        ),
    )

    transformer_config = transformer_config_cls(
        patch_size=transformer_params.get(
            "patch_size", model_params.get("patch_size", 2)
        ),
        hidden_size=transformer_params.get(
            "hidden_size", model_params.get("transformer_hidden_size", 256)
        ),
        use_attn_norm=transformer_params.get("use_attn_norm", True),
        use_ffn_norm=transformer_params.get("use_ffn_norm", True),
        norm_eps=transformer_params.get("norm_eps", 1e-6),
        num_layers=transformer_params.get(
            "num_layers", model_params.get("n_layers", 3)
        ),
        positional_embedding=transformer_params.get(
            "positional_embedding", model_params.get("positional_embedding", "absolute")
        ),
        use_long_range_skip=transformer_params.get("use_long_range_skip", True),
        ffn_multiplier=transformer_params.get("ffn_multiplier", 4),
        attn_config=attn_config,
    )

    return SimpleNamespace(
        name=model_params.get("gaot_name", "gaot"),
        latent_tokens_size=tuple(model_params.get("latent_tokens_size", (64, 64))),
        args=SimpleNamespace(magno=magno_config, transformer=transformer_config),
    )
