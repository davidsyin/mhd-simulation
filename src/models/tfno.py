# Adapted from with significant functional changes:
# https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/mhd_pino/tfno
# which is based on arXiv:2302.08332 (2023).
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import physicsnemo.models.layers as layers
from .layers import (
    FactorizedSpectralConv3d,
)
from physicsnemo.models.mlp import FullyConnected

from .model_factory import register_model


class TFNO3DEncoder(nn.Module):
    """3D Spectral encoder for TFNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    rank : float, optional
        Rank of the decomposition, by default 1.0
    factorization : {'CP', 'TT', 'Tucker'}, optional
        Tensor factorization to use to decompose the tensor, by default 'CP'
    fixed_rank_modes : List[int], optional
        A list of modes for which the initial value is not modified, by default None
        The last mode cannot be fixed due to error computation.
    decomposition_kwargs : dict, optional
        Additional arguments to initialization of factorized tensors, by default dict()
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_fno_layers: int = 4,
        fno_layer_size: int = 32,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
        rank: float = 1.0,
        factorization: str = "cp",
        fixed_rank_modes: List[int] = None,
        decomposition_kwargs: dict = dict(),
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        self.activation_fn = activation_fn

        # TensorLy arguments
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs

        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 3

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding, padding, padding]
        padding = padding + [0, 0, 0]  # Pad with zeros for smaller lists
        self.pad = padding[:3]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes, num_fno_modes, num_fno_modes]

        # build lift
        self.build_lift_network()
        self.build_fno(num_fno_modes)

    def build_lift_network(self) -> None:
        """construct network for lifting variables to latent space."""
        # Initial lift network
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(
            layers.Conv3dFCLayer(self.in_channels, int(self.fno_width / 2))
        )
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(
            layers.Conv3dFCLayer(int(self.fno_width / 2), self.fno_width)
        )

    def build_fno(self, num_fno_modes: List[int]) -> None:
        """construct FNO block."""
        # Build Neural Fourier Operators
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()

        # Convert rank to integer if it's a fractional value
        actual_rank = self.rank
        if isinstance(self.rank, float) and 0 < self.rank < 1:
            min_dim = min(
                self.fno_width,
                self.fno_width,
                num_fno_modes[0],
                num_fno_modes[1],
                num_fno_modes[2],
            )
            actual_rank = max(1, int(min_dim * self.rank))

        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(
                FactorizedSpectralConv3d(
                    self.fno_width,
                    self.fno_width,
                    num_fno_modes[0],
                    num_fno_modes[1],
                    num_fno_modes[2],
                    n_layers=1,
                    bias=False,
                    scale="auto",
                    rank=actual_rank,
                    factorization=self.factorization,
                    fixed_rank_modes=self.fixed_rank_modes,
                    decomposition_kwargs=self.decomposition_kwargs,
                )
            )
            self.conv_layers.append(nn.Conv3d(self.fno_width, self.fno_width, 1))

    def forward(self, x: Tensor) -> Tensor:
        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_network(x)

        x = F.pad(
            x, (0, self.pad[2], 0, self.pad[1], 0, self.pad[0]), mode=self.padding_type
        )

        for i in range(self.num_fno_layers):
            x1 = self.spconv_layers[i](x)
            x2 = self.conv_layers[i](x)

            x = x1 + x2

            if i < len(self.conv_layers) - 1:
                x = self.activation_fn(x)

        x = x[..., : self.ipad[0], : self.ipad[1], : self.ipad[2]]
        return x

    # Uncomment this if you prefer to implement coord_features/grid-embedding
    # inside your model instead of dataset

    def meshgrid(self, shape: List[int], device: torch.device) -> Tensor:
        """Creates 3D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x, size_y, size_z = shape[0], shape[2], shape[3], shape[4]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_z = torch.linspace(0, 1, size_z, dtype=torch.float32, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        grid_z = grid_z.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        return torch.cat((grid_x, grid_y, grid_z), dim=1)

    def grid_to_points(self, value: Tensor) -> Tuple[Tensor, List[int]]:
        """converting from grid based (image) to point based representation

        Parameters
        ----------
        value : Meshgrid tensor

        Returns
        -------
        Tuple
            Tensor, meshgrid shape
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 3, 4, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: Tensor, shape: List[int]) -> Tensor:
        """converting from point based to grid based (image) representation

        Parameters
        ----------
        value : Tensor
            Tensor
        shape : List[int]
            meshgrid shape

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        output = value.reshape(shape[0], shape[2], shape[3], shape[4], value.size(-1))
        return torch.permute(output, (0, 4, 1, 2, 3))


# ===================================================================
# ===================================================================
# General TFNO Model
# ===================================================================
# ===================================================================


class TFNO(nn.Module):
    """Tensor Factorized Fourier neural operator (FNO) model.

    Note
    ----
    The TFNO architecture supports options for 1D, 2D, 3D and 4D fields which can
    be controlled using the `dimension` parameter.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    decoder_layers : int, optional
        Number of decoder layers, by default 1
    decoder_layer_size : int, optional
        Number of neurons in decoder layers, by default 32
    decoder_activation_fn : str, optional
        Activation function for decoder, by default "silu"
    dimension : int
        Model dimensionality (supports 1, 2, 3).
    latent_channels : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding : int, optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : str, optional
        Activation function, by default "gelu"
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    rank : float, optional
        Rank of the decomposition, by default 1.0
    factorization : {'CP', 'TT', 'Tucker'}, optional
        Tensor factorization to use to decompose the tensor, by default 'CP'
    fixed_rank_modes : List[int], optional
        A list of modes for which the initial value is not modified, by default None
        The last mode cannot be fixed due to error computation.
    decomposition_kwargs : dict, optional
        Additional arguments to initialization of factorized tensors, by default dict()

    Example
    -------
    >>> # define the 2d TFNO model
    >>> model = physicsnemo.models.fno.TFNO(
    ...     in_channels=4,
    ...     out_channels=3,
    ...     decoder_layers=2,
    ...     decoder_layer_size=32,
    ...     dimension=2,
    ...     latent_channels=32,
    ...     num_fno_layers=2,
    ...     padding=0,
    ... )
    >>> input = torch.randn(32, 4, 32, 32) #(N, C, H, W)
    >>> output = model(input)
    >>> output.size()
    torch.Size([32, 3, 32, 32])

    Note
    ----
    Reference: Rosofsky, Shawn G. and Huerta, E. A. "Magnetohydrodynamics with
    Physics Informed Neural Operators." arXiv preprint arXiv:2302.08332 (2023).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        decoder_layers: int = 1,
        decoder_layer_size: int = 32,
        decoder_activation_fn: str = "silu",
        dimension: int = 2,
        latent_channels: int = 32,
        num_fno_layers: int = 4,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: int = 8,
        padding_type: str = "constant",
        activation_fn: str = "gelu",
        coord_features: bool = True,
        rank: float = 1.0,
        factorization: str = "cp",
        fixed_rank_modes: List[int] = None,
        decomposition_kwargs: dict = dict(),
    ) -> None:
        super().__init__()

        self.num_fno_layers = num_fno_layers
        self.num_fno_modes = num_fno_modes
        self.padding = padding
        self.padding_type = padding_type
        self.activation_fn = layers.get_activation(activation_fn)
        self.coord_features = coord_features
        self.dimension = dimension

        # TensorLy arguments
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs

        # decoder net
        self.decoder_net = FullyConnected(
            in_features=latent_channels,
            layer_size=decoder_layer_size,
            out_features=out_channels,
            num_layers=decoder_layers,
            activation_fn=decoder_activation_fn,
        )

        TFNOModel = self.getTFNOEncoder()

        self.spec_encoder = TFNOModel(
            in_channels,
            num_fno_layers=self.num_fno_layers,
            fno_layer_size=latent_channels,
            num_fno_modes=self.num_fno_modes,
            padding=self.padding,
            padding_type=self.padding_type,
            activation_fn=self.activation_fn,
            rank=self.rank,
            factorization=self.factorization,
            fixed_rank_modes=self.fixed_rank_modes,
            decomposition_kwargs=self.decomposition_kwargs,
        )

    def getTFNOEncoder(self):
        "Return correct TFNO ND Encoder"
        if self.dimension == 3:
            return TFNO3DEncoder
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        # Fourier encoder
        # print(f"\nx.shape before tfno: {x.shape}")
        y_latent = self.spec_encoder(x)
        # print(f"y_latent shape after tfno: {y_latent.shape}")

        # Reshape to pointwise inputs if not a conv FC model
        y_shape = y_latent.shape
        y_latent, y_shape = self.spec_encoder.grid_to_points(y_latent)
        # print(f"y_latent shape after grid to points: {y_latent.shape}")

        # Decoder
        y = self.decoder_net(y_latent)
        # print(f"y shape after decoder net: {y.shape}")

        # Convert back into grid
        y = self.spec_encoder.points_to_grid(y, y_shape)
        # print(f"y shape after points to grid: {y.shape}\n")

        return y


# Register TFNO models with the repository's registry system
@register_model(model_type="tfno", variant=None)
def tfno_model_default(params):
    """Default TFNO model."""
    return TFNO(
        in_channels=params.get("in_channels", 6),
        out_channels=params.get("out_channels", 3),
        decoder_layers=params.get("decoder_layers", 1),
        dimension=params.get("dimension", 3),
        latent_channels=params.get("latent_channels", 32),
        num_fno_layers=params.get("num_fno_layers", 4),
        num_fno_modes=params.get("num_fno_modes", 16),
        padding=params.get("padding", 8),
        padding_type=params.get("padding_type", "constant"),
        activation_fn=params.get("activation_fn", "gelu"),
        coord_features=params.get("coord_features", False),
        rank=params.get("rank", 1.0),
        factorization=params.get("factorization", "cp"),
        fixed_rank_modes=params.get("fixed_rank_modes", None),
        decomposition_kwargs=params.get("decomposition_kwargs", {}),
    )


@register_model(model_type="tfno", variant="3d")
def tfno_model_3d(params):
    """3D TFNO model."""
    params["dimension"] = 3
    return tfno_model_default(params)
