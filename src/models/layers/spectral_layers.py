import torch
import torch.nn as nn
import tltorch


@torch.jit.script
def contract_3D(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bixyz,ioxyz->boxyz", a, b)
    return res


class FactorizedSpectralConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        modes1,
        modes2,
        modes3,
        n_layers=4,
        bias=True,
        scale="auto",
        fft_norm="backward",
        mlp=False,
        rank=0.5,
        factorization="cp",
        fixed_rank_modes=None,
        decomposition_kwargs=dict(),
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.rank = rank
        self.factorization = factorization
        self.n_layers = n_layers
        self.fft_norm = fft_norm
        if mlp:
            raise NotImplementedError()
        else:
            self.mlp = None

        if scale == "auto":
            scale = 1 / (in_channels * out_channels)

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                fixed_rank_modes = [0]
            else:
                fixed_rank_modes = None

        if factorization is not None:
            # Convert fractional rank to an integer if needed
            actual_rank = rank
            if isinstance(rank, float) and 0 < rank < 1:
                # Calculate rank based on tensor shape
                min_dim = min(in_channels, out_channels, modes1, modes2, modes3)
                actual_rank = max(1, int(min_dim * rank))

            self.weight = tltorch.FactorizedTensor.new(
                (
                    4 * n_layers,
                    in_channels,
                    out_channels,
                    self.modes1,
                    self.modes2,
                    self.modes3,
                ),
                rank=actual_rank,  # Use the integer rank
                factorization=factorization,
                dtype=torch.cfloat,
                fixed_rank_modes=fixed_rank_modes,
                **decomposition_kwargs,
            )
            self.weight.normal_(0, scale)
            self._get_weight = self._get_weight_factorized
        else:
            self.weight = nn.Parameter(
                scale
                * torch.randn(
                    4 * n_layers,
                    in_channels,
                    out_channels,
                    self.modes1,
                    self.modes2,
                    self.modes3,
                    dtype=torch.cfloat,
                )
            )
            self._get_weight = self._get_weight_dense

        if bias:
            self.bias = nn.Parameter(scale * torch.randn(self.out_channels, 1, 1, 1))
        else:
            self.bias = 0

    def _get_weight_factorized(self, layer_index, corner_index):
        """Get the weights corresponding to a particular layer,
        corner of the Fourier coefficient (top=0 or bottom=1) -- corresponding to lower frequencies
        and complex_index (real=0 or imaginary=1)
        """
        return (
            self.weight()[4 * layer_index + corner_index, :, :, :, :, :]
            .to_tensor()
            .contiguous()
        )

    def _get_weight_dense(self, layer_index, corner_index):
        """Get the weights corresponding to a particular layer,
        corner of the Fourier coefficient (top=0 or bottom=1) -- corresponding to lower frequencies
        and complex_index (real=0 or imaginary=1)
        """
        return self.weight[4 * layer_index + corner_index, :, :, :, :, :]

    def forward(self, x, indices=0):
        with torch.autocast(device_type="cuda", enabled=False):
            batchsize, channels, height, width, depth = x.shape
            dtype = x.dtype
            # out_fft = torch.zeros(x.shape, device=x.device)

            # Compute Fourier coeffcients
            # x = torch.fft.rfftn(x.float(), norm=self.fft_norm, dim=[-3, -2, -1])
            x_fft = torch.fft.rfftn(x.float(), norm=self.fft_norm, dim=[2, 3, 4])
            # Multiply relevant Fourier modes
            # x_fft = torch.view_as_real(x_fft)
            # The output will be of size (batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1)
            out_fft = torch.zeros(
                [batchsize, self.out_channels, height, width, depth // 2 + 1],
                device=x.device,
                dtype=torch.cfloat,
            )
            out_fft[:, :, : self.modes1, : self.modes2, : self.modes3] = contract_3D(
                x_fft[:, :, : self.modes1, : self.modes2, : self.modes3],
                self._get_weight(indices, 0),
            )
            out_fft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = contract_3D(
                x_fft[:, :, -self.modes1 :, : self.modes2, : self.modes3],
                self._get_weight(indices, 1),
            )
            out_fft[:, :, : self.modes1 :, -self.modes2 :, : self.modes3] = contract_3D(
                x_fft[:, :, : self.modes1 :, -self.modes2 :, : self.modes3],
                self._get_weight(indices, 2),
            )
            out_fft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = contract_3D(
                x_fft[:, :, -self.modes1 :, : self.modes2, : self.modes3],
                self._get_weight(indices, 1),
            )
            out_fft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = contract_3D(
                x_fft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3],
                self._get_weight(indices, 3),
            )

            # out_size = (int(height*super_res), int(width*super_res))
            x = torch.fft.irfftn(
                out_fft, s=(height, width, depth), norm=self.fft_norm
            ).type(dtype)  # (x.size(-2), x.size(-1))) +
            x = x + self.bias

        if self.mlp is not None:
            x = self.mlp(x)

        return x

    def get_conv(self, indices):
        """Returns a sub-convolutional layer from the joint parametrize main-convolution
        The parametrization of sub-convolutional layers is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single convolution is parametrized, directly use the main class."
            )

        return SubConv(self, indices)

    def __getitem__(self, indices):
        return self.get_conv(indices)
