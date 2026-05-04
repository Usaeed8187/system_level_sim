import torch
from typing import Optional, Union

from sionna.phy import config, dtypes
from sionna.phy.ofdm import RZFPrecodedChannel
from sionna.phy.utils import expand_to_rank


def stream_slnr_precoding_matrix(
    h: torch.Tensor,
    alpha: Union[float, torch.Tensor] = 0.0,
    precision: Optional[str] = None,
) -> torch.Tensor:
    """Compute stream-level SLNR precoding matrix.

    Parameters
    ----------
    h : torch.Tensor
        Desired channel tensors with shape [..., num_streams_per_tx, num_tx_ant].
    alpha : float | torch.Tensor
        Regularization term added to leakage covariance.
    precision : Optional[str]
        Sionna precision identifier.

    Returns
    -------
    g : torch.Tensor
        SLNR precoding matrices with shape [..., num_tx_ant, num_streams_per_tx].
    """
    if precision is None:
        cdtype = config.cdtype
    else:
        cdtype = dtypes[precision]["torch"]["cdtype"]

    h = h.to(dtype=cdtype)
    alpha = torch.as_tensor(alpha, dtype=cdtype, device=h.device)

    # h: [..., S, Nt]
    s = h.shape[-2]
    nt = h.shape[-1]

    # Total stream covariance at TX side: sum_m h_m^H h_m
    total_cov = h.mH @ h  # [..., Nt, Nt]

    alpha = expand_to_rank(alpha, total_cov.dim(), axis=-1)
    eye = torch.eye(nt, dtype=cdtype, device=h.device)
    eye = expand_to_rank(eye, total_cov.dim(), axis=0)

    cols = []
    for si in range(s):
        hs = h[..., si, :]  # [..., Nt]
        hs_col = hs.unsqueeze(-1)  # [..., Nt, 1]
        signal_cov = hs_col @ hs_col.mH
        leak_cov = total_cov - signal_cov + alpha * eye

        # Dominant generalized-eigenvector for rank-1 signal is proportional to
        # leak_cov^{-1} h_s^H
        ws = torch.linalg.solve(leak_cov, hs_col).squeeze(-1)

        # Unit-norm per stream
        norm = torch.sqrt((ws.abs() ** 2).sum(dim=-1, keepdim=True))
        ws = torch.where(norm > 0, ws / norm, ws)
        cols.append(ws)

    g = torch.stack(cols, dim=-1)  # [..., Nt, S]
    return g


class StreamSLNRPrecodedChannel(RZFPrecodedChannel):
    """Compute effective channel after stream-level SLNR precoding.

    The class mirrors the call signature and output shape of RZFPrecodedChannel.
    """

    def call(
        self,
        h: torch.Tensor,
        tx_power: torch.Tensor,
        h_hat: Optional[torch.Tensor] = None,
        alpha: Union[float, torch.Tensor] = 0.0,
    ) -> torch.Tensor:
        if h_hat is None:
            h_hat = h

        # [B, Tx, Ofdm, Sc, S, Nt]
        h_pc_desired = self.get_desired_channels(h_hat)

        alpha = torch.as_tensor(alpha, dtype=self.dtype, device=self.device)
        alpha = expand_to_rank(alpha, 4, axis=-1)
        alpha = torch.broadcast_to(alpha, h_pc_desired.shape[:4])

        # [B, Tx, Ofdm, Sc, Nt, S]
        g = stream_slnr_precoding_matrix(h_pc_desired, alpha=alpha, precision=self.precision)

        g = self.apply_tx_power(g, tx_power)
        h_eff = self.compute_effective_channel(h, g)
        return h_eff