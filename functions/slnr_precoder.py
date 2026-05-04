import torch
from typing import Optional, Union

from sionna.phy import config, dtypes
from sionna.phy.ofdm import RZFPrecodedChannel
from sionna.phy.utils import expand_to_rank


def ue_slnr_precoding_matrix(
    h: torch.Tensor,
    num_streams_per_ue: int,
    alpha: Union[float, torch.Tensor] = 0.0,
    precision: Optional[str] = None,
) -> torch.Tensor:
    """Compute UE-level SLNR precoding matrix.

    Parameters
    ----------
    h : torch.Tensor
        Desired channel tensors with shape [..., num_streams_total, num_tx_ant].
    num_streams_per_ue : int
        Number of streams assigned to each UE.
    alpha : float | torch.Tensor
        Regularization term added to leakage covariance.
    precision : Optional[str]
        Sionna precision identifier.

    Returns
    -------
    g : torch.Tensor
        SLNR precoding matrices with shape [..., num_tx_ant, num_streams_total].
    """
    if precision is None:
        cdtype = config.cdtype
    else:
        cdtype = dtypes[precision]["torch"]["cdtype"]

    h = h.to(dtype=cdtype)
    alpha = torch.as_tensor(alpha, dtype=cdtype, device=h.device)

    s_total = h.shape[-2]
    nt = h.shape[-1]
    if s_total % num_streams_per_ue != 0:
        raise ValueError(
            f"num_streams_total={s_total} must be divisible by num_streams_per_ue={num_streams_per_ue}"
        )
    num_ues = s_total // num_streams_per_ue

    total_cov = h.mH @ h  # [..., Nt, Nt]

    alpha = expand_to_rank(alpha, total_cov.dim(), axis=-1)
    eye = torch.eye(nt, dtype=cdtype, device=h.device)
    eye = expand_to_rank(eye, total_cov.dim(), axis=0)

    cols = []
    for ue in range(num_ues):
        start = ue * num_streams_per_ue
        end = (ue + 1) * num_streams_per_ue
        h_u = h[..., start:end, :]  # [..., d, Nt]

        signal_cov = h_u.mH @ h_u  # [..., Nt, Nt]
        leak_cov = total_cov - signal_cov + alpha * eye

        # Generalized eigenvectors of (signal_cov, leak_cov):
        # equivalent to eig(leak_cov^{-1} signal_cov)
        a = torch.linalg.solve(leak_cov, signal_cov)
        evals, evecs = torch.linalg.eig(a)
        idx = torch.argsort(evals.real, dim=-1, descending=True)[..., :num_streams_per_ue]
        top = torch.take_along_dim(
            evecs, idx.unsqueeze(-2).expand(*evecs.shape[:-1], num_streams_per_ue), dim=-1
        )

        # Unit-norm columns
        norm = torch.sqrt(torch.sum(top.abs() ** 2, dim=-2, keepdim=True))
        top = torch.where(norm > 0, top / norm, top)
        cols.append(top)

    g = torch.cat(cols, dim=-1)  # [..., Nt, S_total]

    return g


class UESLNRPrecodedChannel(RZFPrecodedChannel):
    """Compute effective channel after UE-level SLNR precoding.

    The class mirrors the call signature and output shape of RZFPrecodedChannel.
    """

    def call(
        self,
        h: torch.Tensor,
        tx_power: torch.Tensor,
        h_hat: Optional[torch.Tensor] = None,
        alpha: Union[float, torch.Tensor] = 0.0,
        num_streams_per_ue: Optional[int] = None,
    ) -> torch.Tensor:
        if h_hat is None:
            h_hat = h

        # [B, Tx, Ofdm, Sc, S, Nt]
        h_pc_desired = self.get_desired_channels(h_hat)

        alpha = torch.as_tensor(alpha, dtype=self.dtype, device=self.device)
        alpha = expand_to_rank(alpha, 4, axis=-1)
        alpha = torch.broadcast_to(alpha, h_pc_desired.shape[:4])

        if num_streams_per_ue is None:
            rg = getattr(self, "resource_grid", None)
            if rg is None:
                rg = getattr(self, "_resource_grid", None)
            if rg is None or not hasattr(rg, "num_streams_per_tx"):
                raise ValueError(
                    "num_streams_per_ue must be provided when it cannot be inferred "
                    "from resource_grid.num_streams_per_tx."
                )
            num_streams_per_ue = int(rg.num_streams_per_tx)
        if num_streams_per_ue < 1:
            raise ValueError(f"num_streams_per_ue must be >=1, got {num_streams_per_ue}")

        # [B, Tx, Ofdm, Sc, Nt, S]
        g = ue_slnr_precoding_matrix(
            h_pc_desired,
            num_streams_per_ue=num_streams_per_ue,
            alpha=alpha,
            precision=self.precision,
        )

        g = self.apply_tx_power(g, tx_power)
        h_eff = self.compute_effective_channel(h, g)
        return h_eff