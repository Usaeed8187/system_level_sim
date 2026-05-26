import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


os.makedirs('./results', exist_ok=True)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# if os.getenv('CUDA_VISIBLE_DEVICES') is None:
#     os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use "0" or "1" to use the GPU, "" to use the CPU

import torch
try:
    import sionna.sys
except ImportError as e:
    import sys
    if 'google.colab' in sys.modules:
        print('Installing Sionna and restarting the runtime. Please run the cell again.')
        os.system('pip install sionna')
        os.kill(os.getpid(), 5)
    raise e

from sionna.phy.channel.tr38901 import PanelArray
from functions.antenna_38922 import PanelArray_38922
from sionna.phy.ofdm import ResourceGrid, RZFPrecodedChannel, LMMSEPostEqualizationSINR
from functions.slnr_precoder import UESLNRPrecodedChannel
from sionna.phy.utils import dbm_to_watt

from functions.utils import *
from functions.simulation import *


def build_simulator(num_ut_per_sector: int,
                    num_streams_per_ut: int,
                    num_subcarriers: int,
                    num_ofdm_sym: int,
                    carrier_frequency: float,
                    scenario: str,
                    direction: str,
                    num_rings: int,
                    bs_array: PanelArray_38922,
                    ut_array: PanelArray_38922,
                    bs_max_power_dbm: float,
                    ut_max_power_dbm: float,
                    min_ue_azimuth_separation_deg: float = 10.0,
                    ue_elevation_mode: str = 'None',
                    ue_elevation_angle_deg: float = None,
                    fixed_ue_height: float = 1.5):


    resource_grid = ResourceGrid(num_ofdm_symbols=num_ofdm_sym,
                                 fft_size=num_subcarriers,
                                 subcarrier_spacing=15e3,
                                 num_tx=num_ut_per_sector,
                                 num_streams_per_tx=num_streams_per_ut)

    sls = SystemLevelSimulator(
        batch_size=1,
        num_rings=num_rings,
        num_ut_per_sector=num_ut_per_sector,
        carrier_frequency=carrier_frequency,
        resource_grid=resource_grid,
        scenario=scenario,
        direction=direction,
        ut_array=ut_array,
        bs_array=bs_array,
        bs_max_power_dbm=bs_max_power_dbm,
        ut_max_power_dbm=ut_max_power_dbm,
        coherence_time=1,
        max_bs_ut_dist=80,
        min_bs_ut_dist=0,
        temperature=294,
        o2i_model='low',
        average_street_width=20.0,
        average_building_height=10.0,
        min_ue_azimuth_separation_deg=min_ue_azimuth_separation_deg,
        ue_elevation_mode=ue_elevation_mode,
        ue_elevation_angle_deg=ue_elevation_angle_deg,
        fixed_ue_height=fixed_ue_height)

    return sls

def _compute_logdet_capacity_from_precoded_channel(h_eff_target_rx: torch.Tensor,
                                                   no: torch.Tensor,
                                                   target_tx: int,
                                                   desired_stream_indices: torch.Tensor) -> torch.Tensor:
    """Compute combiner-agnostic capacity using log2 det(I + R^{-1} S)."""
    h = h_eff_target_rx.permute(0, 4, 5, 2, 3, 1)
    # [batch, ofdm, subc, num_tx, num_streams_per_tx, num_rx_ant]
    cov_per_stream = torch.einsum('...m,...n->...mn', h, torch.conj(h))
    # [batch, ofdm, subc, num_tx, num_streams_per_tx, num_rx_ant, num_rx_ant]

    signal_cov = torch.sum(cov_per_stream[:, :, :, target_tx, desired_stream_indices, :, :], dim=3)
    total_cov = torch.sum(cov_per_stream, dim=(3, 4))
    interf_cov = total_cov - signal_cov

    nr = h_eff_target_rx.shape[1]
    eye = torch.eye(nr, dtype=h_eff_target_rx.dtype, device=h_eff_target_rx.device).view(
        1, 1, 1, nr, nr)
    r_cov = interf_cov + no * eye
    cap_arg = eye + torch.linalg.solve(r_cov, signal_cov)
    _, logdet = torch.linalg.slogdet(cap_arg)
    return (logdet / np.log(2.0)).real


def compute_max_ue_channel_correlation(h_freq_fading: torch.Tensor,
                                       target_bs: int,
                                       num_ut_per_sector: int) -> np.ndarray:
    """Return max pairwise channel correlation among UEs in one sector.

    The channel vector for each UE is the serving-BS channel flattened across
    RX antennas, TX antennas, OFDM symbols, and subcarriers. The returned value
    is max_{i != j} |h_i^H h_j| / (||h_i|| ||h_j||). One value is returned per
    batch item.
    """
    if num_ut_per_sector < 2:
        return np.zeros((h_freq_fading.shape[0],), dtype=np.float32)

    target_rx_start = target_bs * num_ut_per_sector
    target_rx_end = (target_bs + 1) * num_ut_per_sector

    # [batch, num_ut_per_sector, num_rx_ant, num_tx_ant, num_ofdm, num_subc]
    h_sector = h_freq_fading[:, target_rx_start:target_rx_end, :, target_bs, :, :, :]

    # [batch, num_ut_per_sector, flattened_channel_dim]
    h_vec = torch.reshape(h_sector, [h_sector.shape[0], num_ut_per_sector, -1])

    h_norm = torch.linalg.norm(h_vec, dim=-1, keepdim=True)
    h_unit = h_vec / torch.clamp(h_norm, min=1e-30)

    # [batch, num_ut_per_sector, num_ut_per_sector]
    corr = torch.abs(torch.matmul(h_unit, torch.conj(torch.transpose(h_unit, -1, -2))))

    eye = torch.eye(num_ut_per_sector, dtype=torch.bool, device=h_freq_fading.device)
    corr = corr.masked_fill(eye.unsqueeze(0), 0.0)
    max_corr = torch.amax(corr.real, dim=(-2, -1))

    return max_corr.detach().cpu().numpy().ravel()



def compute_min_ue_azimuth_distance_deg(sls: SystemLevelSimulator,
                                        target_sector_index: int,
                                        num_ut_per_sector: int) -> np.ndarray:
    """Return the minimum pairwise UE azimuth separation for one sector.

    The azimuth angle of each UE is computed from its serving BS/sector location
    to the UE location in the X-Y plane. The pairwise distance is the circular
    angular distance in degrees, i.e., min(|a-b|, 360-|a-b|). One value is
    returned per batch item.
    """
    if num_ut_per_sector < 2:
        return np.full((sls.batch_size,), np.nan, dtype=np.float32)

    target_bs = int(target_sector_index)
    target_rx_start = target_bs * num_ut_per_sector
    target_rx_end = (target_bs + 1) * num_ut_per_sector

    bs_xy = sls.bs_loc[:, target_bs, :2]
    ut_xy = sls.ut_loc[:, target_rx_start:target_rx_end, :2]
    rel_xy = ut_xy - bs_xy[:, None, :]

    angles = torch.atan2(rel_xy[..., 1], rel_xy[..., 0]) * (180.0 / np.pi)
    diff = torch.abs(angles[:, :, None] - angles[:, None, :])
    diff = torch.minimum(diff, 360.0 - diff)

    eye = torch.eye(num_ut_per_sector, dtype=torch.bool, device=diff.device)
    diff = diff.masked_fill(eye.unsqueeze(0), float('inf'))
    min_diff = torch.amin(diff, dim=(-2, -1))

    return min_diff.detach().cpu().numpy().ravel()

def compute_drop_log_capacity_samples(sls: SystemLevelSimulator,
                                      num_ut_per_sector: int,
                                      num_streams_per_ut: int,
                                      num_slots: int,
                                      target_sector_index: int = 0,
                                      precoder: str = 'rzf',
                                      diagnostics: bool = False,
                                      raw_interference_mode: str = 'all-other-bs'):

    if num_slots < 1:
        raise ValueError(f'num_slots must be >= 1, got {num_slots}')

    sls.channel_matrix.reset()

    h_freq = sls.channel_matrix(sls.channel_model)

    rg = sls.resource_grid
    total_tx_power_watt = dbm_to_watt(sls.bs_max_power_dbm)

    # Force MU scheduling over the full band:
    # all UEs in each sector are active on all REs.
    num_streams_per_sector = num_ut_per_sector * num_streams_per_ut
    tx_power_per_stream = total_tx_power_watt / (
        num_streams_per_sector * rg.num_ofdm_symbols * rg.fft_size)

    tx_power = torch.full(
        [sls.batch_size,
         sls.num_bs,
         1,
         num_streams_per_sector,
         rg.num_ofdm_symbols,
         rg.fft_size],
        fill_value=tx_power_per_stream,
        dtype=sls.dtype,
        device=sls.device)

    target_bs = int(target_sector_index)
    if target_bs < 0 or target_bs >= sls.num_bs:
        raise ValueError(f'target_sector_index must be in [0, {sls.num_bs - 1}], got {target_bs}')

    if precoder == 'rzf':
        precoded_channel = RZFPrecodedChannel(resource_grid=rg,
                                              stream_management=sls.stream_management)
    elif precoder == 'slnr':
        precoded_channel = UESLNRPrecodedChannel(resource_grid=rg,
                                                 stream_management=sls.stream_management)
    else:
        raise ValueError(f"Unsupported precoder '{precoder}'. Use 'rzf' or 'slnr'.")
    # Include thermal-noise loading in SLNR/RZF regularization so that
    # precoder design accounts for both leakage/interference and noise.
    precoder_alpha = torch.as_tensor(sls.no, dtype=sls.dtype, device=sls.device)
    lmmse_posteq_sinr = LMMSEPostEqualizationSINR(resource_grid=rg,
                                                  stream_management=sls.stream_management)

    # tx_power: [batch_size, num_bs, num_tx_per_sector,
    #            num_streams_per_tx, num_ofdm_sym, num_subcarriers]
    # Flatten across sectors
    # [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers]
    s = tx_power.shape
    tx_power = torch.reshape(tx_power, [s[0], s[1]*s[2]] + list(s[3:]))

    slot_stream_sum_samples = []
    slot_logdet_samples = []
    slot_raw_snr_db_samples = []
    slot_raw_sinr_db_samples = []
    slot_max_ue_chan_corr_samples = []

    if raw_interference_mode not in ['all-other-bs', 'other-cells-only']:
        raise ValueError("raw_interference_mode must be 'all-other-bs' or 'other-cells-only'")

    for slot in range(num_slots):
        h_freq = sls.channel_matrix.update(sls.channel_model, h_freq, slot)
        h_freq_fading = sls.channel_matrix.apply_fading(h_freq)

        slot_max_ue_chan_corr_samples.append(
            compute_max_ue_channel_correlation(
                h_freq_fading=h_freq_fading,
                target_bs=target_bs,
                num_ut_per_sector=num_ut_per_sector))

        # ------------------------------------------------------------------
        # Raw no-precoding SNR/SINR samples
        # ------------------------------------------------------------------
        # h_freq_fading is expected to have shape
        # [batch, num_rx, num_rx_ant, num_bs, num_tx_ant, num_ofdm_sym, num_subcarriers].
        #
        # For target UE k served by target_bs, use the scalar raw channel gain
        #     ||H_{k,j}||_F^2 = sum over rx antennas and BS tx antennas |H_{k,j}|^2.
        #
        # If bs_max_power_dbm is the total BS/sector transmit power and the
        # no-precoding transmit covariance is isotropic across TX antennas,
        #     E[x_j x_j^H] = (P_t/N_t) I,
        # then the received signal/interference powers are
        #     (P_t/N_t) ||H_{k,j}||_F^2.
        #
        # Noise is summed over receive antennas to match the summed receive-side
        # channel energy used in ||H||_F^2.
        raw_channel_gain = torch.sum(torch.abs(h_freq_fading)**2, dim=(2, 4)).real
        # [batch, num_rx, num_bs, num_ofdm_sym, num_subcarriers]

        num_rx_ant = h_freq_fading.shape[2]
        num_tx_ant = h_freq_fading.shape[4]
        raw_power_scale = total_tx_power_watt / float(num_tx_ant)

        target_rx_start = target_bs * num_ut_per_sector
        target_rx_end = (target_bs + 1) * num_ut_per_sector
        raw_target_gain_all_bs = raw_channel_gain[:, target_rx_start:target_rx_end, :, :, :]
        # [batch, num_ut_per_sector, num_bs, num_ofdm_sym, num_subcarriers]

        raw_signal_power = raw_power_scale * raw_target_gain_all_bs[:, :, target_bs, :, :]

        bs_indices = torch.arange(sls.num_bs, device=sls.device)
        if raw_interference_mode == 'other-cells-only':
            # Sionna's sectorized hexgrid uses three sectors per site/cell.
            # This excludes the other two sectors of the serving site and only
            # counts sectors from other sites/cells as inter-cell interference.
            target_site = target_bs // 3
            interference_mask = (bs_indices // 3) != target_site
        else:
            # Counts every non-serving sector/BS as interference. This is often
            # useful if num_rings=0, where only the center-site sectors exist.
            interference_mask = bs_indices != target_bs

        raw_interference_power = raw_power_scale * torch.sum(
            raw_target_gain_all_bs[:, :, interference_mask, :, :], dim=2)

        raw_noise_power = torch.as_tensor(
            sls.no, dtype=raw_signal_power.dtype, device=raw_signal_power.device) * float(num_rx_ant)

        raw_snr = raw_signal_power / torch.clamp(raw_noise_power, min=1e-30)
        raw_sinr = raw_signal_power / torch.clamp(raw_interference_power + raw_noise_power, min=1e-30)

        slot_raw_snr_db_samples.append(
            (10.0 * torch.log10(torch.clamp(raw_snr, min=1e-30))).detach().cpu().numpy().ravel())
        slot_raw_sinr_db_samples.append(
            (10.0 * torch.log10(torch.clamp(raw_sinr, min=1e-30))).detach().cpu().numpy().ravel())

        h_eff = precoded_channel(h_freq_fading, tx_power=tx_power, alpha=precoder_alpha)

        # [batch, num_ofdm_sym, num_subcarriers, num_rx, num_streams_per_rx]
        sinr = lmmse_posteq_sinr(h_eff, no=sls.no, interference_whitening=True)
        # [batch, num_ofdm_sym, num_subcarriers, num_bs, num_ut_per_sector, num_streams_per_ut]
        sinr = torch.reshape(
            sinr,
            list(sinr.shape[:-2]) + [sls.num_bs, num_ut_per_sector, num_streams_per_ut])
        # [batch, num_bs, num_ofdm_sym, num_subcarriers, num_ut_per_sector, num_streams_per_ut]
        sinr = torch.permute(sinr, [0, 3, 1, 2, 4, 5])
        target_sinr = sinr[:, target_bs, :, :, :, :]

        if diagnostics:
            sinr_neg_frac = torch.mean((target_sinr < 0).to(sls.dtype)).item()
            sinr_mean_lin = torch.mean(target_sinr).item()
            sinr_mean_db = 10.0 * np.log10(max(sinr_mean_lin, 1e-30))
            h_target = h_eff[:, target_bs*num_ut_per_sector:(target_bs+1)*num_ut_per_sector, :, :, :, :, :]
            h_target = h_target.permute(0, 1, 4, 3, 5, 6, 2)
            cov_per_stream = torch.einsum('...m,...n->...mn', h_target, torch.conj(h_target))
            total_cov = torch.sum(cov_per_stream, dim=(1, 3, 4))
            desired_cov = torch.sum(cov_per_stream[:, :, :, target_bs, :, :, :], dim=(1, 3))
            interference_cov = total_cov - desired_cov
            desired_power = torch.mean(torch.diagonal(desired_cov, dim1=-2, dim2=-1).real).item()
            interference_power = torch.mean(torch.diagonal(interference_cov, dim1=-2, dim2=-1).real).item()
            print(
                f"[diag] slot={slot} precoder={precoder} "
                f"mean|h_eff|={torch.mean(torch.abs(h_eff)).item():.3e} "
                f"mean_sinr_lin={sinr_mean_lin:.3e} mean_sinr_db={sinr_mean_db:.2f} "
                f"sinr_neg_frac={sinr_neg_frac:.3f} "
                f"desired_pow={desired_power:.3e} interference_pow={interference_power:.3e}"
            )

        # Sector sum-throughput metric:
        # sum over streams and users, per RE.
        stream_sum_rate = torch.sum(
            torch.log2(1.0 + torch.clamp(target_sinr, min=0.0)), dim=-1)
        stream_sum_rate = torch.sum(stream_sum_rate, dim=-1)
        slot_stream_sum_samples.append(stream_sum_rate.detach().cpu().numpy().ravel())

        # Combiner-agnostic alternative:
        # per-UE log-det (desired streams = that UE streams), then sum across UEs.
        logdet_rate_per_ut = []
        for ut_idx in range(num_ut_per_sector):
            target_rx = target_bs * num_ut_per_sector + ut_idx
            start = ut_idx * num_streams_per_ut
            end = (ut_idx + 1) * num_streams_per_ut
            desired_stream_indices = torch.arange(
                start, end, dtype=torch.long, device=sls.device)
            h_eff_target_rx = h_eff[:, target_rx, :, :, :, :, :]
            logdet_rate_ut = _compute_logdet_capacity_from_precoded_channel(
                h_eff_target_rx=h_eff_target_rx,
                no=sls.no,
                target_tx=target_bs,
                desired_stream_indices=desired_stream_indices)
            logdet_rate_per_ut.append(logdet_rate_ut)
        logdet_rate_sector = torch.sum(torch.stack(logdet_rate_per_ut, dim=0), dim=0)
        slot_logdet_samples.append(logdet_rate_sector.detach().cpu().numpy().ravel())

        # Match slot-wise behavior used in e2e_example.py.
        sls.ut_loc = sls.ut_loc + sls.ut_velocities * sls.slot_duration
        sls.channel_model.set_topology(
            sls.ut_loc, sls.bs_loc, sls.ut_orientations,
            sls.bs_orientations, sls.ut_velocities,
            sls.in_state, sls.los, sls.bs_virtual_loc)

    return (np.concatenate(slot_stream_sum_samples),
            np.concatenate(slot_logdet_samples),
            np.concatenate(slot_raw_snr_db_samples),
            np.concatenate(slot_raw_sinr_db_samples),
            np.concatenate(slot_max_ue_chan_corr_samples))

def save_grid_plot_for_first_drop(sls: SystemLevelSimulator, out_path: str = './results/grid.png'):
    fig = sls.grid.show()
    ax = fig.get_axes()
    ut_loc_np = sls.ut_loc.cpu().numpy() if hasattr(sls.ut_loc, 'cpu') else sls.ut_loc
    ax[0].plot(ut_loc_np[0, :, 0], ut_loc_np[0, :, 1], 'xk', label='user position')
    ax[0].legend()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f'Saved grid plot to: {out_path}')

def main():
    parser = argparse.ArgumentParser(description='MU-MIMO sector sum-throughput CDF experiment')
    parser.add_argument('--num-drops', type=int, default=10)
    parser.add_argument('--num-slots', type=int, default=10,
                        help='Number of slots simulated per drop (default: 10)')
    parser.add_argument('--num-ut-per-sector', type=int, default=6, help='Number of users per sector (default: 1)')
    parser.add_argument('--num-streams-per-ut', type=int, default=4, help='Number of spatial streams per user (default: 2)')
    parser.add_argument('--scenario', type=str, default='umi', choices=['umi', 'uma', 'rma'])
    parser.add_argument('--num-bs-horizontal-antennas', type=int, default=16, help='Number of horizontal antennas per BS panel (default: 4)')
    parser.add_argument('--num-bs-vertical-antennas', type=int, default=8, help='Number of vertical antennas per BS panel (default: 2)')
    parser.add_argument('--num-ut-horizontal-antennas', type=int, default=4, help='Number of horizontal antennas per UT panel (default: 2)')
    parser.add_argument('--num-ut-vertical-antennas', type=int, default=1, help='Number of vertical antennas per UT panel (default: 1)')
    parser.add_argument('--bs-pattern', type=str, default='38.922', choices=['38.901', '38.922'])
    parser.add_argument('--num-rings', type=int, default=0)
    parser.add_argument('--num-ofdm-sym', type=int, default=1)
    parser.add_argument('--num-subcarriers', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='./results/mu_mimo_log1p_sinr_cdf.png')
    parser.add_argument('--raw-snr-out', type=str, default='./results/mu_mimo_raw_snr_cdf.png',
                        help='Output path for the raw no-precoding SNR CDF plot')
    parser.add_argument('--raw-sinr-out', type=str, default='./results/mu_mimo_raw_sinr_cdf.png',
                        help='Output path for the raw no-precoding SINR CDF plot')
    parser.add_argument('--ue-corr-out', type=str, default='./results/UE_max_chan_corr_CDF.png',
                        help='Output path for the CDF of maximum pairwise UE channel correlation')
    parser.add_argument('--ue-min-azimuth-distance-out', type=str,
                        default='./results/UE_min_azimuth_distance_CDF.png',
                        help='Output path for the CDF of minimum pairwise UE azimuth distance in degrees')
    parser.add_argument('--raw-interference-mode', type=str, default='all-other-bs',
                        choices=['all-other-bs', 'other-cells-only'],
                        help=("How to count raw interference: 'all-other-bs' counts every "
                              "non-serving sector/BS; 'other-cells-only' excludes the "
                              "other sectors of the serving 3-sector site."))
    parser.add_argument('--rawoutput', type=str, default=None,
                        help='Path to save raw samples as .npz (default: save to ./results/raw/some_name.npz where some_name is derived from the simulation parameters)')
    parser.add_argument('--target-sector-index', type=int, default=0,
                        help='Deterministic global sector index (default: 0)')
    parser.add_argument('--min-ue-azimuth-separation-deg', type=float, default=10.0,
                        help=('Minimum azimuth separation in degrees between UEs in the same sector. '
                              'Use 0 or a negative value to disable. Default: 10.'))
    parser.add_argument('--ue-elevation-mode', type=str, default='fixed_height',
                        choices=['None', 'fixed_height'],
                        help=("Elevation constraint mode. 'None' keeps the original random drop. "
                              "'fixed_height' keeps all UEs at fixed_ue_height and sets the "
                              "common elevation/down-tilt angle through the 2D BS-UE distance."))
    parser.add_argument('--ue-elevation-angle-deg', type=float, default=10,
                        help=("Common down-tilt/elevation angle in degrees from the horizontal. "
                              "Required when --ue-elevation-mode fixed_height."))
    parser.add_argument('--fixed-ue-height', type=float, default=1.5,
                        help='Fixed UE height in meters for --ue-elevation-mode fixed_height.')
    parser.add_argument('--precoder', type=str, default='slnr', choices=['rzf', 'slnr'],
                        help='Precoder type to use (default: rzf)')
    parser.add_argument('--diagnostics', action='store_true', default=False,
                        help='Print per-slot diagnostics for effective channel and SINR')
    args = parser.parse_args()

    if args.ue_elevation_mode == 'fixed_height' and args.ue_elevation_angle_deg is None:
        raise ValueError("--ue-elevation-angle-deg is required when --ue-elevation-mode fixed_height")

    # MU-MIMO setup requested by user
    num_ut_per_sector = args.num_ut_per_sector
    num_streams_per_ut = args.num_streams_per_ut
    scenario = args.scenario
    direction = 'downlink'
    carrier_frequency = 3.5e9
    bs_max_power_dbm = 56.0
    ut_max_power_dbm = 26.0

    all_stream_sum_samples = []
    all_logdet_samples = []
    all_raw_snr_db_samples = []
    all_raw_sinr_db_samples = []
    all_max_ue_chan_corr_samples = []
    all_min_ue_azimuth_distance_samples = []

    # Initial post-drop UE angles exposed by SystemLevelSimulator.
    # Each per-drop tensor has shape [batch_size, num_bs, num_ut_per_sector].
    all_initial_ue_theta_deg = []
    all_initial_ue_phi_deg = []

    bs_array = PanelArray_38922(num_rows_per_panel=args.num_bs_vertical_antennas,
                        num_cols_per_panel=args.num_bs_horizontal_antennas,
                        polarization='dual',
                        polarization_type='VH',
                        antenna_pattern=args.bs_pattern,
                        carrier_frequency=carrier_frequency)

    # Two UT antennas are needed to support two spatial streams/user.
    ut_array = PanelArray_38922(num_rows_per_panel=args.num_ut_vertical_antennas,
                          num_cols_per_panel=args.num_ut_horizontal_antennas,
                          polarization='single',
                          polarization_type='V',
                          antenna_pattern='omni',
                          carrier_frequency=carrier_frequency)

    for drop_idx in range(args.num_drops):
        sionna.phy.config.seed = args.seed + drop_idx
        sionna.phy.config.precision = 'single'

        sls = build_simulator(
            num_ut_per_sector=num_ut_per_sector,
            num_streams_per_ut=num_streams_per_ut,
            num_subcarriers=args.num_subcarriers,
            num_ofdm_sym=args.num_ofdm_sym,
            carrier_frequency=carrier_frequency,
            scenario=scenario,
            direction=direction,
            num_rings=args.num_rings,
            bs_array=bs_array,
            ut_array=ut_array,
            bs_max_power_dbm=bs_max_power_dbm,
            ut_max_power_dbm=ut_max_power_dbm,
            min_ue_azimuth_separation_deg=args.min_ue_azimuth_separation_deg,
            ue_elevation_mode=args.ue_elevation_mode,
            ue_elevation_angle_deg=args.ue_elevation_angle_deg,
            fixed_ue_height=args.fixed_ue_height)

        # Store the initial post-drop UE angles. These are computed in
        # SystemLevelSimulator._setup_topology() immediately after the drop
        # and before any mobility updates.
        all_initial_ue_theta_deg.append(sls.ue_theta_deg.detach().cpu().numpy())
        all_initial_ue_phi_deg.append(sls.ue_phi_deg.detach().cpu().numpy())

        min_ue_azimuth_distance_samples = compute_min_ue_azimuth_distance_deg(
            sls=sls,
            target_sector_index=args.target_sector_index,
            num_ut_per_sector=num_ut_per_sector)
        all_min_ue_azimuth_distance_samples.append(min_ue_azimuth_distance_samples)

        if drop_idx == 0:
            save_grid_plot_for_first_drop(sls, './results/mu_mimo_grid.png')

        stream_sum_samples, logdet_samples, raw_snr_db_samples, raw_sinr_db_samples, max_ue_chan_corr_samples = compute_drop_log_capacity_samples(
            sls=sls,
            num_ut_per_sector=num_ut_per_sector,
            num_streams_per_ut=num_streams_per_ut,
            num_slots=args.num_slots,
            target_sector_index=args.target_sector_index,
            precoder=args.precoder,
            diagnostics=args.diagnostics,
            raw_interference_mode=args.raw_interference_mode)
        all_stream_sum_samples.append(stream_sum_samples)
        all_logdet_samples.append(logdet_samples)
        all_raw_snr_db_samples.append(raw_snr_db_samples)
        all_raw_sinr_db_samples.append(raw_sinr_db_samples)
        all_max_ue_chan_corr_samples.append(max_ue_chan_corr_samples)

        if (drop_idx + 1) % 10 == 0:
            print(f'Processed {drop_idx + 1}/{args.num_drops} drops')

    all_stream_sum_samples = np.concatenate(all_stream_sum_samples)
    all_logdet_samples = np.concatenate(all_logdet_samples)
    all_raw_snr_db_samples = np.concatenate(all_raw_snr_db_samples)
    all_raw_sinr_db_samples = np.concatenate(all_raw_sinr_db_samples)
    all_max_ue_chan_corr_samples = np.concatenate(all_max_ue_chan_corr_samples)
    all_min_ue_azimuth_distance_samples = np.concatenate(all_min_ue_azimuth_distance_samples)
    all_initial_ue_theta_deg = np.concatenate(all_initial_ue_theta_deg, axis=0)
    all_initial_ue_phi_deg = np.concatenate(all_initial_ue_phi_deg, axis=0)

    x_stream, y_stream = get_cdf(all_stream_sum_samples)
    x_logdet, y_logdet = get_cdf(all_logdet_samples)
    x_raw_snr, y_raw_snr = get_cdf(all_raw_snr_db_samples)
    x_raw_sinr, y_raw_sinr = get_cdf(all_raw_sinr_db_samples)
    x_ue_corr, y_ue_corr = get_cdf(all_max_ue_chan_corr_samples)
    x_ue_min_az_dist, y_ue_min_az_dist = get_cdf(all_min_ue_azimuth_distance_samples)

    plt.figure(figsize=(6, 4))
    plt.plot(x_stream, y_stream, linewidth=2, label='Sector sum throughput: Σ_{u,s} log2(1+SINR_{u,s})')
    plt.plot(x_logdet, y_logdet, linewidth=2, linestyle='--',
             label='Sector sum throughput: Σ_u log2 det(I + R_u^-1 S_u)')
    plt.xlabel('Sector throughput [bits/s/Hz per RE]')
    plt.ylabel('CDF')
    plt.legend()
    plt.title(f'MU-MIMO sector sum throughput: CDF over {args.num_drops} drops x {args.num_slots} slots')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f'Saved CDF plot to: {args.out}')

    plt.figure(figsize=(6, 4))
    plt.plot(x_raw_snr, y_raw_snr, linewidth=2,
             label=r'Raw SNR: $(P_t/N_t)\|H_{k,b}\|_F^2 / N_0$')
    plt.xlabel('Raw no-precoding SNR [dB]')
    plt.ylabel('CDF')
    plt.legend()
    plt.title(f'Raw no-precoding SNR: CDF over {args.num_drops} drops x {args.num_slots} slots')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.raw_snr_out, dpi=300)
    print(f'Saved raw SNR CDF plot to: {args.raw_snr_out}')

    plt.figure(figsize=(6, 4))
    plt.plot(x_raw_sinr, y_raw_sinr, linewidth=2, linestyle='-',
             label=r'Raw SINR: $(P_t/N_t)\|H_{k,b}\|_F^2 / (I_k+N_0)$')
    plt.xlabel('SINR [dB]')
    plt.ylabel('CDF')
    # plt.legend()
    plt.title(f'SINR CDF (without precoding)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.raw_sinr_out, dpi=300)
    print(f'Saved raw SINR CDF plot to: {args.raw_sinr_out}')

    plt.figure(figsize=(6, 4))
    plt.plot(x_ue_corr, y_ue_corr, linewidth=2)
    plt.xlabel('Maximum pairwise UE channel correlation')
    plt.ylabel('CDF')
    plt.title('CDF of maximum pairwise UE channel correlation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.ue_corr_out, dpi=300)
    print(f'Saved UE channel correlation CDF plot to: {args.ue_corr_out}')

    plt.figure(figsize=(6, 4))
    plt.plot(x_ue_min_az_dist, y_ue_min_az_dist, linewidth=2)
    plt.xlabel('Minimum pairwise UE azimuth distance [deg]')
    plt.ylabel('CDF')
    plt.title('CDF of minimum pairwise UE azimuth distance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.ue_min_azimuth_distance_out, dpi=300)
    print(f'Saved UE minimum azimuth distance CDF plot to: {args.ue_min_azimuth_distance_out}')

    # Save raw samples
    if args.rawoutput is None:
        # Create a filename that encodes the simulation parameters for traceability.
        folder = './results/raw'
        folder += '/mu_mimo'
        folder += f'/{direction}'
        folder += f'/{scenario}'
        folder += f'/rings{args.num_rings}'
        os.makedirs(folder, exist_ok=True)
        args.rawoutput = folder + '/' +  f'ut{num_ut_per_sector}_streams{num_streams_per_ut}_bspattern_{args.bs_pattern}_bsant{args.num_bs_vertical_antennas}x{args.num_bs_horizontal_antennas}_utant{args.num_ut_vertical_antennas}x{args.num_ut_horizontal_antennas}_precoder_{args.precoder}_ofdm{args.num_ofdm_sym}_subc{args.num_subcarriers}_seed{args.seed}.npz'
        
    os.makedirs(os.path.dirname(args.rawoutput), exist_ok=True)
    np.savez(args.rawoutput,
             stream_sum_samples=all_stream_sum_samples,
             logdet_samples=all_logdet_samples,
             raw_snr_db_samples=all_raw_snr_db_samples,
             raw_sinr_db_samples=all_raw_sinr_db_samples,
             max_ue_chan_corr_samples=all_max_ue_chan_corr_samples,
             min_ue_azimuth_distance_samples=all_min_ue_azimuth_distance_samples,
             initial_ue_theta_deg=all_initial_ue_theta_deg,
             initial_ue_phi_deg=all_initial_ue_phi_deg)
    print(f'Saved raw samples to: {args.rawoutput}')

if __name__ == '__main__':
    main()