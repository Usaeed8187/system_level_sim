import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


os.makedirs('./results', exist_ok=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if os.getenv('CUDA_VISIBLE_DEVICES') is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

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
from sionna.phy.ofdm import ResourceGrid, RZFPrecodedChannel, LMMSEPostEqualizationSINR
from functions.slnr_precoder import StreamSLNRPrecodedChannel
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
                    bs_max_power_dbm: float,
                    ut_max_power_dbm: float):
    bs_array = PanelArray(num_rows_per_panel=2,
                          num_cols_per_panel=4,
                          polarization='dual',
                          polarization_type='VH',
                          antenna_pattern='38.901',
                          carrier_frequency=carrier_frequency)

    # Two UT antennas are needed to support two spatial streams/user.
    ut_array = PanelArray(num_rows_per_panel=1,
                          num_cols_per_panel=2,
                          polarization='single',
                          polarization_type='V',
                          antenna_pattern='omni',
                          carrier_frequency=carrier_frequency)

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
        average_building_height=10.0)

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

def compute_drop_log_capacity_samples(sls: SystemLevelSimulator,
                                      num_ut_per_sector: int,
                                      num_streams_per_ut: int,
                                      num_slots: int,
                                      target_sector_index: int = 0,
                                      precoder: str = 'rzf'):

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
        zf_precoder = RZFPrecodedChannel(resource_grid=rg,
                                         stream_management=sls.stream_management)
    elif precoder == 'slnr':
        zf_precoder = StreamSLNRPrecodedChannel(resource_grid=rg,
                                                stream_management=sls.stream_management)
    else:
        raise ValueError(f"Unsupported precoder '{precoder}'. Use 'rzf' or 'slnr'.")
    zf_alpha = torch.zeros(1, dtype=sls.dtype, device=sls.device)
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
    for slot in range(num_slots):
        h_freq = sls.channel_matrix.update(sls.channel_model, h_freq, slot)
        h_freq_fading = sls.channel_matrix.apply_fading(h_freq)

        h_eff = zf_precoder(h_freq_fading, tx_power=tx_power, alpha=zf_alpha)

        # [batch, num_ofdm_sym, num_subcarriers, num_rx, num_streams_per_rx]
        sinr = lmmse_posteq_sinr(h_eff, no=sls.no, interference_whitening=True)
        # [batch, num_ofdm_sym, num_subcarriers, num_bs, num_ut_per_sector, num_streams_per_ut]
        sinr = torch.reshape(
            sinr,
            list(sinr.shape[:-2]) + [sls.num_bs, num_ut_per_sector, num_streams_per_ut])
        # [batch, num_bs, num_ofdm_sym, num_subcarriers, num_ut_per_sector, num_streams_per_ut]
        sinr = torch.permute(sinr, [0, 3, 1, 2, 4, 5])
        target_sinr = sinr[:, target_bs, :, :, :, :]

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

    return np.concatenate(slot_stream_sum_samples), np.concatenate(slot_logdet_samples)


def main():
    parser = argparse.ArgumentParser(description='MU-MIMO sector sum-throughput CDF experiment')
    parser.add_argument('--num-drops', type=int, default=10)
    parser.add_argument('--num-slots', type=int, default=10,
                        help='Number of slots simulated per drop (default: 10)')
    parser.add_argument('--num-rings', type=int, default=2)
    parser.add_argument('--num-ofdm-sym', type=int, default=1)
    parser.add_argument('--num-subcarriers', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='./results/mu_mimo_log1p_sinr_cdf.png')
    parser.add_argument('--target-sector-index', type=int, default=0,
                        help='Deterministic global sector index (default: 0)')
    parser.add_argument('--precoder', type=str, default='slnr', choices=['rzf', 'slnr'],
                        help='Precoder type to use (default: rzf)')
    args = parser.parse_args()

    # MU-MIMO setup requested by user
    num_ut_per_sector = 2
    num_streams_per_ut = 2
    scenario = 'umi'
    direction = 'downlink'
    carrier_frequency = 3.5e9
    bs_max_power_dbm = 56.0
    ut_max_power_dbm = 26.0

    all_stream_sum_samples = []
    all_logdet_samples = []

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
            bs_max_power_dbm=bs_max_power_dbm,
            ut_max_power_dbm=ut_max_power_dbm)

        stream_sum_samples, logdet_samples = compute_drop_log_capacity_samples(
            sls=sls,
            num_ut_per_sector=num_ut_per_sector,
            num_streams_per_ut=num_streams_per_ut,
            num_slots=args.num_slots,
            target_sector_index=args.target_sector_index,
            precoder=args.precoder)
        all_stream_sum_samples.append(stream_sum_samples)
        all_logdet_samples.append(logdet_samples)

        if (drop_idx + 1) % 10 == 0:
            print(f'Processed {drop_idx + 1}/{args.num_drops} drops')

    all_stream_sum_samples = np.concatenate(all_stream_sum_samples)
    all_logdet_samples = np.concatenate(all_logdet_samples)
    x_stream, y_stream = get_cdf(all_stream_sum_samples)
    x_logdet, y_logdet = get_cdf(all_logdet_samples)

    plt.figure(figsize=(6, 4))
    plt.plot(x_stream, y_stream, linewidth=2, label='Sector sum throughput: Σ_{u,s} log2(1+SINR_{u,s})')
    plt.plot(x_logdet, y_logdet, linewidth=2, linestyle='--',
             label='Sector sum throughput: Σ_u log2 det(I + R_u^-1 S_u)')
    plt.xlabel('Sector throughput [bits/s/Hz per RE]')
    plt.ylabel('CDF')
    plt.legend()
    plt.title(f'MU-MIMO ZF sector sum throughput: CDF over {args.num_drops} drops × {args.num_slots} slots')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f'Saved CDF plot to: {args.out}')


if __name__ == '__main__':
    main()