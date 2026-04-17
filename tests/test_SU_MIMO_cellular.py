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
                          num_cols_per_panel=3,
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
                                                   target_tx: int) -> torch.Tensor:
    """Compute combiner-agnostic capacity using a log-det formula.

    h_eff_target_rx shape:
      [batch, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_sym, num_subcarriers]

    Uses: log2 det(I + R^{-1} S), where
      S = H_target F_target P_target F_target^H H_target^H
      R = sum_{interferers} H_i F_i P_i F_i^H H_i^H + no*I.
    Here, H_i F_i sqrt(P_i) is represented directly by h_eff_target_rx slices.
    """
    h = h_eff_target_rx.permute(0, 4, 5, 2, 3, 1)
    # [batch, ofdm, subc, num_tx, num_streams_per_tx, num_rx_ant]
    hh = torch.einsum('...m,...n->...mn', h, torch.conj(h))
    # [batch, ofdm, subc, num_tx, num_streams_per_tx, num_rx_ant, num_rx_ant]
    cov_per_stream = hh

    signal_cov = torch.sum(cov_per_stream[:, :, :, target_tx, :, :, :], dim=3)
    # Sum all streams from all TX then subtract desired TX streams.
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
                                      num_streams_per_ut: int,
                                      num_slots: int,
                                      target_sector_index: int = 0):
    
    if num_slots < 1:
        raise ValueError(f'num_slots must be >= 1, got {num_slots}')

    sls.channel_matrix.reset()

    h_freq = sls.channel_matrix(sls.channel_model)

    rg = sls.resource_grid
    total_tx_power_watt = dbm_to_watt(sls.bs_max_power_dbm)
    # Distribute BS power across streams and REs (all subbands scheduled).
    tx_power_per_stream = total_tx_power_watt / (num_streams_per_ut * rg.num_ofdm_symbols * rg.fft_size)

    # Force full-band scheduling: single user in each sector is active on all REs.
    tx_power = torch.full(
        [sls.batch_size,
         sls.num_bs,
         1,
         num_streams_per_ut,
         rg.num_ofdm_symbols,
         rg.fft_size],
        fill_value=tx_power_per_stream,
        dtype=sls.dtype,
        device=sls.device)

    zf_precoder = RZFPrecodedChannel(resource_grid=rg,
                                     stream_management=sls.stream_management)
    zf_alpha = torch.zeros(1, dtype=sls.dtype, device=sls.device)
    lmmse_posteq_sinr = LMMSEPostEqualizationSINR(resource_grid=rg,
                                                  stream_management=sls.stream_management)
    
    # tx_power: [batch_size, num_bs, num_tx_per_sector,
    #            num_streams_per_tx, num_ofdm_sym, num_subcarriers]
    # Flatten across sectors
    # [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers]
    s = tx_power.shape
    tx_power = torch.reshape(tx_power, [s[0], s[1]*s[2]] + list(s[3:]))
    
    # Deterministic arbitrary choice: select a fixed global sector index.
    target_bs = int(target_sector_index)
    if target_bs < 0 or target_bs >= sls.num_bs:
        raise ValueError(f'target_sector_index must be in [0, {sls.num_bs - 1}], got {target_bs}')
    
    slot_stream_sum_samples = []
    slot_logdet_samples = []
    for slot in range(num_slots):
        h_freq = sls.channel_matrix.update(sls.channel_model, h_freq, slot)
        h_freq_fading = sls.channel_matrix.apply_fading(h_freq)

        h_eff = zf_precoder(h_freq_fading, tx_power=tx_power, alpha=zf_alpha)
        if h_eff.ndim != 7:
            raise RuntimeError(f'Unexpected effective channel shape: {tuple(h_eff.shape)}')
        # h_eff dimensions:
        # [batch, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_sym, num_subcarriers]
        if h_eff.shape[1] != sls.num_ut or h_eff.shape[3] != sls.num_bs:
            raise RuntimeError(f'Unexpected effective channel shape: {tuple(h_eff.shape)}')

        # Select the first UE in the chosen sector (global RX index).
        target_rx = target_bs * sls.num_ut_per_sector
        target_tx = target_bs

        # Post-combining per-stream SINR via LMMSE combiner.
        # [batch, num_ofdm_sym, num_subcarriers, num_rx, num_streams_per_rx]
        sinr = lmmse_posteq_sinr(h_eff, no=sls.no, interference_whitening=True)
        # [batch, num_ofdm_sym, num_subcarriers, num_streams_per_ut]
        sinr_target = sinr[:, :, :, target_rx, :]
        # Sum_s log2(1 + SINR_s), per RE.
        stream_sum_rate = torch.sum(
            torch.log2(1.0 + torch.clamp(sinr_target, min=0.0)), dim=-1)
        slot_stream_sum_samples.append(stream_sum_rate.detach().cpu().numpy().ravel())

        # Alternative combiner-agnostic rate: log2 det(I + R^{-1} S).
        # Use target RX antenna-domain effective channels.
        h_eff_target_rx = h_eff[:, target_rx, :, :, :, :, :]
        logdet_rate = _compute_logdet_capacity_from_precoded_channel(
            h_eff_target_rx=h_eff_target_rx,
            no=sls.no,
            target_tx=target_tx)
        slot_logdet_samples.append(logdet_rate.detach().cpu().numpy().ravel())

        # Match slot-wise behavior used in e2e_example.py.
        sls.ut_loc = sls.ut_loc + sls.ut_velocities * sls.slot_duration
        sls.channel_model.set_topology(
            sls.ut_loc, sls.bs_loc, sls.ut_orientations,
            sls.bs_orientations, sls.ut_velocities,
            sls.in_state, sls.los, sls.bs_virtual_loc)

    return np.concatenate(slot_stream_sum_samples), np.concatenate(slot_logdet_samples)

def main():
    parser = argparse.ArgumentParser(description='SU-MIMO sector sum-throughput CDF experiment')
    parser.add_argument('--num-drops', type=int, default=10)
    parser.add_argument('--num-slots', type=int, default=10,
                        help='Number of slots simulated per drop (default: 10)')
    parser.add_argument('--num-rings', type=int, default=2)
    parser.add_argument('--num-ofdm-sym', type=int, default=1)
    parser.add_argument('--num-subcarriers', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='./results/su_mimo_log1p_sinr_cdf.png')
    parser.add_argument('--target-sector-index', type=int, default=0,
                        help='Deterministic global sector index (default: 0)')
    args = parser.parse_args()

    # SU-MIMO setup requested by user
    num_ut_per_sector = 1
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
            num_streams_per_ut=num_streams_per_ut,
            num_slots=args.num_slots,
            target_sector_index=args.target_sector_index)
        all_stream_sum_samples.append(stream_sum_samples)
        all_logdet_samples.append(logdet_samples)

        if (drop_idx + 1) % 10 == 0:
            print(f'Processed {drop_idx + 1}/{args.num_drops} drops')

    all_stream_sum_samples = np.concatenate(all_stream_sum_samples)
    all_logdet_samples = np.concatenate(all_logdet_samples)
    x_stream, y_stream = get_cdf(all_stream_sum_samples)
    x_logdet, y_logdet = get_cdf(all_logdet_samples)

    plt.figure(figsize=(6, 4))
    plt.plot(x_stream, y_stream, linewidth=2, label='Sector sum throughput: Σ_s log2(1+SINR_s)')
    plt.plot(x_logdet, y_logdet, linewidth=2, linestyle='--',
             label='Sector sum throughput: log2 det(I + R^-1 S)')
    plt.xlabel('Sector throughput [bits/s/Hz per RE]')
    plt.ylabel('CDF')
    plt.title(f'SU-MIMO ZF sector sum throughput: CDF over {args.num_drops} drops × {args.num_slots} slots')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f'Saved CDF plot to: {args.out}')


if __name__ == '__main__':
    main()
