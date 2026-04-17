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
from sionna.phy.ofdm import ResourceGrid, RZFPrecodedChannel
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
    
    slot_samples = []
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

        # Compute per-stream SINR directly from effective channel G (no post-equalization vector u):
        #   SINR_m = |G_{target_rx, :, target_tx, m}|^2
        #            / (sum_{(b,j)!=(target_tx,m)} |G_{target_rx, :, b, j}|^2 + no)
        # 1) Power per receive antenna is accumulated to form scalar stream powers.
        stream_power = torch.sum(torch.abs(h_eff) ** 2, dim=2)
        # [batch, num_tx, num_streams_per_tx, num_ofdm_sym, num_subcarriers]
        rx_power = stream_power[:, target_rx, :, :, :, :]
        # [batch, num_streams_per_tx, num_ofdm_sym, num_subcarriers]
        signal = rx_power[:, target_tx, :, :, :]
        # [batch, num_ofdm_sym, num_subcarriers]
        total_power = torch.sum(rx_power, dim=(1, 2))
        # [batch, num_streams_per_tx, num_ofdm_sym, num_subcarriers]
        interference = torch.clamp(total_power.unsqueeze(1) - signal, min=0.0)
        sinr_target = signal / (interference + sls.no)

        # Capacity-like metric in bits/s/Hz per stream per RE.
        log_capacity = torch.log2(1.0 + torch.clamp(sinr_target, min=0.0))
        slot_samples.append(log_capacity.detach().cpu().numpy().ravel())

        # Match slot-wise behavior used in e2e_example.py.
        sls.ut_loc = sls.ut_loc + sls.ut_velocities * sls.slot_duration
        sls.channel_model.set_topology(
            sls.ut_loc, sls.bs_loc, sls.ut_orientations,
            sls.bs_orientations, sls.ut_velocities,
            sls.in_state, sls.los, sls.bs_virtual_loc)

    return np.concatenate(slot_samples)

def main():
    parser = argparse.ArgumentParser(description='SU-MIMO cellular ZF SINR CDF experiment')
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

    all_samples = []

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

        samples = compute_drop_log_capacity_samples(
            sls=sls,
            num_streams_per_ut=num_streams_per_ut,
            num_slots=args.num_slots,
            target_sector_index=args.target_sector_index)
        all_samples.append(samples)

        if (drop_idx + 1) % 10 == 0:
            print(f'Processed {drop_idx + 1}/{args.num_drops} drops')

    all_samples = np.concatenate(all_samples)
    x, y = get_cdf(all_samples)

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, linewidth=2)
    plt.xlabel('log2(1 + SINR) [bits/s/Hz]')
    plt.ylabel('CDF')
    plt.title(f'SU-MIMO ZF: CDF over {args.num_drops} drops × {args.num_slots} slots')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f'Saved CDF plot to: {args.out}')


if __name__ == '__main__':
    main()
