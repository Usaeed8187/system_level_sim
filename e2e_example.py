
#%%
import os
os.makedirs('./results', exist_ok=True)
res_base = './results/'

import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = ""   # force CPU
    if gpu_num != "":
        print(f"\nUsing GPU {gpu_num}\n")
    else:
        print("\nUsing CPU\n")
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

# Import Sionna
try:
    import sionna.sys
except ImportError as e:
    import sys
    import os
    if 'google.colab' in sys.modules:
       # Install Sionna in Google Colab
       print("Installing Sionna and restarting the runtime. Please run the cell again.")
       os.system("pip install sionna")
       os.kill(os.getpid(), 5)
    else:
       raise e

import torch


        
        
# Additional external libraries
import matplotlib.pyplot as plt
import numpy as np

from sionna.phy.channel.tr38901 import PanelArray

from sionna.phy.ofdm import ResourceGrid

# Set random seed for reproducibility
sionna.phy.config.seed = 42

# Internal computational precision
sionna.phy.config.precision = 'single'  # 'single' or 'double'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from functions.utils import *
from functions.simulation import *

# Communication direction
direction = 'downlink'  # 'uplink' or 'downlink'

# 3GPP scenario parameters
scenario = 'umi'  # 'umi', 'uma' or 'rma'

# Number of rings of the hexagonal grid
# With num_rings=1, 7*3=21 base stations are placed
num_rings = 1

# N. users per sector
num_ut_per_sector = 10

# Max/min distance between base station and served users
max_bs_ut_dist = 80  # [m]
min_bs_ut_dist = 0  # [m]

# Carrier frequency
carrier_frequency = 3.5e9  # [Hz]

# Transmit power for base station and user terminals
bs_max_power_dbm = 56  # [dBm]
ut_max_power_dbm = 26  # [dBm]

# Channel is regenerated every coherence_time slots
coherence_time = 100  # [slots]

# MCS table index
# Ranges within [1;4] for downlink and [1;2] for uplink, as in TS 38.214
mcs_table_index = 1

# Number of examples
batch_size = 1

# Create the antenna arrays at the base stations
bs_array = PanelArray(num_rows_per_panel=2,
                      num_cols_per_panel=3,
                      polarization='dual',
                      polarization_type='VH',
                      antenna_pattern='38.901',
                      carrier_frequency=carrier_frequency)

# Create the antenna array at the user terminals
ut_array = PanelArray(num_rows_per_panel=1,
                      num_cols_per_panel=1,
                      polarization='single',
                      polarization_type='V',
                      antenna_pattern='omni',
                      carrier_frequency=carrier_frequency)


# n. OFDM symbols, i.e., time samples, in a slot
num_ofdm_sym = 1
# N. available subcarriers
num_subcarriers = 128
# Subcarrier spacing, i.e., bandwitdh width of each subcarrier
subcarrier_spacing = 15e3  # [Hz]

# Create the OFDM resource grid
resource_grid = ResourceGrid(num_ofdm_symbols=num_ofdm_sym,
                             fft_size=num_subcarriers,
                             subcarrier_spacing=subcarrier_spacing,
                             num_tx=num_ut_per_sector,
                             num_streams_per_tx=ut_array.num_ant)


# Initialize SYS object
sls = SystemLevelSimulator(
    batch_size,
    num_rings,
    num_ut_per_sector,
    carrier_frequency,
    resource_grid,
    scenario,
    direction,
    ut_array,
    bs_array,
    bs_max_power_dbm,
    ut_max_power_dbm,
    coherence_time,
    max_bs_ut_dist=max_bs_ut_dist,
    min_bs_ut_dist=min_bs_ut_dist,
    temperature=294,  # Environment temperature for noise power computation
    o2i_model='low',  # 'low' or 'high',
    average_street_width=20.,
    average_building_height=10.)

#%%
fig = sls.grid.show()
ax = fig.get_axes()
# Convert CUDA tensor to numpy for plotting
ut_loc_np = sls.ut_loc.cpu().numpy() if hasattr(sls.ut_loc, 'cpu') else sls.ut_loc
ax[0].plot(ut_loc_np[0, :, 0], ut_loc_np[0, :, 1],
           'xk', label='user position')
ax[0].legend()
plt.savefig(res_base + 'grid.png', dpi=300)
plt.show()

#%%

# N. slots to simulate
num_slots = 1000

# Link Adaptation
# Note: bler_target and olla_delta_up must be Python floats (not tensors)
# to avoid graph breaks in torch.compile
bler_target = 0.1  # Must be in [0, 1]
olla_delta_up = 0.2

# Uplink power control parameters
# Pathloss compensation factor
alpha_ul = 1.0  # Must be in [0, 1]
# Target received power at the base station
p0_dbm_ul = -80.0  # [dBm]

#%%

# System-level simulations
hist = sls(num_slots,
           alpha_ul,
           p0_dbm_ul,
           bler_target,
           olla_delta_up)

#%%

hist = clean_hist(hist)

# Average across slots and store in dictionary
results_avg = {
    'TBLER': (1 - np.nanmean(hist['harq'], axis=0)).flatten(),
    'MCS': np.nanmean(hist['mcs_index'], axis=0).flatten(),
    '# decoded bits / slot': np.nanmean(hist['num_decoded_bits'], axis=0).flatten(),
    'Effective SINR [dB]': 10*np.log10(np.nanmean(hist['sinr_eff'], axis=0).flatten()),
    'OLLA offset': np.nanmean(hist['olla_offset'], axis=0).flatten(),
    'TX power [dBm]': 10*np.log10(np.nanmean(hist['tx_power'], axis=0).flatten()) + 30,
    'Pathloss [dB]': 10*np.log10(np.nanmean(hist['pathloss_serving_cell'], axis=0).flatten()),
    '# allocated REs / slot': np.nanmean(hist['num_allocated_re'], axis=0).flatten(),
    'PF metric': np.nanmean(hist['pf_metric'], axis=0).flatten()
}
metrics = list(results_avg.keys())

#%%


fig, axs = plt.subplots(3, 3, figsize=(8, 6.5))
fig.suptitle('Per-user performance metrics', y=.99)

# Convert bler_target to scalar if it's a tensor
bler_target_val = bler_target.item() if hasattr(bler_target, 'item') else float(bler_target)

for ii in range(3):
    for jj in range(3):
        ax = axs[ii, jj]
        metric = metrics[3*ii+jj]
        ax.plot(*get_cdf(results_avg[metric]))
        if metric == 'TBLER':
            # Visualize BLER target
            ax.plot([bler_target_val]*2, [0, 1], '--k', label='target')
            ax.legend()
        if metric == 'TX power [dBm]':
            # Avoid plotting artifacts
            ax.set_xlim(ax.get_xlim()[0]-.5, ax.get_xlim()[1]+.5)
        ax.set_xlabel(metric)
        ax.grid()
        ax.set_ylabel('CDF')

fig.tight_layout()
# plt.show()
plt.savefig(res_base + 'tbler-tx.png', dpi=300)
plt.close()

#%%

fig, axs = pairplot(results_avg,
                    ['Effective SINR [dB]', 'MCS', '# decoded bits / slot'],
                    suptitle='MCS, SINR, and throughput')
# plt.show()
plt.savefig(res_base + 'sinr-mcs-thr.png', dpi=300)
plt.close()

#%%

fig, axs = pairplot(results_avg,
                    ['TBLER', 'MCS', 'OLLA offset'],
                    suptitle='TBLER, MCS, and OLLA offset')
for ii in range(3):
    axs[ii, 0].plot([bler_target]*2, axs[ii, 0].get_ylim(), '--k')
# plt.show()
plt.savefig(res_base + 'tbler-mcs-olla.png', dpi=300)
plt.close()

#%%

fig, axs = pairplot(results_avg,
                    ['# allocated REs / slot', 'PF metric', 'MCS'],
                    suptitle='PF metric, allocated resources, and MCS')
# plt.show()
plt.savefig(res_base + 're-pf-mcs.png', dpi=300)
plt.close()
# %%
