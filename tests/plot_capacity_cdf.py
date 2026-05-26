import os
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
	
from functions.utils import get_cdf

#%%

def load_samples(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f'NPZ file not found: {path}')
    data = np.load(path)
    # Expect keys: stream_sum_samples, logdet_samples
    assert path.endswith('.npz'), f'Input file must be a .npz file, got: {path}'
    if 'stream_sum_samples' not in data or 'logdet_samples' not in data:
        raise KeyError(f'NPZ file {path} must contain "stream_sum_samples" and "logdet_samples"')
    return data['stream_sum_samples'], data['logdet_samples']


def plot_cdfs(npz_path: str, ignore_nan=True, label_stream=None, label_logdet=None, plot_color='C0'):
	
    
    stream, logdet = load_samples(npz_path)
    if ignore_nan:
        stream = stream[~np.isnan(stream)]
        logdet = logdet[~np.isnan(logdet)]
    mean_stream = np.mean(stream)
    mean_logdet = np.mean(logdet)
    x_stream, y_stream = get_cdf(stream)
    x_logdet, y_logdet = get_cdf(logdet)
    if label_stream is not None:
        # mylabel = f'SU: Σ_s log2(1+SINR_s), μ={mean_su_stream:.2f}'
        plt.plot(x_stream, y_stream, linewidth=2, label=label_stream + f' μ={mean_stream:.2f} bps/Hz', color=plot_color)
    # mylabel = f'SU: log2 det(I+R^-1 S), μ={mean_su_logdet:.2f}'
    if label_logdet is not None:
        plt.plot(x_logdet, y_logdet, linewidth=2, linestyle='--', label=label_logdet + f' μ={mean_logdet:.2f} bps/Hz', color=plot_color)
        
#%%

show = True
save = False
out = "../results/plots/su_mu_mimo_capacity_cdf.png"


path_su = "../results/raw/su_mimo/downlink/umi/rings1/ut1_streams4_bspattern_38.922_bsant8x16_utant1x4_precoder_rzf_ofdm1_subc128_seed42.npz"
path_mu_2ut = "../results/raw/mu_mimo/downlink/umi/rings0/ut2_streams4_bspattern_38.922_bsant8x16_utant1x4_precoder_rzf_ofdm1_subc128_seed42.npz"
path_mu_4ut = "../results/raw/mu_mimo/downlink/umi/rings0/ut4_streams4_bspattern_38.922_bsant8x16_utant1x4_precoder_rzf_ofdm1_subc128_seed42.npz"
path_mu_6ut = "../results/raw/mu_mimo/downlink/umi/rings0/ut6_streams4_bspattern_38.922_bsant8x16_utant1x4_precoder_rzf_ofdm1_subc128_seed42.npz"
path_mu_24ut = "../results/raw/mu_mimo/downlink/umi/rings0/ut24_streams1_bspattern_38.922_bsant8x16_utant1x1_precoder_rzf_ofdm1_subc128_seed42.npz"

#%%

plt.figure(figsize=(10, 5))
plot_cdfs(path_su, label_logdet='SU MIMO, 1 UT, 4 Streams per UT', plot_color='C0')
plot_cdfs(path_mu_2ut, label_logdet='MU MIMO 2 UT, 4 Streams per UT', plot_color='C1')
plot_cdfs(path_mu_4ut, label_logdet='MU MIMO 4 UT, 4 Streams per UT', plot_color='C2')
plot_cdfs(path_mu_6ut, label_logdet='MU MIMO 6 UT, 4 Streams per UT', plot_color='C3')
plot_cdfs(path_mu_24ut, label_logdet='MU MIMO 24 UT, 1 Stream per UT', plot_color='C4')



plt.xlabel('Sector throughput [bits/s/Hz]')
plt.ylabel('CDF')
plt.title('Sector LogDet Capacity CDFs')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
if save:
    plt.savefig(out, dpi=300)
    print(f'Saved CDF plot to: {out}')
if show:
    plt.show()

#%%

plt.figure(figsize=(10, 7))
plot_cdfs(path_su, label_stream='SU MIMO Single UT', plot_color='C0')
plot_cdfs(path_mu_2ut, label_stream='MU MIMO 2 UT, 4 Streams per UT', plot_color='C1')
plot_cdfs(path_mu_4ut, label_stream='MU MIMO 4 UT, 4 Streams per UT', plot_color='C2')
plot_cdfs(path_mu_6ut, label_stream='MU MIMO 6 UT, 4 Streams per UT', plot_color='C3')
plot_cdfs(path_mu_24ut, label_stream='MU MIMO 24 UT, 1 Stream per UT', plot_color='C4')

plt.xlabel('Sector throughput [bits/s/Hz]')
plt.ylabel('CDF')
plt.title('Sector Stream Sum Capacity CDFs')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
if save:
    plt.savefig(out, dpi=300)
    print(f'Saved CDF plot to: {out}')
if show:
    plt.show()