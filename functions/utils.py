       
        
# Additional external libraries
import matplotlib.pyplot as plt
import numpy as np

# Sionna components

import torch

from sionna.phy.utils import db_to_lin, dbm_to_watt, insert_dims
from sionna.phy import config, dtypes, Block
from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, RZFPrecodedChannel, EyePrecodedChannel, \
    LMMSEPostEqualizationSINR



class ChannelMatrix(Block):
    def __init__(self,
                 resource_grid,
                 batch_size,
                 num_rx,
                 num_tx,
                 coherence_time,
                 precision=None,
                 device=None):
        super().__init__(precision=precision, device=device)
        self.resource_grid = resource_grid
        self.coherence_time = coherence_time
        self.batch_size = batch_size
        self.num_rx = num_rx
        self.num_tx = num_tx
        self._init_fading()

    def _init_fading(self):
        """Initialize/reset fading state tensors."""
        # Use register_buffer for torch.compile compatibility
        if not hasattr(self, '_rho_fading_initialized'):
            self.register_buffer(
                'rho_fading',
                torch.empty([self.batch_size, self.num_rx, self.num_tx],
                           dtype=self.dtype, device=self.device).uniform_(0.95, 0.99),
                persistent=False
            )
            self.register_buffer(
                'fading',
                torch.ones([self.batch_size, self.num_rx, self.num_tx],
                          dtype=self.dtype, device=self.device),
                persistent=False
            )
            self._rho_fading_initialized = True
        else:
            # Reset fading to ones
            self.fading.fill_(1.0)

    def reset(self):
        """Reset the fading state for a new simulation."""
        self._init_fading()

    def __call__(self, channel_model):
        """Override __call__ to avoid converting channel_model to tensor."""
        return self.call(channel_model)

    def call(self, channel_model):
        """Generate OFDM channel matrix."""
        ofdm_channel = GenerateOFDMChannel(channel_model, self.resource_grid)
        h_freq = ofdm_channel(self.batch_size)
        return h_freq

    def update(self, channel_model, h_freq, slot):
        """Update channel matrix every coherence_time slots."""
        h_freq_new = self.call(channel_model)
        change = ((slot % self.coherence_time) == 0)
        if change:
            h_freq = h_freq_new
        return h_freq

    def apply_fading(self, h_freq):
        """Apply fading, modeled as an autoregressive process, to channel matrix."""
        noise = torch.empty_like(self.fading).uniform_(-0.1, 0.1)
        new_fading = 1.0 - self.rho_fading + self.rho_fading * self.fading + noise
        new_fading = torch.clamp(new_fading, min=1e-6)  # Prevent exactly zero fading
        self.fading.copy_(new_fading)

        fading_expand = insert_dims(self.fading, 1, axis=2)
        fading_expand = insert_dims(fading_expand, 3, axis=4)
        h_freq_fading = torch.sqrt(fading_expand).to(self.cdtype) * h_freq
        return h_freq_fading
    
    

def get_stream_management(direction,
                          num_rx,
                          num_tx,
                          num_streams_per_ut,
                          num_ut_per_sector):
    """
    Instantiate a StreamManagement object.
    It determines which data streams are intended for each receiver
    """
    if direction == 'downlink':
        num_streams_per_tx = num_streams_per_ut * num_ut_per_sector
        # RX-TX association matrix
        rx_tx_association = np.zeros([num_rx, num_tx])
        idx = np.array([[i1, i2] for i2 in range(num_tx) for i1 in
                        np.arange(i2*num_ut_per_sector,
                                  (i2+1)*num_ut_per_sector)])
        rx_tx_association[idx[:, 0], idx[:, 1]] = 1

    else:
        num_streams_per_tx = num_streams_per_ut
        # RX-TX association matrix
        rx_tx_association = np.zeros([num_rx, num_tx])
        idx = np.array([[i1, i2] for i1 in range(num_rx) for i2 in
                        np.arange(i1*num_ut_per_sector,
                                  (i1+1)*num_ut_per_sector)])
        rx_tx_association[idx[:, 0], idx[:, 1]] = 1

    stream_management = StreamManagement(
        rx_tx_association, num_streams_per_tx)
    return stream_management



def get_sinr(tx_power,
             stream_management,
             no,
             direction,
             h_freq_fading,
             num_bs,
             num_ut_per_sector,
             num_streams_per_ut,
             resource_grid):
    """ Compute post-equalization SINR. It is assumed:
     - DL: Regularized zero-forcing precoding
     - UL: No precoding, only power allocation
    LMMSE equalizer is used in both DL and UL.
    """
    # tx_power: [batch_size, num_bs, num_tx_per_sector,
    #            num_streams_per_tx, num_ofdm_sym, num_subcarriers]
    # Flatten across sectors
    # [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers]
    s = tx_power.shape
    tx_power = torch.reshape(tx_power, [s[0], s[1]*s[2]] + list(s[3:]))

    # Compute SINR
    # [batch_size, num_ofdm_sym, num_subcarriers, num_ut,
    #  num_streams_per_ut]
    if direction == 'downlink':
        # Regularized zero-forcing precoding in the DL
        precoded_channel = RZFPrecodedChannel(resource_grid=resource_grid,
                                              stream_management=stream_management)
        h_eff = precoded_channel(h_freq_fading,
                                 tx_power=tx_power,
                                 alpha=no)  # Regularizer
    else:
        # No precoding in the UL: just power allocation
        precoded_channel = EyePrecodedChannel(resource_grid=resource_grid,
                                              stream_management=stream_management)
        h_eff = precoded_channel(h_freq_fading,
                                 tx_power=tx_power)

    # LMMSE equalizer
    lmmse_posteq_sinr = LMMSEPostEqualizationSINR(resource_grid=resource_grid,
                                                  stream_management=stream_management)
    # Post-equalization SINR
    # [batch_size, num_ofdm_symbols, num_subcarriers, num_rx, num_streams_per_rx]
    sinr = lmmse_posteq_sinr(h_eff, no=no, interference_whitening=True)

    # [batch_size, num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut]
    sinr = torch.reshape(
        sinr, list(sinr.shape[:-2]) + [num_bs*num_ut_per_sector, num_streams_per_ut])

    # Regroup by sector
    # [batch_size, num_ofdm_symbols, num_subcarriers, num_bs, num_ut_per_sector, num_streams_per_ut]
    sinr = torch.reshape(
        sinr, list(sinr.shape[:-2]) + [num_bs, num_ut_per_sector, num_streams_per_ut])

    # [batch_size, num_bs, num_ofdm_sym, num_subcarriers, num_ut_per_sector, num_streams_per_ut]
    sinr = torch.permute(sinr, [0, 3, 1, 2, 4, 5])
    return sinr



def estimate_achievable_rate(sinr_eff_db_last,
                             num_ofdm_sym,
                             num_subcarriers):
    """ Estimate achievable rate """
    # [batch_size, num_bs, num_ut_per_sector]
    rate_achievable_est = torch.log2(torch.tensor(1.0, dtype=sinr_eff_db_last.dtype) +
                                     db_to_lin(sinr_eff_db_last))

    # Broadcast to time/frequency grid
    # [batch_size, num_bs, num_ofdm_sym, num_subcarriers, num_ut_per_sector]
    rate_achievable_est = insert_dims(
        rate_achievable_est, 2, axis=-2)
    rate_achievable_est = torch.tile(rate_achievable_est,
                                     [1, 1, num_ofdm_sym, num_subcarriers, 1])
    return rate_achievable_est



class TensorList:
    """A simple TensorArray-like class for accumulating tensors.

    Pre-allocates storage to avoid dynamic list operations that
    cause recompilation in torch.compile.
    """
    def __init__(self, size, element_shape, dtype=torch.float32, device=None):
        self.size = size
        self.element_shape = element_shape
        self.dtype = dtype
        # Pre-allocate tensor storage to avoid dynamic list growth
        # which causes recompilation in torch.compile
        self.data = torch.zeros([size] + list(element_shape), dtype=dtype, device=device)

    def write(self, index, value):
        """Write a value at the given index using tensor indexing."""
        # Direct tensor indexing - no dynamic Python operations
        self.data[index] = value
        return self

    def stack(self):
        """Return the pre-allocated tensor."""
        return self.data


def init_result_history(batch_size,
                        num_slots,
                        num_bs,
                        num_ut_per_sector,
                        device=None):
    """ Initialize dictionary containing history of results """
    hist = {}
    for key in ['pathloss_serving_cell',
                'tx_power', 'olla_offset',
                'sinr_eff', 'pf_metric',
                'num_decoded_bits', 'mcs_index',
                'harq', 'num_allocated_re']:
        hist[key] = TensorList(
            size=num_slots,
            element_shape=[batch_size,
                           num_bs,
                           num_ut_per_sector],
            dtype=torch.float32,
            device=device)
    return hist


def record_results(hist,
                   slot,
                   sim_failed=False,
                   pathloss_serving_cell=None,
                   num_allocated_re=None,
                   tx_power_per_ut=None,
                   num_decoded_bits=None,
                   mcs_index=None,
                   harq_feedback=None,
                   olla_offset=None,
                   sinr_eff=None,
                   pf_metric=None,
                   shape=None):
    """ Record results of last slot """
    if not sim_failed:
        for key, value in zip(['pathloss_serving_cell', 'olla_offset', 'sinr_eff',
                               'num_allocated_re', 'tx_power', 'num_decoded_bits',
                               'mcs_index', 'harq'],
                              [pathloss_serving_cell, olla_offset, sinr_eff,
                               num_allocated_re, tx_power_per_ut, num_decoded_bits,
                               mcs_index, harq_feedback]):
            hist[key] = hist[key].write(slot, value.float())
        # Average PF metric across resources
        hist['pf_metric'] = hist['pf_metric'].write(
            slot, torch.mean(pf_metric, dim=[-2, -3]))
    else:
        nan_tensor = torch.full(shape, float('nan'), dtype=torch.float32)
        for key in hist:
            hist[key] = hist[key].write(slot, nan_tensor)
    return hist


def clean_hist(hist, batch=0):
    """ Extract batch, convert to Numpy, and mask metrics when user is not
    scheduled """
    # Extract batch and convert to Numpy
    for key in hist:
        if isinstance(hist[key], TensorList):
            # Stack and convert to numpy
            tensor = hist[key].stack()
            if hasattr(tensor, 'cpu'):
                tensor = tensor.cpu()
            # [num_slots, num_bs, num_ut_per_sector]
            hist[key] = tensor.numpy()[:, batch, :, :]
        elif isinstance(hist[key], torch.Tensor):
            if hasattr(hist[key], 'cpu'):
                hist[key] = hist[key].cpu()
            hist[key] = hist[key].numpy()[:, batch, :, :]

    # Mask metrics when user is not scheduled
    hist['mcs_index'] = np.where(
        hist['harq'] == -1, np.nan, hist['mcs_index'])
    hist['sinr_eff'] = np.where(
        hist['harq'] == -1, np.nan, hist['sinr_eff'])
    hist['tx_power'] = np.where(
        hist['harq'] == -1, np.nan, hist['tx_power'])
    hist['num_allocated_re'] = np.where(
        hist['harq'] == -1, 0, hist['num_allocated_re'])
    hist['harq'] = np.where(
        hist['harq'] == -1, np.nan, hist['harq'])
    return hist



def get_cdf(values):
    """
    Computes the Cumulative Distribution Function (CDF) of the input
    """
    values = np.array(values).flatten()
    n = len(values)
    sorted_val = np.sort(values)
    cumulative_prob = np.arange(1, n+1) / n
    return sorted_val, cumulative_prob


def pairplot(dict, keys, suptitle=None, figsize=2.5):
    fig, axs = plt.subplots(len(keys), len(keys),
                            figsize=[len(keys)*figsize]*2)
    for row, key_row in enumerate(keys):
        for col, key_col in enumerate(keys):
            ax = axs[row, col]
            ax.grid()
            if row == col:
                ax.hist(dict[key_row], bins=30,
                        color='skyblue', edgecolor='k',
                        linewidth=.5)
            elif col > row:
                fig.delaxes(ax)
            else:
                ax.scatter(dict[key_col], dict[key_row],
                           s=16, color='skyblue', alpha=0.9,
                           linewidths=.5, edgecolor='k')
            ax.set_ylabel(key_row)
            ax.set_xlabel(key_col)
    if suptitle is not None:
        fig.suptitle(suptitle, y=1, fontsize=17)
    fig.tight_layout()
    return fig, axs