
from .utils import *
import numpy as np
import torch
from sionna.sys import PHYAbstraction, \
    OuterLoopLinkAdaptation, \
    get_pathloss, open_loop_uplink_power_control, downlink_fair_power_control, \
    PFSchedulerSUMIMO

# Use the repo-local topology helper so the constrained UE dropping logic is
# version-controlled with this experiment instead of requiring edits to the
# installed Sionna package.
try:
    from .topology import gen_hexgrid_topology, get_num_hex_in_grid
except ImportError:
    from topology import gen_hexgrid_topology, get_num_hex_in_grid
    
from sionna.sys.utils import spread_across_subcarriers
from sionna.phy.utils import dbm_to_watt
from sionna.phy.constants import BOLTZMANN_CONSTANT

from sionna.phy.channel.tr38901 import UMi, UMa, RMa
from sionna.phy import Block
import matplotlib.pyplot as plt

CENTER_CELL_NUM_SECTORS = 3

class CenterCellGrid(torch.nn.Module):
    """Grid view for center-cell-only mode with explicit center-cell plotting."""

    def __init__(self, full_grid, bs_loc, ut_loc):
        super().__init__()
        self.full_grid = full_grid
        self.bs_loc = bs_loc
        self.ut_loc = ut_loc

    def show(self):
        bs_loc_np = self.bs_loc.cpu().numpy() if hasattr(self.bs_loc, 'cpu') else self.bs_loc
        ut_loc_np = self.ut_loc.cpu().numpy() if hasattr(self.ut_loc, 'cpu') else self.ut_loc

        bs_x = bs_loc_np[0, :, 0]
        bs_y = bs_loc_np[0, :, 1]
        ut_x = ut_loc_np[0, :, 0]
        ut_y = ut_loc_np[0, :, 1]

        x_center = float(bs_x.mean())
        y_center = float(bs_y.mean())

        hex_radius = float(self.full_grid._isd / np.sqrt(3.0))
        vertex_angles = np.linspace(0.0, 2.0*np.pi, 7) + np.pi / 6.0
        hex_x = x_center + hex_radius*np.cos(vertex_angles)
        hex_y = y_center + hex_radius*np.sin(vertex_angles)

        fig, ax = plt.subplots(1, 1)
        ax.plot(hex_x, hex_y, color='b')
        ax.scatter(bs_x, bs_y, color='b', label='base cell')
        # Keep UE legend off here because some callers (tests) add their own UE
        # overlay/label after calling grid.show(), which would otherwise create
        # duplicate "user position" legend entries.
        ax.scatter(ut_x, ut_y, color='k', marker='x', label='_nolegend_')

        margin = 0.1 * hex_radius
        ax.set_xlim(x_center - hex_radius - margin, x_center + hex_radius + margin)
        ax.set_ylim(y_center - hex_radius - margin, y_center + hex_radius + margin)
        ax.set_aspect('equal', adjustable='box')
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())

        return fig
    
class SystemLevelSimulator(Block):
    def __init__(self,
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
                 pf_beta=0.98,
                 max_bs_ut_dist=None,
                 min_bs_ut_dist=None,
                 temperature=294,
                 o2i_model='low',
                 average_street_width=20.0,
                 average_building_height=5.0,
                 min_ue_azimuth_separation_deg=10.0,
                 ue_elevation_mode='None',
                 ue_elevation_angle_deg=None,
                 fixed_ue_height=1.5,
                 precision=None):
        super().__init__(precision=precision)

        assert scenario in ['umi', 'uma', 'rma']
        assert direction in ['uplink', 'downlink']
        self.scenario = scenario
        self.batch_size = int(batch_size)
        self.resource_grid = resource_grid
        self.num_ut_per_sector = int(num_ut_per_sector)
        self.direction = direction
        self.bs_max_power_dbm = bs_max_power_dbm
        self.ut_max_power_dbm = ut_max_power_dbm
        self.coherence_time = int(coherence_time)
        self.min_ue_azimuth_separation_deg = min_ue_azimuth_separation_deg
        self.ue_elevation_mode = ue_elevation_mode
        self.ue_elevation_angle_deg = ue_elevation_angle_deg
        self.fixed_ue_height = fixed_ue_height


        # Sionna's built-in HexGrid requires num_rings > 0. We use
        # num_rings == 0 as a custom, efficient "center cell only" mode:
        # generate a valid num_rings=1 topology internally, then slice it to
        # keep only the 3 sectors and UEs of the center cell before the
        # topology is applied to the channel model and before ChannelMatrix is created.
        self.requested_num_rings = int(num_rings)
        self.center_cell_only = (self.requested_num_rings == 0)
        self.topology_num_rings = 1 if self.center_cell_only else self.requested_num_rings

        if self.center_cell_only:
            self.num_cells = 1
            self.num_bs = CENTER_CELL_NUM_SECTORS
        else:
            self.num_cells = get_num_hex_in_grid(self.topology_num_rings)
            self.num_bs = self.num_cells * CENTER_CELL_NUM_SECTORS

        self.num_ut = self.num_bs * self.num_ut_per_sector
        self.num_ut_ant = ut_array.num_ant
        self.num_bs_ant = bs_array.num_ant
        if bs_array.polarization == 'dual':
            self.num_bs_ant *= 2
        if self.direction == 'uplink':
            self.num_tx, self.num_rx = self.num_ut, self.num_bs
            self.num_tx_ant, self.num_rx_ant = self.num_ut_ant, self.num_bs_ant
            self.num_tx_per_sector = self.num_ut_per_sector
        else:
            self.num_tx, self.num_rx = self.num_bs, self.num_ut
            self.num_tx_ant, self.num_rx_ant = self.num_bs_ant, self.num_ut_ant
            self.num_tx_per_sector = 1

        self.num_streams_per_ut = resource_grid.num_streams_per_tx

        # Pre-compute for torch.compile compatibility
        self._mcs_category = 1 if direction == 'downlink' else 0

        self.stream_management = get_stream_management(direction,
                                                       self.num_rx,
                                                       self.num_tx,
                                                       self.num_streams_per_ut,
                                                       num_ut_per_sector)
        self.no = torch.tensor(BOLTZMANN_CONSTANT * temperature *
                               resource_grid.subcarrier_spacing,
                               dtype=self.dtype, device=self.device)

        self.slot_duration = resource_grid.ofdm_symbol_duration * \
            resource_grid.num_ofdm_symbols

        self._setup_channel_model(
            scenario, carrier_frequency, o2i_model, ut_array, bs_array,
            average_street_width, average_building_height)

        self._setup_topology(num_rings, min_bs_ut_dist, max_bs_ut_dist)

        self.phy_abs = PHYAbstraction(precision=self.precision, device=self.device)

        self.olla = OuterLoopLinkAdaptation(
            self.phy_abs,
            self.num_ut_per_sector,
            batch_size=[self.batch_size, self.num_bs],
            precision=self.precision,
            device=self.device)

        self.scheduler = PFSchedulerSUMIMO(
            self.num_ut_per_sector,
            resource_grid.fft_size,
            resource_grid.num_ofdm_symbols,
            batch_size=[self.batch_size, self.num_bs],
            num_streams_per_ut=self.num_streams_per_ut,
            beta=pf_beta,
            precision=self.precision,
            device=self.device)

        # Create ChannelMatrix in __init__ for torch.compile compatibility
        self.channel_matrix = ChannelMatrix(
            self.resource_grid,
            self.batch_size,
            self.num_rx,
            self.num_tx,
            self.coherence_time,
            precision=self.precision,
            device=self.device)

        # Pre-allocate rx_tx_association tensor
        self._rx_tx_association = torch.tensor(
            self.stream_management.rx_tx_association,
            device=self.device)

    def _setup_channel_model(self, scenario, carrier_frequency, o2i_model,
                             ut_array, bs_array, average_street_width,
                             average_building_height):
        common_params = {
            'carrier_frequency': carrier_frequency,
            'ut_array': ut_array,
            'bs_array': bs_array,
            'direction': self.direction,
            'enable_pathloss': True,
            'enable_shadow_fading': True,
            'precision': self.precision
        }

        if scenario == 'umi':
            self.channel_model = UMi(o2i_model=o2i_model, **common_params)
        elif scenario == 'uma':
            self.channel_model = UMa(o2i_model=o2i_model, **common_params)
        elif scenario == 'rma':
            self.channel_model = RMa(
                average_street_width=average_street_width,
                average_building_height=average_building_height,
                **common_params)

    def _slice_center_cell_topology(self):
        """Keep only the 3 sectors and UEs of the center cell.

        gen_hexgrid_topology(num_rings=1) returns the center cell first,
        followed by the outer-ring cells. Since each cell has 3 sectors, the
        center-cell BS/sector indices are 0, 1, and 2. The UTs are grouped by
        sector, so the center-cell UT indices are the first
        3*num_ut_per_sector entries.
        """
        bs_slice = slice(0, CENTER_CELL_NUM_SECTORS)
        ut_slice = slice(0, CENTER_CELL_NUM_SECTORS * self.num_ut_per_sector)

        self.bs_loc = self.bs_loc[:, bs_slice, :]
        self.bs_orientations = self.bs_orientations[:, bs_slice, :]

        self.ut_loc = self.ut_loc[:, ut_slice, :]
        self.ut_orientations = self.ut_orientations[:, ut_slice, :]
        self.ut_velocities = self.ut_velocities[:, ut_slice, :]
        self.in_state = self.in_state[:, ut_slice]

        # bs_virtual_loc is normally indexed as [batch, num_bs, num_ut, 3].
        if torch.is_tensor(self.bs_virtual_loc):
            self.bs_virtual_loc = self.bs_virtual_loc[:, bs_slice, ut_slice, :]

        # Depending on Sionna version / los argument, self.los can be a bool or
        # a tensor. Slice only when it is tensor-valued.
        if torch.is_tensor(self.los):
            if self.los.ndim >= 3:
                # Typical shape: [batch, num_bs, num_ut, ...]
                self.los = self.los[:, bs_slice, ut_slice, ...]
            elif self.los.ndim == 2:
                # Fallback for a UT-only shape: [batch, num_ut]
                self.los = self.los[:, ut_slice]

    def _setup_topology(self, num_rings, min_bs_ut_dist, max_bs_ut_dist):
        self.ut_loc, self.bs_loc, self.ut_orientations, self.bs_orientations, \
            self.ut_velocities, self.in_state, self.los, self.bs_virtual_loc, self.grid = \
            gen_hexgrid_topology(
                batch_size=self.batch_size,
                num_rings=self.topology_num_rings,
                num_ut_per_sector=self.num_ut_per_sector,
                min_bs_ut_dist=min_bs_ut_dist,
                max_bs_ut_dist=max_bs_ut_dist,
                scenario=self.scenario,
                los=True,
                return_grid=True,
                min_ue_azimuth_separation_deg=self.min_ue_azimuth_separation_deg,
                ue_elevation_mode=self.ue_elevation_mode,
                ue_elevation_angle_deg=self.ue_elevation_angle_deg,
                fixed_ue_height=self.fixed_ue_height,
                precision=self.precision)

        if self.center_cell_only:
            self._slice_center_cell_topology()
            self.grid = CenterCellGrid(self.grid, self.bs_loc, self.ut_loc)

        # Expose the initial post-drop UE angles for each serving sector.
        # Shapes: [batch_size, num_bs, num_ut_per_sector].
        self._compute_ue_theta_phi()

        self.channel_model.set_topology(
            self.ut_loc, self.bs_loc, self.ut_orientations,
            self.bs_orientations, self.ut_velocities,
            self.in_state, self.los, self.bs_virtual_loc)

    def _compute_ue_theta_phi(self):
        """Expose the serving-BS UE angles after the topology drop.

        The exposed tensors are:

        * ``self.ue_theta_deg``: 3GPP-style zenith angle, in degrees.
          A UE exactly at the BS horizon has theta = 90 deg; a ground UE below
          the BS horizon has theta > 90 deg.
        * ``self.ue_phi_deg``: azimuth angle, in degrees, relative to the
          serving sector boresight/yaw. This is the angle you can later use as
          the 38.922 observation azimuth phi for that UE, and as phi_escan if
          you steer the beam directly toward the UE.

        Both tensors have shape [batch_size, num_bs, num_ut_per_sector].
        The UT ordering in ``self.ut_loc`` is sector-major, so reshaping by
        [batch_size, num_bs, num_ut_per_sector, 3] aligns each UE with its
        serving sector/BS.
        """
        ut_loc_by_sector = torch.reshape(
            self.ut_loc,
            [self.batch_size, self.num_bs, self.num_ut_per_sector, 3])

        rel = ut_loc_by_sector - self.bs_loc[:, :, None, :]
        dx = rel[..., 0]
        dy = rel[..., 1]
        dz = rel[..., 2]

        d3d = torch.sqrt(dx**2 + dy**2 + dz**2)

        # 3GPP-style zenith angle theta: angle from +z axis.
        cos_theta = dz / torch.clamp(d3d, min=1e-30)
        theta_rad = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))

        # Global azimuth from BS to UE, then convert to sector-local azimuth by
        # subtracting the BS yaw. Wrap to [-pi, pi].
        phi_global_rad = torch.atan2(dy, dx)
        bs_yaw_rad = self.bs_orientations[:, :, 0]
        phi_rad = phi_global_rad - bs_yaw_rad[:, :, None]
        phi_rad = torch.atan2(torch.sin(phi_rad), torch.cos(phi_rad))

        self.ue_theta_rad = theta_rad
        self.ue_phi_rad = phi_rad
        self.ue_theta_deg = theta_rad * (180.0 / np.pi)
        self.ue_phi_deg = phi_rad * (180.0 / np.pi)

        # Minimal convenience dictionary. Keep only theta and phi, as requested.
        self.ue_angles = {
            'theta_deg': self.ue_theta_deg,
            'phi_deg': self.ue_phi_deg,
        }
        return self.ue_angles

    def refresh_ue_theta_phi(self):
        """Recompute UE theta/phi from the current UE locations.

        This is useful only if you want angles after mobility updates. The
        initial angles are already computed immediately after the drop.
        """
        return self._compute_ue_theta_phi()

    def _reset_olla(self, bler_target, olla_delta_up):
        """Reset OLLA parameters - must be called OUTSIDE compiled code."""
        self.olla.reset()
        self.olla.bler_target = bler_target
        self.olla.delta_up = olla_delta_up

    def _init_feedback_tensors(self):
        """Initialize feedback tensors - can be called inside compiled code."""
        self.channel_matrix.reset()

        last_harq_feedback = -torch.ones(
            [self.batch_size, self.num_bs, self.num_ut_per_sector],
            dtype=torch.int32, device=self.device)

        sinr_eff_feedback = torch.ones(
            [self.batch_size, self.num_bs, self.num_ut_per_sector],
            dtype=self.dtype, device=self.device)

        num_decoded_bits = torch.zeros(
            [self.batch_size, self.num_bs, self.num_ut_per_sector],
            dtype=torch.int32, device=self.device)
        return last_harq_feedback, sinr_eff_feedback, num_decoded_bits

    def _group_by_sector(self, tensor):
        tensor = torch.reshape(tensor, [self.batch_size,
                                        self.num_bs,
                                        self.num_ut_per_sector,
                                        self.resource_grid.num_ofdm_symbols])
        return torch.permute(tensor, [0, 1, 3, 2])

    def call(self,
             num_slots,
             alpha_ul,
             p0_dbm_ul,
             bler_target,
             olla_delta_up,
             mcs_table_index=1,
             fairness_dl=0,
             guaranteed_power_ratio_dl=0.5):
        """Main entry point - resets OLLA then runs compiled simulation."""
        # Reset OLLA parameters OUTSIDE compiled code to avoid graph breaks
        self._reset_olla(bler_target, olla_delta_up)
        # Run compiled simulation
        return self._run_simulation(num_slots, alpha_ul, p0_dbm_ul,
                                    mcs_table_index, fairness_dl,
                                    guaranteed_power_ratio_dl)

    @torch.compile
    def _run_simulation(self,
                        num_slots,
                        alpha_ul,
                        p0_dbm_ul,
                        mcs_table_index=1,
                        fairness_dl=0,
                        guaranteed_power_ratio_dl=0.5):
        """Compiled simulation loop."""
        # Initialize result history
        hist = init_result_history(self.batch_size,
                                   num_slots,
                                   self.num_bs,
                                   self.num_ut_per_sector,
                                   device=self.device)

        # Initialize feedback tensors
        harq_feedback, sinr_eff_feedback, num_decoded_bits = \
            self._init_feedback_tensors()

        # Generate initial channel
        h_freq = self.channel_matrix(self.channel_model)

        # Simulation loop
        for slot in range(num_slots):
            # Channel update
            h_freq = self.channel_matrix.update(self.channel_model, h_freq, slot)
            h_freq_fading = self.channel_matrix.apply_fading(h_freq)

            # Scheduler
            rate_achievable_est = estimate_achievable_rate(
                self.olla.sinr_eff_db_last,
                self.resource_grid.num_ofdm_symbols,
                self.resource_grid.fft_size)

            is_scheduled = self.scheduler(num_decoded_bits, rate_achievable_est)

            num_allocated_sc = torch.minimum(
                torch.sum(is_scheduled.int(), dim=-1),
                torch.ones(1, dtype=torch.int32, device=self.device))
            num_allocated_sc = torch.sum(num_allocated_sc, dim=-2)
            num_allocated_re = torch.sum(is_scheduled.int(), dim=[-1, -3, -4])

            # Power control
            pathloss_all_pairs, pathloss_serving_cell = get_pathloss(
                h_freq_fading, rx_tx_association=self._rx_tx_association)
            pathloss_serving_cell = self._group_by_sector(pathloss_serving_cell)

            if self.direction == 'uplink':
                tx_power_per_ut = open_loop_uplink_power_control(
                    pathloss_serving_cell,
                    num_allocated_sc,
                    alpha=alpha_ul,
                    p0_dbm=p0_dbm_ul,
                    ut_max_power_dbm=self.ut_max_power_dbm)
            else:
                one = torch.ones(1, dtype=pathloss_serving_cell.dtype, device=self.device)
                rx_power_tot = torch.sum(one / pathloss_all_pairs, dim=-2)
                rx_power_tot = self._group_by_sector(rx_power_tot)
                interference_dl = rx_power_tot - one / pathloss_serving_cell
                interference_dl = interference_dl * dbm_to_watt(self.bs_max_power_dbm)

                tx_power_per_ut, _ = downlink_fair_power_control(
                    pathloss_serving_cell,
                    interference_dl + self.no,
                    num_allocated_sc,
                    bs_max_power_dbm=self.bs_max_power_dbm,
                    guaranteed_power_ratio=guaranteed_power_ratio_dl,
                    fairness=fairness_dl,
                    precision=self.precision)

            tx_power = spread_across_subcarriers(
                tx_power_per_ut,
                is_scheduled,
                num_tx=self.num_tx_per_sector,
                precision=self.precision)

            # Per-stream SINR
            sinr = get_sinr(tx_power,
                            self.stream_management,
                            self.no,
                            self.direction,
                            h_freq_fading,
                            self.num_bs,
                            self.num_ut_per_sector,
                            self.num_streams_per_ut,
                            self.resource_grid)

            # Link adaptation
            mcs_index = self.olla(num_allocated_re,
                                  harq_feedback=harq_feedback,
                                  sinr_eff=sinr_eff_feedback)

            # PHY abstraction
            num_decoded_bits, harq_feedback, sinr_eff, _, _ = self.phy_abs(
                mcs_index,
                sinr=sinr,
                mcs_table_index=mcs_table_index,
                mcs_category=self._mcs_category)

            # SINR feedback
            sinr_eff_feedback = torch.where(
                num_allocated_re > 0,
                sinr_eff,
                torch.zeros(1, dtype=self.dtype, device=self.device))

            # Record results
            hist = record_results(hist,
                                  slot,
                                  sim_failed=False,
                                  pathloss_serving_cell=torch.sum(pathloss_serving_cell, dim=-2),
                                  num_allocated_re=num_allocated_re,
                                  tx_power_per_ut=torch.sum(tx_power_per_ut, dim=-2),
                                  num_decoded_bits=num_decoded_bits,
                                  mcs_index=mcs_index,
                                  harq_feedback=harq_feedback,
                                  olla_offset=self.olla.offset,
                                  sinr_eff=sinr_eff,
                                  pf_metric=self.scheduler.pf_metric)

            # User mobility
            self.ut_loc = self.ut_loc + self.ut_velocities * self.slot_duration
            self.channel_model.set_topology(
                self.ut_loc, self.bs_loc, self.ut_orientations,
                self.bs_orientations, self.ut_velocities,
                self.in_state, self.los, self.bs_virtual_loc)

        for key in hist:
            hist[key] = hist[key].stack()
        return hist