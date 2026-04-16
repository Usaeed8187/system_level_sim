
from .utils import *
from sionna.sys import PHYAbstraction, \
    OuterLoopLinkAdaptation, gen_hexgrid_topology, \
    get_pathloss, open_loop_uplink_power_control, downlink_fair_power_control, \
    get_num_hex_in_grid, PFSchedulerSUMIMO
    
from sionna.sys.utils import spread_across_subcarriers
from sionna.phy.utils import dbm_to_watt
from sionna.phy.constants import BOLTZMANN_CONSTANT

from sionna.phy.channel.tr38901 import UMi, UMa, RMa
from sionna.phy import Block



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
        num_cells = get_num_hex_in_grid(num_rings)
        self.num_bs = num_cells * 3
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

    def _setup_topology(self, num_rings, min_bs_ut_dist, max_bs_ut_dist):
        self.ut_loc, self.bs_loc, self.ut_orientations, self.bs_orientations, \
            self.ut_velocities, self.in_state, self.los, self.bs_virtual_loc, self.grid = \
            gen_hexgrid_topology(
                batch_size=self.batch_size,
                num_rings=num_rings,
                num_ut_per_sector=self.num_ut_per_sector,
                min_bs_ut_dist=min_bs_ut_dist,
                max_bs_ut_dist=max_bs_ut_dist,
                scenario=self.scenario,
                los=True,
                return_grid=True,
                precision=self.precision)

        self.channel_model.set_topology(
            self.ut_loc, self.bs_loc, self.ut_orientations,
            self.bs_orientations, self.ut_velocities,
            self.in_state, self.los, self.bs_virtual_loc)

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