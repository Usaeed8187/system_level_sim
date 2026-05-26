#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""3GPP TR 38.901 antenna modeling"""

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import torch

from sionna.phy import SPEED_OF_LIGHT, PI
from sionna.phy.object import Object
from scipy.constants import c as SPEED_OF_LIGHT_SI

from sionna.phy.channel.tr38901 import PanelArray


class AntennaElement(Object):
    """Antenna element following the :cite:p:`TR38901` specification

    :param pattern: Radiation pattern. One of ``"omni"`` or ``"38.901"``.
    :param slant_angle: Polarization slant angle [radian]
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., ``"cpu"``, ``"cuda:0"``).
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.channel.tr38901 import AntennaElement
        import torch

        # Create an antenna element with 38.901 radiation pattern
        ant = AntennaElement(pattern="38.901", slant_angle=0.0)

        # Compute field at zenith angle pi/2 and azimuth angle 0
        theta = torch.tensor([1.5708])
        phi = torch.tensor([0.0])
        f_theta, f_phi = ant.field(theta, phi)
    """

    def __init__(
        self,
        pattern: str,
        slant_angle: float = 0.0,
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(precision=precision, device=device)
        assert pattern in ["omni", "38.901", "38.922"], \
            "The radiation_pattern must be one of [\"omni\", \"38.901\", \"38.922\"]."

        self._pattern = pattern
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer("_slant_angle", torch.tensor(slant_angle, dtype=self.dtype, device=self.device))

        # Select the radiation field corresponding to the requested pattern
        if pattern == "omni":
            self._radiation_pattern = self._radiation_pattern_omni
        elif pattern == "38.901":
            self._radiation_pattern = self._radiation_pattern_38901
        else:
            # TR 38.922 element defaults (see TR 38.922 Table 4.4.1.2-1 / 5.4.1.2-1)
            # H-plane 3dB = 90 deg, V-plane 3dB = 65 deg, A_m = SLA_V = 30 dB,
            # element peak gain G_E,max ~= 6.4 dBi for many macro cases.
            self.register_buffer("_phi_3db", torch.tensor(90.0 / 180 * PI, dtype=self.dtype, device=self.device))
            self.register_buffer("_theta_3db", torch.tensor(65.0 / 180 * PI, dtype=self.dtype, device=self.device))
            self.register_buffer("_A_m", torch.tensor(30.0, dtype=self.dtype, device=self.device))
            self.register_buffer("_SLA_v", torch.tensor(30.0, dtype=self.dtype, device=self.device))
            self.register_buffer("_G_E_max", torch.tensor(6.4, dtype=self.dtype, device=self.device))
            self._radiation_pattern = self._element_radiation_pattern_38922

    @property
    def pattern(self) -> str:
        """Radiation pattern type ('omni', '38.901', or '38.922')"""
        return self._pattern

    @property
    def slant_angle(self) -> torch.Tensor:
        """Polarization slant angle [radian]"""
        return self._slant_angle

    def field(self, theta: torch.Tensor, phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Field pattern in the vertical and horizontal polarization (7.3-4/5)

        :param theta: Zenith angle wrapped within (0, pi) [radian]
        :param phi: Azimuth angle wrapped within (-pi, pi) [radian]
        """
        theta = theta.to(dtype=self.dtype, device=self.device)
        phi = phi.to(dtype=self.dtype, device=self.device)
        a = torch.sqrt(self._radiation_pattern(theta, phi))
        f_theta = a * torch.cos(self._slant_angle)
        f_phi = a * torch.sin(self._slant_angle)
        return (f_theta, f_phi)

    def show(self) -> None:
        """Shows the field pattern of an antenna element"""
        theta = torch.linspace(0.0, PI, 361, dtype=self.dtype, device=self.device)
        phi = torch.linspace(-PI, PI, 361, dtype=self.dtype, device=self.device)
        a_v = 10 * torch.log10(self._radiation_pattern(theta, torch.zeros_like(theta)))
        a_h = 10 * torch.log10(self._radiation_pattern(PI / 2 * torch.ones_like(phi), phi))

        # Convert to numpy for plotting
        theta_np = theta.cpu().numpy()
        phi_np = phi.cpu().numpy()
        a_v_np = a_v.cpu().numpy()
        a_h_np = a_h.cpu().numpy()

        fig = plt.figure()
        plt.polar(theta_np, a_v_np)
        fig.axes[0].set_theta_zero_location("N")
        fig.axes[0].set_theta_direction(-1)
        plt.title(r"Vertical cut of the radiation pattern ($\phi = 0$)")
        plt.legend([f"{self._pattern}"])

        fig = plt.figure()
        plt.polar(phi_np, a_h_np)
        fig.axes[0].set_theta_zero_location("E")
        plt.title(r"Horizontal cut of the radiation pattern ($\theta = \pi/2$)")
        plt.legend([f"{self._pattern}"])

        theta = torch.linspace(0.0, PI, 50, dtype=self.dtype, device=self.device)
        phi = torch.linspace(-PI, PI, 50, dtype=self.dtype, device=self.device)
        phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing='xy')
        a = self._radiation_pattern(theta_grid, phi_grid)
        x = a * torch.sin(theta_grid) * torch.cos(phi_grid)
        y = a * torch.sin(theta_grid) * torch.sin(phi_grid)
        z = a * torch.cos(theta_grid)

        # Convert to numpy for 3D plotting
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        z_np = z.cpu().numpy()

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(x_np, y_np, z_np, rstride=1, cstride=1,
                        linewidth=0, antialiased=False, alpha=0.5)
        ax.view_init(elev=30., azim=-45)
        plt.xlabel("x")
        plt.ylabel("y")
        ax.set_zlabel("z")
        plt.title(f"Radiation power pattern ({self._pattern})")

    def _radiation_pattern_omni(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Radiation pattern of an omnidirectional 3D radiation pattern

        :param theta: Zenith angle
        :param phi: Azimuth angle
        """
        return torch.ones_like(theta)

    def _radiation_pattern_38901(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Radiation pattern from TR38901 (Table 7.3-1)

        :param theta: Zenith angle wrapped within (0, pi) [radian]
        :param phi: Azimuth angle wrapped within (-pi, pi) [radian]
        """
        theta_3db = phi_3db = 65 / 180 * PI
        a_max = sla_v = 30.0
        g_e_max = 8.0
        a_v = -torch.minimum(12 * ((theta - PI / 2) / theta_3db) ** 2,
                              torch.tensor(sla_v, dtype=self.dtype, device=self.device))
        a_h = -torch.minimum(12 * (phi / phi_3db) ** 2,
                              torch.tensor(a_max, dtype=self.dtype, device=self.device))
        a_db = -torch.minimum(-(a_v + a_h),
                               torch.tensor(a_max, dtype=self.dtype, device=self.device)) + g_e_max
        return 10 ** (a_db / 10)
    
    def _element_radiation_pattern_38922(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Radiation pattern based on TR 38.922 element model (Table 7.1.3-1 / 4.4.1.2)

        Returns the element power gain in linear units (not dB).
        :param theta: Zenith angle wrapped within (0, pi) [radian]
        :param phi: Azimuth angle wrapped within (-pi, pi) [radian]
        """
        # Use registered buffers (set in __init__ when pattern == '38.922')
        theta_3db = self._theta_3db
        phi_3db = self._phi_3db
        A_m = self._A_m
        SLA_v = self._SLA_v
        G_E_max = self._G_E_max

        a_v = -torch.minimum(12 * ((theta - PI / 2) / theta_3db) ** 2,
                              SLA_v)
        a_h = -torch.minimum(12 * (phi / phi_3db) ** 2,
                              A_m)
        a_db = -torch.minimum(-(a_v + a_h), A_m) + G_E_max
        return 10 ** (a_db / 10)

    def _compute_gain(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute antenna gain and directivity through numerical integration"""
        # Create angular meshgrid
        theta = torch.linspace(0.0, PI, 181, dtype=self.dtype, device=self.device)
        phi = torch.linspace(-PI, PI, 361, dtype=self.dtype, device=self.device)
        phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing='xy')

        # Compute field strength over the grid
        f_theta, f_phi = self.field(theta_grid, phi_grid)
        u = f_theta ** 2 + f_phi ** 2
        gain_db = 10 * torch.log10(torch.max(u))

        # Numerical integration of the field components
        dtheta = theta[1] - theta[0]
        dphi = phi[1] - phi[0]
        po = torch.sum(u * torch.sin(theta_grid) * dtheta * dphi)

        # Compute directivity
        u_bar = po / (4 * PI)  # Equivalent isotropic radiator
        d = u / u_bar  # Directivity grid
        directivity_db = 10 * torch.log10(torch.max(d))
        return (gain_db, directivity_db)

class AntennaSubarray(AntennaElement):
    """Antenna subarray and composite array following the TR 38.922 specification.
    Inherits from AntennaElement to reuse the 38.922 element pattern.
    """

    def __init__(
        self,
        slant_angle: float = 0.0,
        # 1. Sub-array Parameters
        subarray_num_rows: int = 4,             # M_sub (from Table 4.4.1.2-1)
        subarray_vertical_spacing: float = 0.7, # d_v,sub (in units of lambda_d)
        subarray_subtilt: float = 3.0,          # theta_subtilt (in degrees)
        # 2. Full Array Parameters (For composite pattern calculations)
        num_array_rows: int = 1,                # M
        num_array_cols: int = 1,                # N
        array_horizontal_spacing: float = 0.5,  # d_h (in units of lambda_d)
        array_vertical_spacing: float = 2.1,    # d_v (in units of lambda_d)
        theta_etilt: float = 0.0,              # theta_etilt (in degrees)
        phi_escan: float = 0.0,                 # phi_escan (in degrees)
        rho: float = 1.0,                       # Correlation factor (rho)
        lambda_ratio: float = 1.0,              # lambda_d / lambda_e (defaults to 1.0)
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(pattern="38.922", slant_angle=slant_angle, precision=precision, device=device)

        # Register Sub-array buffers
        self.register_buffer("_subarray_num_rows", torch.tensor(subarray_num_rows, dtype=torch.int64, device=self.device))
        self.register_buffer("_subarray_vertical_spacing", torch.tensor(subarray_vertical_spacing, dtype=self.dtype, device=self.device))
        self.register_buffer("_subarray_subtilt", torch.tensor(subarray_subtilt / 180 * PI, dtype=self.dtype, device=self.device))

        # Register Full Array buffers
        self.register_buffer("_num_array_rows", torch.tensor(num_array_rows, dtype=torch.int64, device=self.device))
        self.register_buffer("_num_array_cols", torch.tensor(num_array_cols, dtype=torch.int64, device=self.device))
        self.register_buffer("_array_horizontal_spacing", torch.tensor(array_horizontal_spacing, dtype=self.dtype, device=self.device))
        self.register_buffer("_array_vertical_spacing", torch.tensor(array_vertical_spacing, dtype=self.dtype, device=self.device))
        self.register_buffer("_theta_etilt", torch.tensor(theta_etilt / 180 * PI, dtype=self.dtype, device=self.device))
        self.register_buffer("_phi_escan", torch.tensor(phi_escan / 180 * PI, dtype=self.dtype, device=self.device))
        self.register_buffer("_rho", torch.tensor(rho, dtype=self.dtype, device=self.device))
        self.register_buffer("_lambda_ratio", torch.tensor(lambda_ratio, dtype=self.dtype, device=self.device))

        if num_array_rows > 1 or num_array_cols > 1:
            # Default Sionna's internal radiation pattern method to the composite array pattern
            self._radiation_pattern = self.composite_array_radiation_pattern
            raise NotImplementedError("Composite array pattern calculations are not compatible with the current Sionna panel array implementation. Please set num_array_rows and num_array_cols to 1 for now.")
        else:
            # Default Sionna's internal radiation pattern method to the sub-array pattern
            self._radiation_pattern = self.subarray_radiation_pattern

    def subarray_radiation_pattern(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Computes the Sub-array radiation pattern A_sub(theta, phi) in linear units."""
        # 1. Get underlying single element pattern A_E(theta, phi) in linear units
        a_e_linear = self._element_radiation_pattern_38922(theta, phi)

        # 2. Generate sub-array elements 1D index array: shape (M_sub,)
        m = torch.arange(1, self._subarray_num_rows.item() + 1, dtype=self.dtype, device=self.device)

        # 3. Add a trailing dimension to theta to broadcast nicely with the 1D m-index array
        theta_unsqueezed = theta.unsqueeze(-1)

        # 4. Compute phase components: 2 * pi * (m - 1) * d_v,sub / lambda_d * (sin(theta_subtilt) + cos(theta))
        phase = (2.0 * PI * (m - 1) * self._subarray_vertical_spacing * (torch.sin(self._subarray_subtilt) + torch.cos(theta_unsqueezed)))

        # 5. Calculate complex excitation vector (w_m * v_m)
        j_phase = torch.complex(torch.zeros_like(phase), phase)
        w_m_v_m = (1.0 / torch.sqrt(self._subarray_num_rows.to(self.dtype))) * torch.exp(j_phase)

        # 6. Sum across the internal sub-array row elements and find the squared magnitude
        subarray_factor = torch.abs(torch.sum(w_m_v_m, dim=-1)) ** 2

        # Return A_sub = A_E * Subarray_Factor
        return a_e_linear * subarray_factor

    def composite_array_radiation_pattern(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Computes the Full Composite Array radiation pattern A_A(theta, phi) in linear units."""
        # 1. Fetch the baseline sub-array pattern response
        a_sub_linear = self.subarray_radiation_pattern(theta, phi)

        # 2. Setup 1D index arrays for full array grid dimensions
        m_idx = torch.arange(1, self._num_array_rows.item() + 1, dtype=self.dtype, device=self.device)
        n_idx = torch.arange(1, self._num_array_cols.item() + 1, dtype=self.dtype, device=self.device)

        # 3. Shape index matrices to allow parallel broadcasting over the incoming angular shapes
        # If theta/phi have shape (...), we append two dimensions to index matrices: (M, 1) and (1, N)
        # and append two dimensions to angular matrices: (..., 1, 1)
        m_grid = m_idx.view(-1, 1) # Shape: (M, 1)
        n_grid = n_idx.view(1, -1) # Shape: (1, N)
        
        theta_ext = theta.unsqueeze(-1).unsqueeze(-1) # Shape: (..., 1, 1)
        phi_ext = phi.unsqueeze(-1).unsqueeze(-1)     # Shape: (..., 1, 1)

        # 4. Compute Array Excitation Weights (w_m,n) phase using lambda_e (via _lambda_ratio = lambda_d / lambda_e)
        w_phase = 2.0 * PI * self._lambda_ratio * (
            (m_grid - 1) * self._array_vertical_spacing * torch.sin(self._theta_etilt) -
            (n_grid - 1) * self._array_horizontal_spacing * torch.cos(self._theta_etilt) * torch.sin(self._phi_escan)
        )

        # 5. Compute Array Geometry Phase Matrix (v_m,n) using lambda_d
        v_phase = 2.0 * PI * (
            (m_grid - 1) * self._array_vertical_spacing * torch.cos(theta_ext) +
            (n_grid - 1) * self._array_horizontal_spacing * torch.sin(theta_ext) * torch.sin(phi_ext)
        )

        # 6. Combined total phase tensor: shape (..., M, N)
        total_phase = w_phase + v_phase
        j_total_phase = torch.complex(torch.zeros_like(total_phase), total_phase)
        
        # Compute normalized array factor elements
        norm_factor = 1.0 / torch.sqrt((self._num_array_rows * self._num_array_cols).to(self.dtype))
        array_elements = norm_factor * torch.exp(j_total_phase)

        # 7. Double summation over row (dim=-2) and column (dim=-1) arrays
        summed_array_factor = torch.sum(torch.sum(array_elements, dim=-1), dim=-1)
        af_squared_magnitude = torch.abs(summed_array_factor) ** 2

        # 8. Apply the correlation scaling factor equation: 1 + rho * (|Sum|^2 - 1)
        composite_factor = 1.0 + self._rho * (af_squared_magnitude - 1.0)

        # Return A_A = A_sub * Composite_Factor
        return a_sub_linear * composite_factor

class AntennaPanel(Object):
    """Antenna panel following the :cite:p:`TR38901` specification

    :param num_rows: Number of rows forming the panel
    :param num_cols: Number of columns forming the panel
    :param polarization: Polarization. One of ``"single"`` or ``"dual"``.
    :param vertical_spacing: Vertical antenna element spacing
        [multiples of wavelength]
    :param horizontal_spacing: Horizontal antenna element spacing
        [multiples of wavelength]
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., ``"cpu"``, ``"cuda:0"``).
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.
    """

    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        polarization: str,
        vertical_spacing: float,
        horizontal_spacing: float,
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(precision=precision, device=device)
        assert polarization in ('single', 'dual'), \
            "polarization must be either 'single' or 'dual'"

        self._num_rows = num_rows
        self._num_cols = num_cols
        self._polarization = polarization
        # Register as buffers for CUDAGraph compatibility
        self.register_buffer("_horizontal_spacing", torch.tensor(horizontal_spacing, dtype=self.dtype, device=self.device))
        self.register_buffer("_vertical_spacing", torch.tensor(vertical_spacing, dtype=self.dtype, device=self.device))

        # Place the antenna elements of the first polarization direction
        # on the y-z-plane
        p = 1 if polarization == 'single' else 2
        ant_pos = np.zeros([num_rows * num_cols * p, 3])
        for i in range(num_rows):
            for j in range(num_cols):
                ant_pos[i + j * num_rows] = [0,
                                              j * horizontal_spacing,
                                              -i * vertical_spacing]

        # Center the panel around the origin
        offset = [0,
                  -(num_cols - 1) * horizontal_spacing / 2,
                  (num_rows - 1) * vertical_spacing / 2]
        ant_pos += offset

        # Create the antenna elements of the second polarization direction
        if polarization == 'dual':
            ant_pos[num_rows * num_cols:] = ant_pos[:num_rows * num_cols]
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer("_ant_pos", torch.tensor(ant_pos, dtype=self.dtype, device=self.device))

    @property
    def ant_pos(self) -> torch.Tensor:
        """Antenna positions in the local coordinate system"""
        return self._ant_pos

    @property
    def num_rows(self) -> int:
        """Number of rows"""
        return self._num_rows

    @property
    def num_cols(self) -> int:
        """Number of columns"""
        return self._num_cols

    @property
    def polarization(self) -> str:
        """Polarization ('single' or 'dual')"""
        return self._polarization

    @property
    def vertical_spacing(self) -> torch.Tensor:
        """Vertical spacing between elements [multiple of wavelength]"""
        return self._vertical_spacing

    @property
    def horizontal_spacing(self) -> torch.Tensor:
        """Horizontal spacing between elements [multiple of wavelength]"""
        return self._horizontal_spacing

    def show(self) -> None:
        """Shows the panel geometry"""
        fig = plt.figure()
        pos = self._ant_pos[:self._num_rows * self._num_cols].cpu().numpy()
        plt.plot(pos[:, 1], pos[:, 2], marker="|", markeredgecolor='red',
                 markersize="20", linestyle="None", markeredgewidth="2")
        for i, p in enumerate(pos):
            fig.axes[0].annotate(i + 1, (p[1], p[2]))
        if self._polarization == 'dual':
            pos = self._ant_pos[self._num_rows * self._num_cols:].cpu().numpy()
            plt.plot(pos[:, 1], pos[:, 2], marker="_", markeredgecolor='black',
                     markersize="20", linestyle="None", markeredgewidth="1")
        plt.xlabel(r"y ($\lambda_0$)")
        plt.ylabel(r"z ($\lambda_0$)")
        plt.title("Antenna Panel")
        plt.legend(["Polarization 1", "Polarization 2"], loc="upper right")


class PanelArray_38922(PanelArray):
    # pylint: disable=line-too-long
    r"""
    Antenna panel array following the :cite:p:`TR38901` specification

    This class is used to create models of the panel arrays used by the
    transmitters and receivers and that need to be specified when using the
    :class:`~sionna.phy.channel.tr38901.CDL`,
    :class:`~sionna.phy.channel.tr38901.UMi`,
    :class:`~sionna.phy.channel.tr38901.UMa`, and
    :class:`~sionna.phy.channel.tr38901.RMa` models.

    :param num_rows_per_panel: Number of rows of elements per panel
    :param num_cols_per_panel: Number of columns of elements per panel
    :param polarization: Polarization. One of ``"single"`` or ``"dual"``.
    :param polarization_type: Type of polarization. For single polarization,
        must be ``"V"`` or ``"H"``.
        For dual polarization, must be ``"VH"`` or ``"cross"``.
    :param antenna_pattern: Element radiation pattern. One of ``"omni"``
        or ``"38.901"``.
    :param carrier_frequency: Carrier frequency [Hz]
    :param num_rows: Number of rows of panels. Defaults to 1.
    :param num_cols: Number of columns of panels. Defaults to 1.
    :param panel_vertical_spacing: Vertical spacing of panels
        [multiples of wavelength].
        Must be greater than the panel width.
        If set to `None`, it is set to the panel width + 0.5.
    :param panel_horizontal_spacing: Horizontal spacing of panels
        [in multiples of wavelength].
        Must be greater than the panel height.
        If set to `None`, it is set to the panel height + 0.5.
    :param element_vertical_spacing: Element vertical spacing
        [multiple of wavelength].
        Defaults to 0.5 if set to `None`.
    :param element_horizontal_spacing: Element horizontal spacing
        [multiple of wavelength].
        Defaults to 0.5 if set to `None`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., ``"cpu"``, ``"cuda:0"``).
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.channel.tr38901 import PanelArray

        array = PanelArray(num_rows_per_panel=4,
                           num_cols_per_panel=4,
                           polarization='dual',
                           polarization_type='VH',
                           antenna_pattern='38.901',
                           carrier_frequency=3.5e9,
                           num_cols=2,
                           panel_horizontal_spacing=3.)
        array.show()
    """

    def __init__(
        self,
        num_rows_per_panel: int,
        num_cols_per_panel: int,
        polarization: str,
        polarization_type: str,
        antenna_pattern: str,
        carrier_frequency: float,
        num_rows: int = 1,
        num_cols: int = 1,
        panel_vertical_spacing: Optional[float] = None,
        panel_horizontal_spacing: Optional[float] = None,
        element_vertical_spacing: Optional[float] = None,
        element_horizontal_spacing: Optional[float] = None,
        # 38.922 composite pattern parameters (not yet implemented)
        # 1. Sub-array Parameters
        subarray_num_rows: int = 4,             # M_sub (from Table 4.4.1.2-1)
        subarray_vertical_spacing: float = 0.7, # d_v,sub (in units of lambda_d)
        subarray_subtilt: float = 3.0,          # theta_subtilt (in degrees)
        # 2. Full Array Parameters (For composite pattern calculations)
        num_array_rows: int = 1,                # M
        num_array_cols: int = 1,                # N
        array_horizontal_spacing: float = 0.5,  # d_h (in units of lambda_d)
        array_vertical_spacing: float = 2.1,    # d_v (in units of lambda_d)
        theta_etilt: float = 0.0,              # theta_etilt (in degrees)
        phi_escan: float = 0.0,                 # phi_escan (in degrees)
        rho: float = 1.0,                       # Correlation factor (rho)
        lambda_ratio: float = 1.0,              # lambda_d / lambda_e (defaults to 1.0)
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super(PanelArray, self).__init__(precision=precision, device=device)

        assert polarization in ('single', 'dual'), \
            "polarization must be either 'single' or 'dual'"

        # Setting default values for antenna and panel spacings if not
        # specified by the user
        # Default spacing for antenna elements is half a wavelength
        if element_vertical_spacing is None:
            element_vertical_spacing = 0.5
        if element_horizontal_spacing is None:
            element_horizontal_spacing = 0.5
        # Default values of panel spacing is the panel size + 0.5
        if panel_vertical_spacing is None:
            panel_vertical_spacing = (num_rows_per_panel - 1) \
                * element_vertical_spacing + 0.5
        if panel_horizontal_spacing is None:
            panel_horizontal_spacing = (num_cols_per_panel - 1) \
                * element_horizontal_spacing + 0.5

        # Check that panel spacing is larger than panel dimensions
        assert panel_horizontal_spacing > (num_cols_per_panel - 1) \
            * element_horizontal_spacing, \
            "Panel horizontal spacing must be larger than the panel width"
        assert panel_vertical_spacing > (num_rows_per_panel - 1) \
            * element_vertical_spacing, \
            "Panel vertical spacing must be larger than panel height"

        self._num_rows = num_rows
        self._num_cols = num_cols
        self._num_rows_per_panel = num_rows_per_panel
        self._num_cols_per_panel = num_cols_per_panel
        self._polarization = polarization
        self._polarization_type = polarization_type
        # Register as buffers for CUDAGraph compatibility
        self.register_buffer("_panel_vertical_spacing", torch.tensor(panel_vertical_spacing, dtype=self.dtype, device=self.device))
        self.register_buffer("_panel_horizontal_spacing", torch.tensor(panel_horizontal_spacing, dtype=self.dtype, device=self.device))
        self.register_buffer("_element_vertical_spacing", torch.tensor(element_vertical_spacing, dtype=self.dtype, device=self.device))
        self.register_buffer("_element_horizontal_spacing", torch.tensor(element_horizontal_spacing, dtype=self.dtype, device=self.device))

        self._num_panels = num_cols * num_rows

        p = 1 if polarization == 'single' else 2
        self._num_panel_ant = num_cols_per_panel * num_rows_per_panel * p
        # Total number of antenna elements
        self._num_ant = self._num_panels * self._num_panel_ant

        # Wavelength (m)
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer("_lambda_0", torch.tensor(SPEED_OF_LIGHT / carrier_frequency, dtype=self.dtype, device=self.device))

        # Create one antenna element for each polarization direction
        # polarization must be one of {"V", "H", "VH", "cross"}
        if polarization == 'single':
            assert polarization_type in ["V", "H"], \
                "For single polarization, polarization_type must be 'V' or 'H'"
            slant_angle = 0 if polarization_type == "V" else PI / 2
            if antenna_pattern == '38.922':
                self._ant_pol1 = AntennaSubarray(slant_angle=slant_angle,
                                                subarray_num_rows=subarray_num_rows,
                                                subarray_vertical_spacing=subarray_vertical_spacing,
                                                subarray_subtilt=subarray_subtilt,
                                                num_array_rows=num_array_rows,
                                                num_array_cols=num_array_cols,
                                                array_horizontal_spacing=array_horizontal_spacing,
                                                array_vertical_spacing=array_vertical_spacing,                                                    
                                                theta_etilt=theta_etilt,
                                                phi_escan=phi_escan,
                                                rho=rho,
                                                lambda_ratio=lambda_ratio,
                                                precision=self.precision, 
                                                device=self.device)
            else:
                self._ant_pol1 = AntennaElement(antenna_pattern, slant_angle,
                                                precision=self.precision, device=self.device)
            self._ant_pol2 = None
        else:
            assert polarization_type in ["VH", "cross"], \
                "For dual polarization, polarization_type must be 'VH' or 'cross'"
            # For TR 38.922 many AAS configs use linear +/-45deg polarization.
            # If the requested element pattern is 38.922, default to slants of
            # -45deg and +45deg (i.e. polarization_type behaves like 'cross').
            if antenna_pattern == '38.922':
                slant_angle = -PI / 4
            else:
                slant_angle = 0 if polarization_type == "VH" else -PI / 4
            
            if antenna_pattern == '38.922':
                self._ant_pol1 = AntennaSubarray(slant_angle=slant_angle,
                                                subarray_num_rows=subarray_num_rows,
                                                subarray_vertical_spacing=subarray_vertical_spacing,
                                                subarray_subtilt=subarray_subtilt,
                                                num_array_rows=num_array_rows,
                                                num_array_cols=num_array_cols,
                                                array_horizontal_spacing=array_horizontal_spacing,
                                                array_vertical_spacing=array_vertical_spacing,                                                    
                                                theta_etilt=theta_etilt,
                                                phi_escan=phi_escan,
                                                rho=rho,
                                                lambda_ratio=lambda_ratio,
                                                precision=self.precision, 
                                                device=self.device)
                self._ant_pol2 = AntennaSubarray(slant_angle=slant_angle + PI / 2,
                                                subarray_num_rows=subarray_num_rows,
                                                subarray_vertical_spacing=subarray_vertical_spacing,
                                                subarray_subtilt=subarray_subtilt,
                                                num_array_rows=num_array_rows,
                                                num_array_cols=num_array_cols,
                                                array_horizontal_spacing=array_horizontal_spacing,
                                                array_vertical_spacing=array_vertical_spacing,                                                    
                                                theta_etilt=theta_etilt,
                                                phi_escan=phi_escan,
                                                rho=rho,
                                                lambda_ratio=lambda_ratio,
                                                precision=self.precision, 
                                                device=self.device)
            else:
                self._ant_pol1 = AntennaElement(antenna_pattern, slant_angle,
                                                precision=self.precision, device=self.device)
                self._ant_pol2 = AntennaElement(antenna_pattern, slant_angle + PI / 2,
                                                precision=self.precision, device=self.device)

        # Compose array from panels
        ant_pos = np.zeros([self._num_ant, 3])
        panel = AntennaPanel(num_rows_per_panel, num_cols_per_panel,
                             polarization, element_vertical_spacing, element_horizontal_spacing,
                             precision=self.precision, device=self.device)
        pos = panel.ant_pos.cpu().numpy()
        count = 0
        num_panel_ant = self._num_panel_ant
        for j in range(num_cols):
            for i in range(num_rows):
                offset = [0,
                          j * panel_horizontal_spacing,
                          -i * panel_vertical_spacing]
                new_pos = pos + offset
                ant_pos[count * num_panel_ant:(count + 1) * num_panel_ant] = new_pos
                count += 1

        # Center the entire panel array around the origin of the y-z plane
        offset = [0,
                  -(num_cols - 1) * panel_horizontal_spacing / 2,
                  (num_rows - 1) * panel_vertical_spacing / 2]
        ant_pos += offset

        # Scale antenna element positions by the wavelength
        ant_pos *= self._lambda_0.cpu().numpy()
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer("_ant_pos", torch.tensor(ant_pos, dtype=self.dtype, device=self.device))

        # Compute indices of antennas for polarization directions
        ind = np.arange(0, self._num_ant)
        ind = np.reshape(ind, [self._num_panels * p, -1])
        # Register as buffers for CUDAGraph compatibility
        self.register_buffer("_ant_ind_pol1", torch.tensor(np.reshape(ind[::p], [-1]), dtype=torch.int64, device=self.device))
        if polarization == 'single':
            self.register_buffer("_ant_ind_pol2", torch.tensor(np.array([]), dtype=torch.int64, device=self.device))
        else:
            self.register_buffer("_ant_ind_pol2", torch.tensor(np.reshape(
                ind[1:self._num_panels * p:2], [-1]), dtype=torch.int64, device=self.device))

        # Get positions of antenna elements for each polarization direction
        self.register_buffer("_ant_pos_pol1", self._ant_pos[self._ant_ind_pol1])
        self.register_buffer("_ant_pos_pol2", self._ant_pos[self._ant_ind_pol2] if polarization == 'dual' else torch.tensor([], dtype=self.dtype, device=self.device))

    @property
    def num_rows(self) -> int:
        """Number of rows of panels"""
        return self._num_rows

    @property
    def num_cols(self) -> int:
        """Number of columns of panels"""
        return self._num_cols

    @property
    def num_rows_per_panel(self) -> int:
        """Number of rows of elements per panel"""
        return self._num_rows_per_panel

    @property
    def num_cols_per_panel(self) -> int:
        """Number of columns of elements per panel"""
        return self._num_cols_per_panel

    @property
    def polarization(self) -> str:
        """Polarization ('single' or 'dual')"""
        return self._polarization

    @property
    def polarization_type(self) -> str:
        """Polarization type. ``"V"`` or ``"H"`` for single polarization.
        ``"VH"`` or ``"cross"`` for dual polarization."""
        return self._polarization_type

    @property
    def panel_vertical_spacing(self) -> torch.Tensor:
        """Vertical spacing between the panels [multiple of wavelength]"""
        return self._panel_vertical_spacing

    @property
    def panel_horizontal_spacing(self) -> torch.Tensor:
        """Horizontal spacing between the panels [multiple of wavelength]"""
        return self._panel_horizontal_spacing

    @property
    def element_vertical_spacing(self) -> torch.Tensor:
        """Vertical spacing between the antenna elements within a panel
        [multiple of wavelength]"""
        return self._element_vertical_spacing

    @property
    def element_horizontal_spacing(self) -> torch.Tensor:
        """Horizontal spacing between the antenna elements within a panel
        [multiple of wavelength]"""
        return self._element_horizontal_spacing

    @property
    def num_panels(self) -> int:
        """Number of panels"""
        return self._num_panels

    @property
    def num_panels_ant(self) -> int:
        """Number of antenna elements per panel"""
        return self._num_panel_ant

    @property
    def num_ant(self) -> int:
        """Total number of antenna elements"""
        return self._num_ant

    @property
    def ant_pol1(self) -> AntennaElement:
        """Field of an antenna element with the first polarization direction"""
        return self._ant_pol1

    @property
    def ant_pol2(self) -> AntennaElement:
        """Field of an antenna element with the second polarization direction.
        Only defined with dual polarization."""
        assert self._polarization == 'dual', \
            "This property is not defined with single polarization"
        return self._ant_pol2

    @property
    def ant_pos(self) -> torch.Tensor:
        """Positions of the antennas"""
        return self._ant_pos

    @property
    def ant_ind_pol1(self) -> torch.Tensor:
        """Indices of antenna elements with the first polarization direction"""
        return self._ant_ind_pol1

    @property
    def ant_ind_pol2(self) -> torch.Tensor:
        """Indices of antenna elements with the second polarization direction.
        Only defined with dual polarization."""
        assert self._polarization == 'dual', \
            "This property is not defined with single polarization"
        return self._ant_ind_pol2

    @property
    def ant_pos_pol1(self) -> torch.Tensor:
        """Positions of the antenna elements with the first polarization
        direction"""
        return self._ant_pos_pol1

    @property
    def ant_pos_pol2(self) -> torch.Tensor:
        """Positions of antenna elements with the second polarization direction.
        Only defined with dual polarization."""
        assert self._polarization == 'dual', \
            "This property is not defined with single polarization"
        return self._ant_pos_pol2

    def show(self) -> None:
        """Show the panel array geometry"""
        if self._polarization == 'single':
            if self._polarization_type == 'H':
                marker_p1 = MarkerStyle("_").get_marker()
            else:
                marker_p1 = MarkerStyle("|")
        else:  # 'dual'
            if self._polarization_type == 'cross':
                marker_p1 = (2, 0, -45)
                marker_p2 = (2, 0, 45)
            else:
                marker_p1 = MarkerStyle("_").get_marker()
                marker_p2 = MarkerStyle("|").get_marker()

        fig = plt.figure()
        pos_pol1 = self._ant_pos_pol1.cpu().numpy()
        plt.plot(pos_pol1[:, 1], pos_pol1[:, 2],
                 marker=marker_p1, markeredgecolor='red',
                 markersize="20", linestyle="None", markeredgewidth="2")
        ant_ind_pol1 = self._ant_ind_pol1.cpu().numpy()
        for i, p in enumerate(pos_pol1):
            fig.axes[0].annotate(ant_ind_pol1[i] + 1, (p[1], p[2]))
        if self._polarization == 'dual':
            pos_pol2 = self._ant_pos_pol2.cpu().numpy()
            plt.plot(pos_pol2[:, 1], pos_pol2[:, 2],
                     marker=marker_p2,  # pylint: disable=possibly-used-before-assignment
                     markeredgecolor='black',
                     markersize="20", linestyle="None", markeredgewidth="1")
        plt.xlabel("y (m)")
        plt.ylabel("z (m)")
        plt.title("Panel Array")
        plt.legend(["Polarization 1", "Polarization 2"], loc="upper right")

    def show_element_radiation_pattern(self) -> None:
        """Show the radiation field of antenna elements forming the panel"""
        self._ant_pol1.show()


class Antenna(PanelArray_38922):
    # pylint: disable=line-too-long
    r"""
    Single antenna following the :cite:p:`TR38901` specification

    This class is a special case of :class:`~sionna.phy.channel.tr38901.PanelArray`,
    and can be used in lieu of it.

    :param polarization: Polarization. One of ``"single"`` or ``"dual"``.
    :param polarization_type: Type of polarization. For single polarization,
        must be ``"V"`` or ``"H"``.
        For dual polarization, must be ``"VH"`` or ``"cross"``.
    :param antenna_pattern: Element radiation pattern. One of ``"omni"``
        or ``"38.901"``.
    :param carrier_frequency: Carrier frequency [Hz]
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., ``"cpu"``, ``"cuda:0"``).
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.channel.tr38901 import Antenna

        ant = Antenna(polarization='single',
                      polarization_type='V',
                      antenna_pattern='omni',
                      carrier_frequency=3.5e9)
        print(ant.num_ant)
        # 1
    """

    def __init__(
        self,
        polarization: str,
        polarization_type: str,
        antenna_pattern: str,
        carrier_frequency: float,
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization=polarization,
            polarization_type=polarization_type,
            antenna_pattern=antenna_pattern,
            carrier_frequency=carrier_frequency,
            precision=precision,
            device=device,
        )


class AntennaArray(PanelArray_38922):
    # pylint: disable=line-too-long
    r"""
    Antenna array following the :cite:p:`TR38901` specification

    This class is a special case of :class:`~sionna.phy.channel.tr38901.PanelArray`,
    and can be used in lieu of it.

    :param num_rows: Number of rows of elements
    :param num_cols: Number of columns of elements
    :param polarization: Polarization. One of ``"single"`` or ``"dual"``.
    :param polarization_type: Type of polarization. For single polarization,
        must be ``"V"`` or ``"H"``.
        For dual polarization, must be ``"VH"`` or ``"cross"``.
    :param antenna_pattern: Element radiation pattern. One of ``"omni"``
        or ``"38.901"``.
    :param carrier_frequency: Carrier frequency [Hz]
    :param vertical_spacing: Element vertical spacing [multiple of wavelength].
        Defaults to 0.5 if set to `None`.
    :param horizontal_spacing: Element horizontal spacing [multiple of wavelength].
        Defaults to 0.5 if set to `None`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., ``"cpu"``, ``"cuda:0"``).
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.channel.tr38901 import AntennaArray

        array = AntennaArray(num_rows=4,
                             num_cols=4,
                             polarization='dual',
                             polarization_type='cross',
                             antenna_pattern='38.901',
                             carrier_frequency=3.5e9)
        print(array.num_ant)
        # 32
    """

    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        polarization: str,
        polarization_type: str,
        antenna_pattern: str,
        carrier_frequency: float,
        vertical_spacing: Optional[float] = None,
        horizontal_spacing: Optional[float] = None,
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(
            num_rows_per_panel=num_rows,
            num_cols_per_panel=num_cols,
            polarization=polarization,
            polarization_type=polarization_type,
            antenna_pattern=antenna_pattern,
            carrier_frequency=carrier_frequency,
            element_vertical_spacing=vertical_spacing,
            element_horizontal_spacing=horizontal_spacing,
            precision=precision,
            device=device,
        )

