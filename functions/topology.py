#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Multicell topology generation for Sionna SYS"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch

from sionna.phy import PI, Block, Object, config, dtypes
from sionna.phy.channel.utils import random_ut_properties, set_3gpp_scenario_parameters
from sionna.phy.config import Precision
from sionna.phy.utils import flatten_dims, insert_dims, sample_bernoulli

__all__ = [
    "get_num_hex_in_grid",
    "convert_hex_coord",
    "Hexagon",
    "HexGrid",
    "gen_hexgrid_topology",
]


def get_num_hex_in_grid(num_rings: int) -> int:
    r"""Computes the number of hexagons in a spiral hexagonal grid with a given
    number of rings :math:`N`. It equals :math:`1+3N(N+1)`.

    :param num_rings: Number of rings of the hexagonal spiral grid

    :output num_hexagons: Number of hexagons in the spiral hexagonal grid

    .. rubric:: Examples

    .. code-block:: python

        from sionna.sys import get_num_hex_in_grid

        print(get_num_hex_in_grid(1))
        # 7
        print(get_num_hex_in_grid(2))
        # 19
    """
    return 1 + 3 * num_rings * (num_rings + 1)


def convert_hex_coord(
    coord: torch.Tensor,
    conversion_type: str,
    hex_radius: Optional[torch.Tensor] = None,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Converts the center coordinates of a hexagon within a grid between any two
    of the types {"offset", "axial", "euclid"}.

    :param coord: Coordinates of the center of a hexagon contained in a
        hexagonal grid with shape [..., 2]
    :param conversion_type: Type of coordinate conversion. One of
        'offset2euclid', 'euclid2offset', 'euclid2axial', 'offset2axial',
        'axial2offset', 'axial2euclid'.
    :param hex_radius: Hexagon radius, i.e., distance between its center and any of
        its corners with shape [...]. It must be specified if ``conversion_type``
        is 'offset2euclid', 'axial2euclid', 'euclid2offset', or 'euclid2axial'.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output coord_out: Output coordinates with shape [..., 2]

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.sys import convert_hex_coord

        # Convert offset to Euclidean coordinates
        offset_coord = torch.tensor([1, 2])
        euclid = convert_hex_coord(offset_coord, 'offset2euclid', hex_radius=1.0)
        print(euclid)
        # tensor([1.5000, 4.3301])
    """
    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]

    if device is None:
        device = config.device

    def inter_center_distance() -> Tuple[torch.Tensor, torch.Tensor]:
        # Inter-center distance between two horizontally adjacent hexagons
        dist_x = hex_radius * 1.5
        # Inter-center distance between two vertically adjacent hexagons
        dist_y = hex_radius * math.sqrt(3.0)
        return dist_x, dist_y

    valid_types = [
        "offset2euclid",
        "euclid2offset",
        "euclid2axial",
        "offset2axial",
        "axial2offset",
        "axial2euclid",
    ]
    assert conversion_type in valid_types, (
        f"Invalid conversion_type. Must be one of {valid_types}"
    )

    if conversion_type.startswith("euclid"):
        coord = coord.to(dtype=dtype, device=device)
    else:
        coord = coord.to(dtype=torch.int32, device=device)

    if hex_radius is not None:
        if not isinstance(hex_radius, torch.Tensor):
            hex_radius = torch.tensor(hex_radius, dtype=dtype, device=device)
        else:
            hex_radius = hex_radius.to(dtype=dtype, device=device)
        # Broadcast to match coord shape (excluding last dim)
        while hex_radius.dim() < coord.dim() - 1:
            hex_radius = hex_radius.unsqueeze(0)

    if conversion_type == "offset2euclid":
        assert hex_radius is not None, (
            "hex_radius must be specified for 'offset2euclid'"
        )
        col, row = coord[..., 0], coord[..., 1]
        dist_x, dist_y = inter_center_distance()
        # Euclidean coordinates
        col_f = col.to(dtype)
        row_f = row.to(dtype)
        x = col_f * dist_x
        y = row_f * dist_y + (col % 2).to(dtype) * dist_y / 2
        coord_out = torch.stack([x, y], dim=-1)

    elif conversion_type == "euclid2offset":
        assert hex_radius is not None, (
            "hex_radius must be specified for 'euclid2offset'"
        )
        x, y = coord[..., 0], coord[..., 1]
        dist_x, dist_y = inter_center_distance()
        col = x / dist_x
        # Use float modulo (matching TF behavior) before casting to int
        row = (y - (col % 2) * dist_y / 2) / dist_y
        col = col.to(torch.int32)
        row = row.to(torch.int32)
        coord_out = torch.stack([col, row], dim=-1)

    elif conversion_type == "euclid2axial":
        assert hex_radius is not None, (
            "hex_radius must be specified for 'euclid2axial'"
        )
        coord_offset = convert_hex_coord(
            coord,
            conversion_type="euclid2offset",
            hex_radius=hex_radius,
            precision=precision,
            device=device,
        )
        coord_out = convert_hex_coord(
            coord_offset,
            conversion_type="offset2axial",
            precision=precision,
            device=device,
        )

    elif conversion_type == "offset2axial":
        col, row = coord[..., 0], coord[..., 1]
        q = col.to(torch.int32)
        r = row - ((col - (col % 2)) // 2).to(torch.int32)
        coord_out = torch.stack([q, r], dim=-1)

    elif conversion_type == "axial2offset":
        q, r = coord[..., 0], coord[..., 1]
        col = q.to(torch.int32)
        row = r + ((q - (q % 2)) // 2).to(torch.int32)
        coord_out = torch.stack([col, row], dim=-1)

    else:  # axial2euclid
        coord_offset = convert_hex_coord(
            coord,
            conversion_type="axial2offset",
            precision=precision,
            device=device,
        )
        coord_out = convert_hex_coord(
            coord_offset,
            conversion_type="offset2euclid",
            hex_radius=hex_radius,
            precision=precision,
            device=device,
        )

    return coord_out


class Hexagon(Object):
    """Class defining a hexagon placed in a hexagonal grid.

    :param radius: Hexagon radius, defined as the distance between the hexagon
        center and any of its corners
    :param coord: Coordinates of the hexagon center within the grid with
        shape [2]. If ``coord_type`` is 'euclid', the unit of measurement
        is meters [m].
    :param coord_type: Coordinate type of ``coord``. One of 'offset'
        (default), 'axial', or 'euclid'.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.
    """

    def __init__(
        self,
        radius: float,
        coord: Union[List[int], Tuple[int, int], torch.Tensor],
        coord_type: str = "offset",
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(precision=precision, device=device)

        self._coord_offset: Optional[torch.Tensor] = None
        self._coord_axial: Optional[torch.Tensor] = None
        self._coord_euclid: Optional[torch.Tensor] = None
        self._radius: Optional[torch.Tensor] = None

        if coord_type not in ["offset", "axial", "euclid"]:
            raise ValueError("Invalid input value for coord_type")

        # Set radius first (needed for coordinate conversions)
        self._radius = torch.tensor(radius, dtype=self.dtype, device=self.device)

        if coord_type == "offset":
            self.coord_offset = coord
        elif coord_type == "axial":
            self.coord_axial = coord
        else:  # coord_type == 'euclid'
            self.coord_euclid = coord

        self._neighbor_axial_directions = torch.tensor(
            [[1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]],
            dtype=torch.int32,
            device=self.device,
        )

    @property
    def coord_offset(self) -> torch.Tensor:
        """[2], `torch.int32` : Offset coordinates of the hexagon within a grid.
        The first (second) coordinate defines the horizontal (vertical) offset
        with respect to the grid center.

        .. figure:: ../figures/offset_coord.png
            :align: center
        """
        return self._coord_offset

    @coord_offset.setter
    def coord_offset(self, value: Union[List[int], Tuple[int, int], torch.Tensor]) -> None:
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.int32, device=self.device)
        self._coord_offset = value.to(dtype=torch.int32, device=self.device)

        # Compute axial coordinates
        self._coord_axial = convert_hex_coord(
            self._coord_offset,
            conversion_type="offset2axial",
            precision=self.precision,
            device=self.device,
        )

        # Compute Euclidean center
        self._coord_euclid = convert_hex_coord(
            self._coord_offset,
            conversion_type="offset2euclid",
            hex_radius=self._radius,
            precision=self.precision,
            device=self.device,
        )

    @property
    def coord_axial(self) -> torch.Tensor:
        r"""[2], `torch.int32` : Axial coordinates of the hexagon within a grid.

        .. figure:: ../figures/axial_coord.png
            :align: center

        The basis of axial coordinates are 2D vectors
        :math:`\mathbf{b}^{(1)}=\left(\frac{3}{2}r,\frac{\sqrt{3}}{2}r \right)`,
        :math:`\mathbf{b}^{(2)}=\left(0, \sqrt{3}r \right)`. Thus, the
        relationship between axial coordinates :math:`\mathbf{a}=(a_1,a_2)` and
        their corresponding Euclidean ones :math:`\mathbf{x}=(x_1,x_2)` is the
        following:

        .. math::
            \mathbf{x} = a_1 \mathbf{b}^{(1)} + a_2 \mathbf{b}^{(2)}

        .. figure:: ../figures/axial_coord_basis.png
            :align: center
            :width: 70%
        """
        return self._coord_axial

    @coord_axial.setter
    def coord_axial(self, value: Union[List[int], Tuple[int, int], torch.Tensor]) -> None:
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.int32, device=self.device)
        self._coord_axial = value.to(dtype=torch.int32, device=self.device)

        # Compute offset coordinates
        self._coord_offset = convert_hex_coord(
            self._coord_axial,
            conversion_type="axial2offset",
            precision=self.precision,
            device=self.device,
        )

        # Compute Euclidean center
        self._coord_euclid = convert_hex_coord(
            self._coord_offset,
            conversion_type="offset2euclid",
            hex_radius=self._radius,
            precision=self.precision,
            device=self.device,
        )

    @property
    def coord_euclid(self) -> torch.Tensor:
        """[2], `torch.float` : Euclidean coordinates of the hexagon within a grid.

        .. figure:: ../figures/euclid_coord.png
            :align: center
        """
        return self._coord_euclid

    @coord_euclid.setter
    def coord_euclid(self, value: Union[List[float], Tuple[float, float], torch.Tensor]) -> None:
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=self.dtype, device=self.device)
        value = value.to(dtype=self.dtype, device=self.device)

        # Compute offset coordinates
        self._coord_offset = convert_hex_coord(
            value,
            conversion_type="euclid2offset",
            hex_radius=self._radius,
            precision=self.precision,
            device=self.device,
        )

        # Convert back to Euclidean coordinates (snap to grid)
        self._coord_euclid = convert_hex_coord(
            self._coord_offset,
            conversion_type="offset2euclid",
            hex_radius=self._radius,
            precision=self.precision,
            device=self.device,
        )

        # Compute axial coordinates
        self._coord_axial = convert_hex_coord(
            self._coord_offset,
            conversion_type="offset2axial",
            precision=self.precision,
            device=self.device,
        )

    @property
    def radius(self) -> torch.Tensor:
        """`torch.float` : Hexagon radius, defined as the distance between its
        center and any of its corners.
        """
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        self._radius = torch.tensor(value, dtype=self.dtype, device=self.device)
        if self._coord_offset is not None:
            # Update Euclidean coordinates
            self._coord_euclid = convert_hex_coord(
                self._coord_offset,
                conversion_type="offset2euclid",
                hex_radius=self._radius,
                precision=self.precision,
                device=self.device,
            )

    def corners(self) -> torch.Tensor:
        """Computes the Euclidean coordinates of the 6 corners of the hexagon.

        :output corners: Euclidean coordinates of the 6 corners with shape [6, 2],
            `torch.float`
        """
        angles = torch.arange(6, dtype=self.dtype, device=self.device) * PI / 3
        corners = torch.stack(
            [self._radius * torch.cos(angles), self._radius * torch.sin(angles)],
            dim=1,
        )
        return self._coord_euclid.unsqueeze(0) + corners

    def neighbor(self, axial_direction_idx: int) -> "Hexagon":
        """Returns the neighboring hexagon over the specified axial direction.

        :param axial_direction_idx: Index determining the neighbor relative
            axial direction with respect to the current hexagon. Must be one
            of {0,...,5}.

        :output neighbor: :class:`~sionna.sys.topology.Hexagon` -- Neighboring hexagon,
            in the axial relative direction
        """
        neighbor_coord_axial = [
            (self._coord_axial[0] + self._neighbor_axial_directions[axial_direction_idx][0]).item(),
            (self._coord_axial[1] + self._neighbor_axial_directions[axial_direction_idx][1]).item(),
        ]
        return Hexagon(
            radius=self._radius.item(),
            coord=neighbor_coord_axial,
            coord_type="axial",
            precision=self.precision,
            device=self.device,
        )

    def coord_dict(self) -> Dict[str, torch.Tensor]:
        """Returns the hexagon coordinates in the form of a dictionary.

        :output coord_dict: `dict` -- Dictionary containing the three hexagon coordinates,
            with keys 'euclid', 'offset', 'axial'
        """
        return {
            "euclid": self._coord_euclid,
            "offset": self._coord_offset,
            "axial": self._coord_axial,
        }


class HexGrid(Block):
    r"""Creates a hexagonal spiral grid of cells, drops users uniformly at
    random and computes wraparound distances and base station positions.

    Cell sectors are numbered as follows:

    .. figure:: ../figures/multicell_sectors.png
        :align: center
        :width: 80%

    To eliminate border effects that would cause users at the edge of the grid
    to experience reduced interference, the wraparound principle artificially
    translates each base station to its closest corresponding "mirror" image in
    a neighboring hexagon for each user.

    .. figure:: ../figures/wraparound.png
        :align: center

    :param num_rings: Number of spiral rings in the grid
    :param cell_radius: Radius of each hexagonal cell in the grid, defined as
        the distance between the cell center and any of its corners. Either
        ``isd`` or ``cell_radius`` must be specified.
    :param cell_height: Cell height [m]. Defaults to 0.
    :param isd: Inter-site distance. Either ``isd`` or ``cell_radius`` must
        be specified.
    :param center_loc: Coordinates of the grid center with shape [2].
        Defaults to (0, 0).
    :param center_loc_type: Coordinate type of ``center_loc``. One of
        'offset' (default), 'axial', or 'euclid'.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input batch_size: `int`.
        Batch size.
    :input num_ut_per_sector: `int`.
        Number of users to sample per sector and per batch.
    :input min_bs_ut_dist: `float`.
        Minimum distance between a base station (BS) and a user [m].
    :input max_bs_ut_dist: `float` | `None`.
        Maximum distance between a base station (BS) and a user [m]. If
        `None`, it is not considered.
    :input min_ut_height: `float`.
        Minimum user height [m]. Defaults to 0.
    :input max_ut_height: `float`.
        Maximum user height [m]. Defaults to 0.

    :output ut_loc: [batch_size, num_cells, num_sectors=3, num_ut_per_sector, 3], `torch.float`.
        Location of users, dropped uniformly at random within each sector.
    :output mirror_cell_per_ut_loc: [batch_size, num_cells, num_sectors=3, num_ut_per_sector, num_cells, 3], `torch.float`.
        Coordinates of the artificial mirror cell centers, located
        at Euclidean distance ``wraparound_dist`` from each user.
    :output wraparound_dist: [batch_size, num_cells, num_sectors=3, num_ut_per_sector, num_cells], `torch.float`.
        Wraparound distance in the X-Y plane between each user
        and the cell centers.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.sys import HexGrid

        # Create a hexagonal grid with a specified radius and number of rings
        grid = HexGrid(cell_radius=1,
                       cell_height=10,
                       num_rings=1,
                       center_loc=(0, 0))

        # Cell center locations
        print(grid.cell_loc)
        # tensor([[ 0.0000,  0.0000, 10.0000],
        #         [-1.5000,  0.8660, 10.0000],
        #         [ 0.0000,  1.7321, 10.0000],
        #         [ 1.5000,  0.8660, 10.0000],
        #         [ 1.5000, -0.8660, 10.0000],
        #         [ 0.0000, -1.7321, 10.0000],
        #         [-1.5000, -0.8660, 10.0000]])
    """

    def __init__(
        self,
        num_rings: int,
        cell_radius: Optional[float] = None,
        cell_height: float = 0.0,
        isd: Optional[float] = None,
        center_loc: Union[List[int], Tuple[int, int]] = (0, 0),
        center_loc_type: str = "offset",
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(precision=precision, device=device)

        if (cell_radius is None and isd is None) or (
            cell_radius is not None and isd is not None
        ):
            raise ValueError(
                "Exactly one of {'cell_radius', 'isd'} must be provided as input"
            )

        self._grid: Dict[int, Hexagon] = {}
        self._num_rings: Optional[int] = None
        self._cell_radius: Optional[torch.Tensor] = None
        self._isd: Optional[torch.Tensor] = None
        self._cell_height: Optional[torch.Tensor] = None
        self._mirror_cell_loc: Optional[torch.Tensor] = None
        self._mirror_displacements_offset: Optional[torch.Tensor] = None
        self._mirror_displacements_euclid: Optional[torch.Tensor] = None
        self._center_loc_type = center_loc_type
        self._center_loc: Optional[torch.Tensor] = None

        self.center_loc = center_loc
        self.cell_height = cell_height
        if cell_radius is not None:
            self.cell_radius = cell_radius
        if isd is not None:
            self.isd = isd
        self.num_rings = num_rings

    @property
    def grid(self) -> Dict[int, Hexagon]:
        """`dict` : Collection of :class:`~sionna.sys.topology.Hexagon` objects
        corresponding to the cells in the grid.
        """
        return self._grid

    @property
    def cell_loc(self) -> torch.Tensor:
        """[num_cells, 3], `torch.float` : Euclidean coordinates of the cell centers [m]."""
        cell_locs = [cell.coord_euclid for _, cell in self._grid.items()]
        cell_loc = torch.stack(cell_locs, dim=0)
        cell_height = torch.full(
            (cell_loc.shape[0], 1), self._cell_height.item(),
            dtype=self.dtype, device=self.device
        )
        return torch.cat([cell_loc, cell_height], dim=-1)

    @property
    def center_loc(self) -> torch.Tensor:
        """[2], `int` | `float` : Grid center coordinates in the X-Y plane,
        of type ``center_loc_type``.
        """
        return self._center_loc

    @center_loc.setter
    def center_loc(self, value: Union[List, Tuple, torch.Tensor]) -> None:
        if self._center_loc_type == "euclid":
            dtype = self.dtype
        else:
            dtype = torch.int32
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=dtype, device=self.device)
        self._center_loc = value.to(dtype=dtype, device=self.device)
        if self._num_rings is not None and self._cell_radius is not None:
            self._compute_grid()

    @property
    def num_rings(self) -> int:
        """`int` : Number of rings of the spiral grid."""
        return self._num_rings

    @num_rings.setter
    def num_rings(self, value: int) -> None:
        assert value > 0, "The number of rings must be positive"
        self._num_rings = value
        if self._cell_radius is not None:
            self._compute_grid()
            self._get_mirror_displacements()
            self._get_mirror_cell_loc()

    @property
    def num_cells(self) -> int:
        """`int` : Number of cells in the grid."""
        return len(self._grid)

    @property
    def cell_radius(self) -> torch.Tensor:
        """`torch.float` : Radius of any hexagonal cell in the grid [m]."""
        return self._cell_radius

    @cell_radius.setter
    def cell_radius(self, value: float) -> None:
        assert value > 0, "The cell radius must be positive"
        self._cell_radius = torch.tensor(value, dtype=self.dtype, device=self.device)
        self._isd = self._cell_radius * math.sqrt(3.0)
        for _, cell in self._grid.items():
            cell.radius = self._cell_radius.item()
        if self._num_rings is not None:
            self._get_mirror_displacements()
            self._get_mirror_cell_loc()

    @property
    def isd(self) -> torch.Tensor:
        """`torch.float` : Inter-site Euclidean distance [m]."""
        return self._isd

    @isd.setter
    def isd(self, value: float) -> None:
        assert value > 0, "The inter-site distance must be positive"
        self._isd = torch.tensor(value, dtype=self.dtype, device=self.device)
        self._cell_radius = self._isd / math.sqrt(3.0)
        for _, cell in self._grid.items():
            cell.radius = self._cell_radius.item()
        if self._num_rings is not None:
            self._get_mirror_displacements()
            self._get_mirror_cell_loc()

    @property
    def cell_height(self) -> torch.Tensor:
        """`torch.float` : Cell height [m]."""
        return self._cell_height

    @cell_height.setter
    def cell_height(self, value: float) -> None:
        assert value >= 0, "The cell height must be non-negative"
        self._cell_height = torch.tensor(value, dtype=self.dtype, device=self.device)

    @property
    def mirror_cell_loc(self) -> torch.Tensor:
        """[num_cells, num_mirror_grids+1=7, 3], `torch.float` : Euclidean
        (x,y,z) coordinates (axis=2) of the 6 mirror + base cells (axis=1)
        for each base cell (axis=0).
        """
        return self._mirror_cell_loc

    def _get_mirror_cell_loc(self) -> None:
        """For each cell (axis=0), returns the coordinates (axis=2) of the
        corresponding mirror cells (axis=1).
        """
        # [7, 3]
        mirror_displacements_euclid_3d = torch.cat(
            [
                self._mirror_displacements_euclid,
                torch.zeros(7, 1, dtype=self.dtype, device=self.device),
            ],
            dim=-1,
        )
        # [num_cells, 1, 3] + [1, 7, 3]
        self._mirror_cell_loc = (
            self.cell_loc.unsqueeze(1) + mirror_displacements_euclid_3d.unsqueeze(0)
        )

    def _get_mirror_displacements(self) -> None:
        """Computes the 2D displacement between the grid center and the mirror
        grid centers, in both offset and Euclidean coordinates.
        """
        nr = self._num_rings
        # [7, 2]
        self._mirror_displacements_offset = torch.tensor(
            [
                [0, 0],
                [2 * nr + 1, 0],
                [nr, int(3 * nr / 2 + 1 - 0.5 * (nr & 1))],
                [-nr - 1, int(3 * nr / 2 + 0.5 * (nr & 1))],
                [-(2 * nr + 1), -1],
                [-nr, -int(3 * nr / 2 + 0.5 * (nr & 1) + 1)],
                [nr + 1, -int(3 * nr / 2 + 1 - 0.5 * (nr & 1))],
            ],
            dtype=torch.int32,
            device=self.device,
        )

        # [7, 2]
        self._mirror_displacements_euclid = convert_hex_coord(
            self._mirror_displacements_offset,
            conversion_type="offset2euclid",
            hex_radius=self._cell_radius,
            precision=self.precision,
            device=self.device,
        )


    def _sample_local_sector_angles(
        self,
        batch_size: int,
        num_ut_per_sector: int,
        min_azimuth_separation_deg: Optional[float],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Sample per-sector local azimuths in [0, 120 deg].

        If ``min_azimuth_separation_deg`` is positive, the returned angles are
        separated by at least that amount within each sector. This controls the
        UE azimuth directions that later determine the beam scan angles
        ``phi_escan`` for MU-MIMO users in a sector.
        """
        sector_width = 2.0 * math.pi / 3.0
        shape = (batch_size, self.num_cells, 3, num_ut_per_sector)

        if min_azimuth_separation_deg is None or min_azimuth_separation_deg <= 0.0:
            return torch.rand(
                *shape,
                dtype=self.dtype,
                device=self.device,
                generator=generator,
            ) * sector_width

        min_sep = float(min_azimuth_separation_deg) * math.pi / 180.0
        if num_ut_per_sector > 1 and (num_ut_per_sector - 1) * min_sep > sector_width:
            raise ValueError(
                f"Cannot place {num_ut_per_sector} UEs in one 120 degree sector "
                f"with {min_azimuth_separation_deg} degree minimum azimuth separation."
            )

        alpha_local = torch.empty(*shape, dtype=self.dtype, device=self.device)
        max_attempts_per_ue = 10000

        for b in range(batch_size):
            for c in range(self.num_cells):
                for sec in range(3):
                    accepted = []
                    for u in range(num_ut_per_sector):
                        cand = None
                        for _ in range(max_attempts_per_ue):
                            cand_t = torch.rand(
                                (),
                                dtype=self.dtype,
                                device=self.device,
                                generator=generator,
                            ) * sector_width
                            cand_f = float(cand_t.detach().cpu())
                            if all(abs(cand_f - prev) >= min_sep for prev in accepted):
                                cand = cand_t
                                accepted.append(cand_f)
                                break
                        if cand is None:
                            raise RuntimeError(
                                "Failed to sample azimuth-separated UE angles. "
                                "Try a smaller minimum separation or fewer UEs per sector."
                            )
                        alpha_local[b, c, sec, u] = cand

        return alpha_local

    def call(
        self,
        batch_size: int,
        num_ut_per_sector: int,
        min_bs_ut_dist: float,
        max_bs_ut_dist: Optional[float] = None,
        min_ut_height: float = 0.0,
        max_ut_height: float = 0.0,
        min_ue_azimuth_separation_deg: Optional[float] = 10.0,
        ue_elevation_mode: str = "None",
        ue_elevation_angle_deg: Optional[float] = None,
        fixed_ue_height: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Drops users uniformly at random and computes wraparound distances."""
        if torch.is_tensor(min_ut_height):
            min_ut_height = min_ut_height.detach().clone().to(dtype=self.dtype, device=self.device)
        else:
            min_ut_height = torch.tensor(min_ut_height, dtype=self.dtype, device=self.device)
        if torch.is_tensor(max_ut_height):
            max_ut_height = max_ut_height.detach().clone().to(dtype=self.dtype, device=self.device)
        else:
            max_ut_height = torch.tensor(max_ut_height, dtype=self.dtype, device=self.device)
        assert max_ut_height >= min_ut_height, "max_ut_height must be >= min_ut_height"

        # Cast to dtype
        if torch.is_tensor(min_bs_ut_dist):
            min_bs_ut_dist = min_bs_ut_dist.detach().clone().to(dtype=self.dtype, device=self.device)
        else:
            min_bs_ut_dist = torch.tensor(min_bs_ut_dist, dtype=self.dtype, device=self.device)
        if max_bs_ut_dist is None:
            max_bs_ut_dist = self._cell_radius
        elif torch.is_tensor(max_bs_ut_dist):
            max_bs_ut_dist = max_bs_ut_dist.detach().clone().to(dtype=self.dtype, device=self.device)
        else:
            max_bs_ut_dist = torch.tensor(max_bs_ut_dist, dtype=self.dtype, device=self.device)
        assert min_bs_ut_dist <= max_bs_ut_dist, (
            "min_bs_ut_dist must not exceed max_bs_ut_dist"
        )

        # Minimum cell-UT vertical distance
        cell_height = self._cell_height
        if max_ut_height >= cell_height >= min_ut_height:
            cell_ut_min_dist_z = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        else:
            cell_ut_min_dist_z = torch.minimum(
                torch.abs(cell_height - min_ut_height),
                torch.abs(cell_height - max_ut_height),
            )

        # Maximum cell-UT vertical distance
        cell_ut_max_dist_z = torch.maximum(
            torch.abs(cell_height - min_ut_height),
            torch.abs(cell_height - max_ut_height),
        )

        # Force minimum BS-UT distance >= their height difference
        min_bs_ut_dist = torch.maximum(min_bs_ut_dist, cell_ut_min_dist_z)

        # Minimum squared distance between BS and UT on the X-Y plane
        r_min2 = min_bs_ut_dist**2 - cell_ut_min_dist_z**2

        # Maximum squared distance between BS and UT on the X-Y plane
        r_max2 = max_bs_ut_dist**2 - cell_ut_max_dist_z**2

        # Check the consistency of input parameters
        assert torch.sqrt(r_min2) <= self._isd / 2, (
            "The minimum BS-UT distance cannot be larger than half the inter-site distance"
        )

        # -------- #
        # UT drop  #
        # -------- #
        # Broadcast to [1, num_cells, 1, 1, 3]
        cell_loc_bcast = insert_dims(self.cell_loc, num_dims=1, axis=0)
        cell_loc_bcast = insert_dims(cell_loc_bcast, num_dims=2, axis=2)
        cell_loc_bcast = cell_loc_bcast.to(self.dtype)

        # Get generator for random numbers
        generator = None if torch.compiler.is_compiling() else self.torch_rng

        # Local azimuth angle within a 120 degree sector. When requested, this
        # directly enforces the inter-UE azimuth separation used as the
        # phi_escan separation in the 38.922-style MU-MIMO setup.
        # [batch_size, num_cells, 3, num_ut_per_sector]
        alpha_local = self._sample_local_sector_angles(
            batch_size=batch_size,
            num_ut_per_sector=num_ut_per_sector,
            min_azimuth_separation_deg=min_ue_azimuth_separation_deg,
            generator=generator,
        )

        # The hexagonal sector distance limit is symmetric over the two
        # 60-degree half-sectors. Convert the local angle in [0, 120 deg] to
        # an equivalent half-sector offset in [-30 deg, 30 deg].
        alpha_half = torch.remainder(alpha_local, PI / 3.0) - PI / 6.0

        # Maximum distance (on the X-Y plane) from BS to a point in the sector,
        # at each half-sector angle.
        r_max = self._isd.to(self.dtype) / (2 * torch.cos(alpha_half))
        r_max = torch.minimum(r_max, torch.sqrt(r_max2))

        ue_elevation_mode_l = str(ue_elevation_mode).lower()
        if ue_elevation_mode_l not in ["none", "fixed_height"]:
            raise ValueError(
                "ue_elevation_mode must be one of {'None', 'fixed_height'}."
            )

        if ue_elevation_mode_l == "fixed_height":
            if ue_elevation_angle_deg is None:
                raise ValueError(
                    "ue_elevation_angle_deg must be provided when "
                    "ue_elevation_mode='fixed_height'."
                )

            if fixed_ue_height is None:
                fixed_ue_height_t = min_ut_height
            elif torch.is_tensor(fixed_ue_height):
                fixed_ue_height_t = fixed_ue_height.detach().clone().to(
                    dtype=self.dtype, device=self.device)
            else:
                fixed_ue_height_t = torch.tensor(
                    fixed_ue_height, dtype=self.dtype, device=self.device)

            theta = float(ue_elevation_angle_deg) * math.pi / 180.0
            if theta <= 0.0 or theta >= math.pi / 2.0:
                raise ValueError(
                    "For ue_elevation_mode='fixed_height', "
                    "ue_elevation_angle_deg must be a down-tilt angle in "
                    "(0, 90) degrees from the horizontal."
                )

            distance_fixed = (cell_height - fixed_ue_height_t) / math.tan(theta)
            if bool(torch.any(distance_fixed <= 0).detach().cpu()):
                raise ValueError(
                    "fixed_ue_height must be lower than the BS height for a "
                    "positive down-tilt elevation angle."
                )

            distance = torch.full_like(r_max, float(distance_fixed.detach().cpu()))

            min_xy_distance = torch.sqrt(torch.clamp(r_min2, min=0.0))
            if bool(torch.any(distance < min_xy_distance).detach().cpu()):
                raise ValueError(
                    "The requested fixed-height elevation angle places UEs "
                    "closer than min_bs_ut_dist allows."
                )
            if bool(torch.any(distance > r_max).detach().cpu()):
                raise ValueError(
                    "The requested fixed-height elevation angle places at "
                    "least one UE outside its sector. Increase "
                    "ue_elevation_angle_deg, increase max_bs_ut_dist, or "
                    "reduce the BS-UE height difference."
                )
        else:
            # To ensure the UT distribution to be uniformly distributed across
            # the sector, sample positions such that their squared distance
            # from the BS is uniformly distributed within (r_min^2, r_max^2).
            distance2 = torch.rand(
                batch_size, self.num_cells, 3, num_ut_per_sector,
                dtype=self.dtype, device=self.device, generator=generator,
            ) * (r_max**2 - r_min2) + r_min2
            distance = torch.sqrt(distance2)

        # Add an offset to angles alpha depending on the sector they belong to
        alpha_offset = torch.tensor(
            [0, 2 * PI / 3, 4 * PI / 3], dtype=self.dtype, device=self.device
        )
        # [1, 1, 3, 1]
        alpha_offset = insert_dims(alpha_offset, num_dims=2, axis=0)
        alpha_offset = insert_dims(alpha_offset, num_dims=1, axis=-1)
        alpha = alpha_local + alpha_offset

        # Compute UT locations on the X-Y plane
        # [batch_size, num_cells, 3, num_ut_per_sector, 2]
        ut_loc = torch.stack(
            [distance * torch.cos(alpha), distance * torch.sin(alpha)], dim=-1
        )
        ut_loc = ut_loc + cell_loc_bcast[..., :2]

        # Add 3rd dimension
        # [batch_size, num_cells, 3, num_ut_per_sector, 3]
        if str(ue_elevation_mode).lower() == "fixed_height":
            ut_loc_z = torch.full(
                (*ut_loc.shape[:-1], 1),
                float(fixed_ue_height_t.detach().cpu()),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            ut_loc_z = torch.rand(
                *ut_loc.shape[:-1], 1,
                dtype=self.dtype, device=self.device, generator=generator,
            ) * (max_ut_height - min_ut_height) + min_ut_height
        ut_loc = torch.cat([ut_loc, ut_loc_z], dim=-1)

        # ------------ #
        # Wraparound   #
        # ------------ #
        # [..., 1, 1, 3]
        ut_loc_bcast = insert_dims(ut_loc, num_dims=2, axis=4)

        # [..., num_cells, num_mirror_grids+1=7, 3]
        mirror_loc_bcast = insert_dims(self._mirror_cell_loc, num_dims=4, axis=0)
        mirror_loc_bcast = mirror_loc_bcast.expand(
            batch_size, self.num_cells, 3, num_ut_per_sector, -1, -1, -1
        )

        # Distance between each point and the 6 mirror + 1 base cells
        # [..., num_cells, num_mirror_grids+1=7]
        ut_mirror_cells_dist = torch.norm(
            ut_loc_bcast - mirror_loc_bcast.to(self.dtype),
            p=2,
            dim=-1,
        )

        # Wraparound distance: min across 6 mirror + 1 base cells
        # [..., num_cells]
        wraparound_dist = ut_mirror_cells_dist.min(dim=-1).values

        # The closest among 6 mirror + 1 base cells for each (UT, base cell)
        # [..., num_cells]
        wraparound_mirror_idx = ut_mirror_cells_dist.argmin(dim=-1)

        # Coordinates of the cell at wraparound distance for each (UT, base cell)
        # [..., num_cells, 3]
        # Gather using advanced indexing
        batch_idx = torch.arange(batch_size, device=self.device)
        cell_idx = torch.arange(self.num_cells, device=self.device)
        sector_idx = torch.arange(3, device=self.device)
        ut_idx = torch.arange(num_ut_per_sector, device=self.device)
        cell2_idx = torch.arange(self.num_cells, device=self.device)

        # Create meshgrid for all indices
        b, c, s, u, c2 = torch.meshgrid(
            batch_idx, cell_idx, sector_idx, ut_idx, cell2_idx, indexing="ij"
        )

        mirror_cell_per_ut_loc = mirror_loc_bcast[
            b, c, s, u, c2, wraparound_mirror_idx, :
        ]

        return ut_loc, mirror_cell_per_ut_loc, wraparound_dist

    def _compute_grid(self) -> None:
        """Compute the spiral grid of hexagonal cells."""
        self._grid = {}
        # Add the central hexagon
        self._grid[0] = Hexagon(
            self._cell_radius.item(),
            coord=self._center_loc.tolist(),
            coord_type=self._center_loc_type,
            precision=self.precision,
            device=self.device,
        )
        # Grid center (axial coordinates)
        grid_center_axial = self._grid[0].coord_axial

        # Spiral over concentric circles of radius ring_radius
        hex_key = 1
        for ring_radius in range(1, self._num_rings + 1):
            hex_curr = Hexagon(
                self._cell_radius.item(),
                coord=(
                    -ring_radius + grid_center_axial[0].item(),
                    ring_radius + grid_center_axial[1].item(),
                ),
                coord_type="axial",
                precision=self.precision,
                device=self.device,
            )
            # Loop over 6 corners
            for ii in range(6):
                # Add 'ring_radius' hexagons in the ii-th direction
                for _ in range(ring_radius):
                    self._grid[hex_key] = hex_curr
                    hex_curr = hex_curr.neighbor(axial_direction_idx=ii)
                    hex_key += 1

    def show(
        self,
        show_mirrors: bool = False,
        show_coord: bool = False,
        show_coord_type: str = "euclid",
        show_sectors: bool = False,
        coord_fontsize: int = 8,
        fig: Optional[plt.Figure] = None,
        color: str = "b",
        label: Optional[str] = "base",
    ) -> plt.Figure:
        """Visualizes the base hexagonal grid and, if specified, the mirror
        grids too.

        Note that a mirror grid is a replica of the base grid, repeated
        around its boundaries to enable wraparound.

        :param show_mirrors: If `True`, then the mirror grids are visualized
        :param show_coord: If `True`, then the hexagon coordinates are
            visualized
        :param show_coord_type: Type of coordinates to be visualized. Must be
            one of {'offset', 'axial', 'euclid'}. Only effective if
            ``show_coord`` is `True`.
        :param show_sectors: If `True`, then the three sectors within each
            hexagon are visualized
        :param coord_fontsize: Coordinate fontsize. Only effective if
            ``show_coord`` is `True`.
        :param fig: Existing figure handle on which the grid is overlayed.
            If `None`, then a new figure is created.
        :param color: Matplotlib line color
        :param label: Label for the cells. If `None`, no label is added.

        :output fig: Figure handle
        """
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.gca()

        if show_mirrors:
            for rr in range(6):
                # Mirror spiral grid
                grid_mirror = HexGrid(
                    cell_radius=self._cell_radius.item(),
                    num_rings=self._num_rings,
                    center_loc=(
                        (self._center_loc[:2] + self._mirror_displacements_offset[rr + 1][:2])
                        .tolist()
                    ),
                    center_loc_type="offset",
                    precision=self.precision,
                    device=self.device,
                )
                # Plot mirror grid
                fig = grid_mirror.show(
                    color="r",
                    fig=fig,
                    show_mirrors=False,
                    show_coord=show_coord,
                    show_coord_type=show_coord_type,
                    label="mirror" if rr == 0 else None,
                )

        for cell_idx, cell in self._grid.items():
            # Visualize hexagon edges
            corners = cell.corners().cpu().numpy()
            ax.plot(
                [corners[-1][0]] + [c[0] for c in corners],
                [corners[-1][1]] + [c[1] for c in corners],
                color=color,
            )

            # Visualize sectors
            if show_sectors:
                center = cell.coord_euclid.cpu().numpy()
                for sector, ii in enumerate([0, 2, 4]):
                    ax.plot(
                        [center[0], corners[ii][0]],
                        [center[1], corners[ii][1]],
                        linestyle="--",
                        color=color,
                    )
                    ax.annotate(
                        str(sector + 1),
                        xy=(
                            (center[0] + corners[ii + 1][0]) / 2,
                            (center[1] + corners[ii + 1][1]) / 2,
                        ),
                        horizontalalignment="center",
                        verticalalignment="center",
                    )

            # Visualize hexagon coordinates
            if show_coord:
                center = cell.coord_euclid.cpu().numpy()
                if show_coord_type == "euclid":
                    coord_val = cell.coord_dict()[show_coord_type].cpu().numpy()
                    text = f"({coord_val[0]:.1f},{coord_val[1]:.1f})"
                else:
                    coord_val = cell.coord_dict()[show_coord_type].cpu().numpy()
                    text = f"({coord_val[0]},{coord_val[1]})"
                ax.annotate(
                    text,
                    xy=(center[0], center[1]),
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=coord_fontsize,
                )
            else:
                center = cell.coord_euclid.cpu().numpy()
                ax.plot(
                    *center,
                    marker=".",
                    color=color,
                    label=(label + " cell")
                    if (label is not None) and (cell_idx == 0)
                    else None,
                )
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        fig.tight_layout()
        return fig


def gen_hexgrid_topology(
    batch_size: int,
    num_rings: int,
    num_ut_per_sector: int,
    scenario: str,
    min_bs_ut_dist: Optional[float] = None,
    max_bs_ut_dist: Optional[float] = None,
    isd: Optional[float] = None,
    bs_height: Optional[float] = None,
    min_ut_height: Optional[float] = None,
    max_ut_height: Optional[float] = None,
    indoor_probability: Optional[float] = None,
    min_ut_velocity: Optional[float] = None,
    max_ut_velocity: Optional[float] = None,
    downtilt_to_sector_center: bool = True,
    los: Optional[bool] = None,
    return_grid: bool = False,
    min_ue_azimuth_separation_deg: Optional[float] = 10.0,
    ue_elevation_mode: str = "None",
    ue_elevation_angle_deg: Optional[float] = None,
    fixed_ue_height: Optional[float] = None,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> Union[
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[bool],
        torch.Tensor,
    ],
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[bool],
        torch.Tensor,
        HexGrid,
    ],
]:
    r"""Generates a batch of topologies with hexagonal cells placed on a spiral
    grid, 3 base stations per cell, and user terminals (UT) dropped uniformly
    at random across the cells.

    UT orientation and velocity are drawn uniformly randomly within the
    specified bounds, whereas the BSs point toward the center of their
    respective sector.

    Parameters provided as `None` are set to valid values according to the
    chosen ``scenario`` (see :cite:p:`TR38901`).

    The returned batch of topologies can be fed into the
    :meth:`~sionna.phy.channel.tr38901.UMa.set_topology` method of the system
    level models, i.e.,
    :class:`~sionna.phy.channel.tr38901.UMi`,
    :class:`~sionna.phy.channel.tr38901.UMa`, and
    :class:`~sionna.phy.channel.tr38901.RMa`.

    :param batch_size: Batch size
    :param num_rings: Number of rings in the hexagonal grid
    :param num_ut_per_sector: Number of UTs to sample per sector and per batch
    :param scenario: System level model scenario. One of "uma", "umi", "rma",
        "uma-calibration", "umi-calibration".
    :param min_bs_ut_dist: Minimum BS-UT distance [m]
    :param max_bs_ut_dist: Maximum BS-UT distance [m]
    :param isd: Inter-site distance [m]
    :param bs_height: BS elevation [m]
    :param min_ut_height: Minimum UT elevation [m]
    :param max_ut_height: Maximum UT elevation [m]
    :param indoor_probability: Probability of a UT to be indoor
    :param min_ut_velocity: Minimum UT velocity [m/s]
    :param max_ut_velocity: Maximum UT velocity [m/s]
    :param downtilt_to_sector_center: If `True`, the BS is mechanically
        downtilted and points towards the sector center. Else, no mechanical
        downtilting is applied.
    :param los: LoS/NLoS states of UTs
    :param return_grid: Determines whether the
        :class:`~sionna.sys.topology.HexGrid` object is returned
    :param min_ue_azimuth_separation_deg: Minimum pairwise azimuth separation
        in degrees between UEs dropped in the same sector. Set to `None` or a
        non-positive value to disable this constraint.
    :param ue_elevation_mode: Elevation constraint mode. "None" keeps the
        original random-height/random-distance behavior. "fixed_height" keeps
        all UEs at ``fixed_ue_height`` and sets the requested down-tilt angle by
        fixing the 2D BS-UE distance.
    :param ue_elevation_angle_deg: Requested common down-tilt angle in degrees
        from the horizontal for ``ue_elevation_mode="fixed_height"``.
    :param fixed_ue_height: UE height [m] used by
        ``ue_elevation_mode="fixed_height"``.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output ut_loc: [batch_size, num_ut, 3], `torch.float`.
        UT locations.
    :output bs_loc: [batch_size, num_cells\*3, 3], `torch.float`.
        BS locations.
    :output ut_orientations: [batch_size, num_ut, 3], `torch.float`.
        UT orientations [radian].
    :output bs_orientations: [batch_size, num_cells\*3, 3], `torch.float`.
        BS orientations [radian]. Oriented toward the center of the sector.
    :output ut_velocities: [batch_size, num_ut, 3], `torch.float`.
        UT velocities [m/s].
    :output in_state: [batch_size, num_ut], `torch.float`.
        Indoor/outdoor state of UTs. `True` means indoor, `False` means
        outdoor.
    :output los: `None`.
        LoS/NLoS states of UTs. This is convenient for directly using the
        function's output as input to
        :meth:`~sionna.phy.channel.SystemLevelScenario.set_topology`, ensuring
        that the LoS/NLoS states adhere to the 3GPP specification (Section
        7.4.2 of TR 38.901).
    :output bs_virtual_loc: [batch_size, num_cells\*3, num_ut, 3], `torch.float`.
        Virtual, i.e., mirror, BS positions for each UT, computed according to
        the wraparound principle.
    :output grid: :class:`~sionna.sys.topology.HexGrid`.
        Hexagonal grid object. Only returned if ``return_grid`` is `True`.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.channel.tr38901 import PanelArray, UMi
        from sionna.sys import gen_hexgrid_topology

        # Create antenna arrays
        bs_array = PanelArray(num_rows_per_panel=4,
                              num_cols_per_panel=4,
                              polarization='dual',
                              polarization_type='VH',
                              antenna_pattern='38.901',
                              carrier_frequency=3.5e9)

        ut_array = PanelArray(num_rows_per_panel=1,
                              num_cols_per_panel=1,
                              polarization='single',
                              polarization_type='V',
                              antenna_pattern='omni',
                              carrier_frequency=3.5e9)

        # Create channel model
        channel_model = UMi(carrier_frequency=3.5e9,
                            o2i_model='low',
                            ut_array=ut_array,
                            bs_array=bs_array,
                            direction='uplink')

        # Generate the topology
        topology = gen_hexgrid_topology(batch_size=100,
                                        num_rings=1,
                                        num_ut_per_sector=3,
                                        scenario='umi')

        # Set the topology
        channel_model.set_topology(*topology)
        channel_model.show_topology()

    .. image:: ../figures/drop_uts_in_hexgrid.png
    """
    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]

    if device is None:
        device = config.device

    # ----------------- #
    # 3GPP parameters   #
    # ----------------- #
    params = set_3gpp_scenario_parameters(
        scenario,
        min_bs_ut_dist,
        isd,
        bs_height,
        min_ut_height,
        max_ut_height,
        indoor_probability,
        min_ut_velocity,
        max_ut_velocity,
        precision=precision,
        device=device,
    )
    (
        min_bs_ut_dist,
        isd,
        bs_height,
        min_ut_height,
        max_ut_height,
        indoor_probability,
        min_ut_velocity,
        max_ut_velocity,
    ) = params

    # Convert max_bs_ut_dist to tensor if provided as a number
    if max_bs_ut_dist is not None and not isinstance(max_bs_ut_dist, torch.Tensor):
        max_bs_ut_dist = torch.tensor(max_bs_ut_dist, dtype=dtype, device=device)

    # ------------ #
    # BS placement #
    # ------------ #
    grid = HexGrid(
        isd=isd.item(),
        cell_height=bs_height.item(),
        num_rings=num_rings,
        precision=precision,
        device=device,
    )
    num_cells = grid.num_cells

    # [num_cells*3, 3]
    bs_loc = grid.cell_loc.repeat_interleave(3, dim=0)
    # [1, num_cells*3, 3]
    bs_loc = insert_dims(bs_loc, num_dims=1, axis=0)
    # [batch_size, num_cells*3, 3]
    bs_loc = bs_loc.expand(batch_size, -1, -1)

    # ---------------- #
    # BS orientation   #
    # ---------------- #
    # Yaw varies according to the sector
    # [num_cells*3]
    bs_yaw = torch.tensor(
        [PI / 3.0, PI, 5.0 * PI / 3.0], dtype=dtype, device=device
    ).repeat(num_cells)
    # [1, num_cells*3]
    bs_yaw = insert_dims(bs_yaw, 1, axis=0)
    # [batch_size, num_cells*3]
    bs_yaw = bs_yaw.expand(batch_size, -1)
    # [batch_size, num_cells*3, 1]
    bs_yaw = insert_dims(bs_yaw, 1, axis=-1)

    # BSs are downtilted towards the sector center
    if downtilt_to_sector_center:
        sector_center = (min_bs_ut_dist + 0.5 * isd) * 0.5
        bs_downtilt = 0.5 * PI - torch.atan(sector_center / bs_height)
    else:
        bs_downtilt = torch.tensor(0.0, dtype=dtype, device=device)

    # [batch_size, num_cells*3, 1]
    bs_pitch = torch.full(
        (batch_size, num_cells * 3, 1), bs_downtilt.item(), dtype=dtype, device=device
    )

    # [batch_size, num_cells*3, 1]
    bs_roll = torch.zeros(batch_size, num_cells * 3, 1, dtype=dtype, device=device)

    # [batch_size, num_cells*3, 3]
    bs_orientations = torch.cat([bs_yaw, bs_pitch, bs_roll], dim=-1)

    # ---------- #
    # Drop UTs   #
    # ---------- #
    # ut_loc: [batch_size, num_cells, num_sectors, num_ut_per_sector, 3]
    ut_loc, bs_virtual_loc, _ = grid(
        batch_size,
        num_ut_per_sector,
        min_bs_ut_dist.item(),
        max_bs_ut_dist=max_bs_ut_dist.item() if max_bs_ut_dist is not None else None,
        min_ut_height=min_ut_height.item(),
        max_ut_height=max_ut_height.item(),
        min_ue_azimuth_separation_deg=min_ue_azimuth_separation_deg,
        ue_elevation_mode=ue_elevation_mode,
        ue_elevation_angle_deg=ue_elevation_angle_deg,
        fixed_ue_height=fixed_ue_height,
    )
    # [batch_size, num_ut, 3]
    ut_loc = flatten_dims(ut_loc, num_dims=3, axis=1)
    num_ut = ut_loc.shape[1]

    # [batch_size, num_ut, num_cells, 3]
    bs_virtual_loc = flatten_dims(bs_virtual_loc, num_dims=3, axis=1)
    # [batch_size, num_ut, num_cells*3, 3]
    bs_virtual_loc = bs_virtual_loc.repeat_interleave(3, dim=2)
    # [batch_size, num_cells*3, num_ut, 3]
    bs_virtual_loc = bs_virtual_loc.permute(0, 2, 1, 3)

    # ---------- #
    # UT state   #
    # ---------- #
    # Draw random UT orientation, velocity and indoor state
    ut_orientations, ut_velocities, in_state = random_ut_properties(
        batch_size,
        num_ut,
        indoor_probability,
        min_ut_velocity,
        max_ut_velocity,
        precision=precision,
        device=device,
    )

    if return_grid:
        return (
            ut_loc,
            bs_loc,
            ut_orientations,
            bs_orientations,
            ut_velocities,
            in_state,
            los,
            bs_virtual_loc,
            grid,
        )
    else:
        return (
            ut_loc,
            bs_loc,
            ut_orientations,
            bs_orientations,
            ut_velocities,
            in_state,
            los,
            bs_virtual_loc,
        )

