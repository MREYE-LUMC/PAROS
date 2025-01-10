"""Paraxial optical modelling of an eye phantom for fundus camera calibrations.

This module contains classes to model an eye phantom used to calibrate a fundus camera.
Lenses are specified using the `Lens` class. The full phantom is modeled with the `EyePhantom` class.
Lens and phantom data is included in `PAROS.data`.
This module can be used in conjunction with `PAROS.fundscale` to calculate the central magnification of the phantom.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
import sympy as sp

from PAROS.fundscale import spherical_interface, uniform_medium

sp.init_printing(use_unicode=True)


class Lens:
    """Ray transfer model of a lens."""

    def __init__(
        self,
        name: str,
        center_thickness: float,
        edge_thickness: float,
        back_curvature: float,
        front_curvature: float,
        refractive_index: float,
        diameter: float,
        power: float,
        lens_type: Literal["biconvex", "convex-concave"],
    ):
        """Initialize a lens model.

        Parameters
        ----------
        name : str
            Name of the lens.
        center_thickness : float
            Central thickness of the lens, in millimeters.
        edge_thickness : float
            Edge thickness of the lens, in millimeters.
        back_curvature : float
            Radius of curvature of the back surface, in millimeters.
        front_curvature : float
            Radius of curvature of the front surface, in millimeters.
        refractive_index : float
            Refractive index of the lens.
        diameter : float
            Diameter of the lens, in millimeters.
        power : float
            Power according to the lens data sheet, in diopters.
        lens_type : str
            Lens type, one of ['biconvex', 'convex-concave'].
        """
        self.name = name
        self.center_thickness = center_thickness
        self.edge_thickness = edge_thickness
        self.back_curvature = back_curvature
        self.front_curvature = front_curvature
        self.refractive_index = refractive_index
        self.lens_type = lens_type
        self.diameter = diameter
        self.power = power

    def __str__(self):
        return f"{self.name}"

    def ray_transfer_matrix(self) -> sp.Matrix:
        """Calculate the ray transfer matrix for the lens.

        Returns
        -------
        sympy.Matrix
            Ray transfer matrix for the lens.
        """
        air_refractive_index = 1.0

        return (
            spherical_interface(self.refractive_index, air_refractive_index, self.back_curvature * 1e-3)
            * uniform_medium(self.center_thickness * 1e-3)
            * spherical_interface(air_refractive_index, self.refractive_index, self.front_curvature * 1e-3)
        )


def make_lens(lens_file: pd.DataFrame, name: str) -> Lens:
    """Create a `Lens` object for the lens `name` from `lens_file`.

    Parameters
    ----------
    lens_file : pandas.DataFrame
        DataFrame with lens specifications.
    name : str
        Name of the lens, must occur in `lens_file`.

    Returns
    -------
    Lens
        `Lens` object.
    """
    lens_info = lens_file[lens_file["name"] == name].iloc[0]

    return Lens(
        name=lens_info["name"],  # 'name' is a property of pandas.Series
        center_thickness=float(lens_info.thickness_centre),
        edge_thickness=float(lens_info.thickness_edge),
        back_curvature=float(lens_info.back_radius),
        front_curvature=float(lens_info.front_radius),
        refractive_index=float(lens_info.refractive_index),
        diameter=float(lens_info.diameter),
        power=float(lens_info.power),
        lens_type=lens_info.lens_type,
    )


class EyePhantom:
    """Eye phantom consisting of two lenses and a grid.

    Specification of an eye phantom consisting of two lenses and a grid.
    The first lens models the cornea, the second lens the eye lens.

    Attributes
    ----------
    name : str
        Name of the phantom.
    iol : Lens
        `Lens` object for the eye lens.
    cornea : Lens
        `Lens` object for the cornea.
    lens_to_lens_distance : float
        Distance between the cornea and the lens.
    lens_to_grid_distance : float
        Distance between the lens and the grid.
    camera_distance : float
        Distance from the cornea front to the camera.
    ray_transfer_matrix : sympy.Matrix
        Ray transfer matrix for the phantom.
    ray_transfer_matrix_to_camera : sympy.Matrix
        Ray transfer matrix between the phantom and the camera.
    """

    def __init__(self, experiment: pd.Series = None, lens_file: pd.DataFrame = None):
        """Initialize an eye phantom from a dataframe.

        Currently only implemented for models of type 'set1'. If the dataframe
        contains another model type, an empty model is created.

        Parameters
        ----------
        experiment : pandas.Series
            Series with experiment parameters.
        lens_file : pandas.DataFrame
            DataFrame with lens specifications.
        """
        if experiment is not None and lens_file is not None:
            self.build_from_dataframe(experiment, lens_file)
        else:
            self.name = "unknown"
            self.iol = None
            self.cornea = None
            self.lens_to_lens_distance = None
            self.lens_to_grid_distance = None
            self.camera_distance = None
            self.ray_transfer_matrix = None
            self.ray_transfer_matrix_to_camera = None

    def __str__(self):
        return f"{self.name}"

    def build_from_dataframe(self, experiment: pd.Series, lens_file: pd.DataFrame) -> None:
        """Initializes an `EyePhantom` instance from a pandas DataFrame.

        Parameters
        ----------
        experiment : pandas.Series
            Series with experiment parameters.
        lens_file : pandas.DataFrame
            DataFrame with lens specifications.
        """
        self.name: str = experiment.get("name")
        self.iol: Lens = make_lens(lens_file, experiment.lens_2)
        self.cornea: Lens = make_lens(lens_file, experiment.lens_1)
        self.lens_to_lens_distance: float = float(experiment.distance_lens1_lens2_mm)
        self.lens_to_grid_distance: float = float(experiment.distance_lens2_grid_mm)
        self.camera_distance: float = experiment.distance_camera_eye_mm

        matrix = (
            self.cornea.ray_transfer_matrix()
            * uniform_medium(self.lens_to_lens_distance * 1e-3)
            * self.iol.ray_transfer_matrix()
            * uniform_medium(self.lens_to_grid_distance * 1e-3)
        )
        self.ray_transfer_matrix: sp.Matrix = matrix
        self.ray_transfer_matrix_to_camera = uniform_medium(self.camera_distance * 1e-3)
