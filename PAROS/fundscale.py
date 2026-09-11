"""Paraxial central scaling of fundus images.

This module implements a paraxial method to calculate the central scaling
of fundus images. The scaling is calculated by defining an eye using the
`Eye` class and a fundus camera using the `Camera` class. The magnification
can then be calculated using `calculate_magnification`.

Throughout this module, the following abbreviations are used for ocular
geometrical parameters:

- R_corF: cornea front radius
- R_corB: cornea back radius
- R_lensF: lens front radius
- R_lensB: lens back radius
- D_cor: cornea thickness
- D_ACD: anterior chamber depth
- D_lens: lens thickness
- D_vitr: vitreous thickness

Unless otherwise specified, sizes and distances are specified in meters.
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from sys import version_info
from typing import Literal, NamedTuple, cast
from warnings import warn

import numpy as np
import sympy as sp

if version_info >= (3, 12):
    from typing import NotRequired, TypedDict, Unpack
else:
    from typing_extensions import NotRequired, TypedDict, Unpack


__all__ = [
    "Camera",
    "Eye",
    "calculate_piol_curvature",
    "calculate_piol_matrix",
    "medium_change",
    "spherical_interface",
    "uniform_medium",
]

NumberOrSymbol = int | float | sp.Symbol
EyeModelType = Literal["Navarro", "VughtIOL"]


def uniform_medium(thickness: float) -> sp.Matrix:
    """Uniform medium without refraction.

    Parameters
    ----------
    thickness : float
        Thickness of the medium.

    Returns
    -------
    sympy.Matrix
        Ray transfer matrix for the uniform medium.
    """
    return sp.Matrix([[1, thickness], [0, 1]])


def medium_change(n_in: float, n_out: float) -> sp.Matrix:
    """Change between media.

    Parameters
    ----------
    n_in : float
        Refractive index of the first medium.
    n_out : float
        Refractive index of the second medium.

    Returns
    -------
    sympy.Matrix
        Ray transfer matrix for the medium change.
    """
    return sp.Matrix([[1, 0], [0, n_in / n_out]])


def spherical_interface(n_in: float, n_out: float, curvature: float) -> sp.Matrix:
    """Spherical interface between two media.

    Parameters
    ----------
    n_in : float
        Refractive index of the first medium.
    n_out : float
        Refractive index of the second medium.
    curvature : float
        Radius of curvature of the interface.

    Returns
    -------
    sympy.Matrix
        Ray transfer matrix for the spherical interface.
    """
    return sp.Matrix([[1, 0], [(n_in - n_out) / (n_out * curvature), (n_in / n_out)]])


def calculate_piol_curvature(
    piol_power: float, thickness: float = 0.2e-3, n_iol: float = 1.47
) -> float:
    """Calculate the radius of curvature for a phakic IOL.

    Calculate the IOL radii for a pIOL with power `piol_power`. The front and back
    curvatures of the IOL are equal.

    The radii are calculated according to ISO 11979-2-2014 A.2.1:

    .. math:: D_{iol} = 2 D_{front} - (t / n_{iol}) D_{front} ^ 2
    .. math:: D_{front} = (n_{iol} - n_{medium}) / R_{front}
    .. math:: n_{medium} = 1.336

    Parameters
    ----------
    piol_power : float
        Power of the pIOL, in diopters.
    thickness : float
        Thickness of the pIOL, in meters.
    n_iol : float
        Refractive index of the pIOL.

    Returns
    -------
    float
        Radius of curvature of the front and back pIOL surface.

    Notes
    -----
    The calculation of the pIOL curvature seems to be in congruence with ANSI standard
    and signs are correct.
    """
    n_medium = 1.336

    A = -thickness / n_iol
    B = 2
    C = -piol_power

    front_power = (-B + math.sqrt(B * B - 4 * A * C)) / (2 * A)

    return (n_iol - n_medium) / front_power


def calculate_piol_matrix(
    piol_power: NumberOrSymbol,
    thickness: float,
    n_iol: float,
    distance_iol_lens: float,
    n_aq: float,
) -> sp.Matrix:
    """Calculate the ray transfer matrix for a phakic IOL.

    Parameters
    ----------
    piol_power : float
        Power of the pIOL, in diopters.
    thickness : float
        Thickness of the pIOL, in meters.
    n_iol : float
        Refractive index of the pIOL.
    distance_iol_lens : float
        Distance between the pIOL and the crystalline lens, in m.
    n_aq : float
        Refractive index of the aqueous humor.

    Returns
    -------
    sympy.Matrix
        Ray transfer matrix for the pIOL.
    """
    piol_curvature = calculate_piol_curvature(piol_power, thickness, n_iol)

    piol_matrix = (
        spherical_interface(n_iol, n_aq, -1 * piol_curvature)
        * uniform_medium(thickness)
        * spherical_interface(n_aq, n_iol, piol_curvature)
    )
    distance_matrix = uniform_medium(distance_iol_lens)

    return piol_matrix * distance_matrix


class EyeGeometry(TypedDict, total=True):
    """Eye geometry parameters.

    All curvatures and distances are specified in meters and the spherical equivalent in diopters.
    PAROS uses an inverted eye model, so for a normal eye the radii of curvature for the cornea and
    lens front surfaces are negative.

    Attributes
    ----------
    R_corF : float
        Cornea front radius.
    R_corB : float
        Cornea back radius.
    R_lensF : float
        Lens front radius.
    R_lensB : float
        Lens back radius.
    D_cor : float
        Cornea thickness.
    D_ACD : float
        Anterior chamber depth.
    D_lens : float
        Lens thickness.
    D_vitr : float
        Vitreous thickness.
    SE : float
        Spherical equivalent of refraction of the eye model.
    """

    R_corF: float
    R_corB: float
    R_lensF: float
    R_lensB: float
    D_cor: float
    D_ACD: float
    D_lens: float
    D_vitr: float
    SE: NotRequired[float]


class _PartialEyeGeometry(EyeGeometry, total=False):
    """Eye geometry parameters.

    This class can be used to define a partial eye geometry, in which only a subset
    of the parameters are specified.

    See Also
    --------
    EyeGeometry : Full eye geometry parameters.
    """

    R_corF: float
    R_corB: float
    R_lensF: float
    R_lensB: float
    D_cor: float
    D_ACD: float
    D_lens: float
    D_vitr: float


class PhakicIOL(NamedTuple):
    """Parameters of a phakic IOL.

    Attributes
    ----------
    power : NumberOrSymbol
        Power in diopters.
    thickness : NumberOrSymbol
        Thickness in meters.
    refractive_index : NumberOrSymbol
        Refractive index of the pIOL material.
    lens_distance : NumberOrSymbol
        Distance to the crystalline lens in meters.
    """

    power: NumberOrSymbol
    thickness: NumberOrSymbol
    refractive_index: NumberOrSymbol
    lens_distance: NumberOrSymbol


class RefractiveIndices(TypedDict):
    """Refractive indices of the eye media.

    Attributes
    ----------
    cor : float
        Refractive index of the cornea.
    aq : float
        Refractive index of the aqueous humor.
    lens : float
        Refractive index of the crystalline lens.
    vit : float
        Refractive index of the vitreous humor.
    """

    cor: float
    aq: float
    lens: float
    vit: float


class _PartialRefractiveIndices(RefractiveIndices, total=False):
    """Refractive indices of the eye media.

    This class can be used to define a partial set of refractive indices, in which only a
    subset of the indices are specified.

    See Also
    --------
    RefractiveIndices : Full set of refractive indices.
    """

    cor: float
    aq: float
    lens: float
    vit: float


_DEFAULT_GEOMETRIES: dict[EyeModelType, EyeGeometry] = {
    "Navarro": EyeGeometry(
        R_corF=-7.72e-3,
        R_corB=-6.50e-3,
        R_lensF=-10.20e-3,
        R_lensB=+6.00e-3,
        D_cor=0.55e-3,
        D_ACD=3.05e-3,
        D_lens=4.00e-3,
        D_vitr=16.3203e-3,
    ),
    "VughtIOL": EyeGeometry(
        R_corF=-7.72e-3,
        R_corB=-6.50e-3,
        R_lensF=-8.16e-3,
        R_lensB=+11.18e-3,
        D_cor=0.55e-3,
        D_ACD=3.05e-3,
        D_lens=0.6896e-3,
        D_vitr=19.3203e-3,
    ),
}

_DEFAULT_REFRACTIVE_INDICES: dict[EyeModelType, RefractiveIndices] = {
    "Navarro": RefractiveIndices(
        cor=1.3777,
        aq=1.3391,
        lens=1.4222,
        vit=1.3377,
    ),
    "VughtIOL": RefractiveIndices(
        cor=1.3777,
        aq=1.3391,
        lens=1.47,
        vit=1.3377,
    ),
}


class Eye:
    """Symbolic representation of an eye model."""

    def __init__(
        self,
        name: str = "testEye",
        geometry: _PartialEyeGeometry | EyeGeometry | None = None,
        model_type: EyeModelType = "Navarro",
        NType: EyeModelType = "Navarro",
        refractive_indices: _PartialRefractiveIndices | None = None,
        refraction: float | None = None,
        pIOL: PhakicIOL | None = None,
    ) -> None:
        """Initialize an eye model.

        This class works with a symbolic representation of the eye model, in which the
        numeric values of the model parameters as defined in `geometry` are substituted
        when needed. The eye's geometry can be passed to this function as a dictionary.
        If this is a partial dictionary, the geometrical parameters corresponding to
        `model_type` are updated with these values.

        Parameters
        ----------
        name : str
            Name of the eye model.
        geometry : _PartialEyeGeometry | None
            Dictionary with numerical geometric parameters of the eye. If `None`, the
            geometry is based on `model_type`. Can be a partial dictionary, in which
            case missing values are based on the eye model specified by `model_type`.
        model_type : EyeModelType
            Standard eye model on which the eye geometry is based. One of
            ["Navarro", "VughtIOL"].
        NType : EyeModelType
            Standard eye model from which the refractive indices are used. One of
            ["Navarro", "VughtIOL"].
        refraction : float
            Spherical equivalent of refraction of the eye model. The lens back curvature
            can be adjusted to obtain this refraction with `Eye.adjust_lens_back`.
        refractive_indices : _PartialRefractiveIndices | None
            Dictionary with numerical values of the refractive indices of the eye media.
            If `None`, the refractive indices are based on `NType`. If specified, the
            refractive indices corresponding to `NType` are updated with the values in this dictionary.
        pIOL : PhakicIOL
            Properties of the pIOL, if present. Tuple of power, thickness, refractive
            index, distance to lens.
        """
        self.name = name
        self.spherical_equivalent = refraction
        self.is_pseudophakic = model_type == "VughtIOL"
        self.pIOL = pIOL

        self.geometry = self._build_geometry_dictionary(model_type, geometry)

        self.refractive_indices: RefractiveIndices

        if NType in _DEFAULT_REFRACTIVE_INDICES:
            self.refractive_indices = _DEFAULT_REFRACTIVE_INDICES[NType].copy()
        else:
            raise ValueError(f"Model type {NType} is undefined.")

        if refractive_indices is not None:
            self.refractive_indices.update(refractive_indices)

        self.R_corF, self.R_corB, self.R_lensF, self.R_lensB = sp.symbols(
            "R_corF R_corB R_lensF R_lensB"
        )
        self.D_cor, self.D_ACD, self.D_lens, self.D_vitr = sp.symbols(
            "D_cor D_ACD D_lens D_vitr"
        )

        self._update_ray_transfer_matrix()

    @property
    def axial_length(self) -> float:
        return (
            self.geometry["D_cor"]
            + self.geometry["D_ACD"]
            + self.geometry["D_lens"]
            + self.geometry["D_vitr"]
        )

    def __str__(self) -> str:
        return f"{self.name}"

    @staticmethod
    def _build_geometry_dictionary(
        model_type: EyeModelType,
        partial_geometry: _PartialEyeGeometry | EyeGeometry | None = None,
    ) -> EyeGeometry:
        if model_type not in _DEFAULT_GEOMETRIES:
            raise ValueError(f"Model type {model_type} is undefined.")

        geometry = _DEFAULT_GEOMETRIES[model_type].copy()

        if partial_geometry is not None:
            # Estimate the cornea back curvature if it is unspecified and the cornea
            # front curvature is specified
            if "R_corF" in partial_geometry and "R_corB" not in partial_geometry:
                partial_geometry["R_corB"] = 0.81 * partial_geometry["R_corF"]

            # Update the geometry with the parameters specified in partial_geometry
            geometry.update(partial_geometry)

        return geometry

    def _matrix_cornea(self) -> sp.Matrix:
        return (
            spherical_interface(self.refractive_indices["cor"], 1.0, self.R_corF)
            * uniform_medium(self.D_cor)
            * spherical_interface(
                self.refractive_indices["aq"],
                self.refractive_indices["cor"],
                self.R_corB,
            )
        )

    def _matrix_lens(self) -> sp.Matrix:
        return (
            spherical_interface(
                self.refractive_indices["lens"],
                self.refractive_indices["aq"],
                self.R_lensF,
            )
            * uniform_medium(self.D_lens)
            * spherical_interface(
                self.refractive_indices["vit"],
                self.refractive_indices["lens"],
                self.R_lensB,
            )
        )

    def _matrix_anterior_segment(self) -> sp.Matrix:
        """Compute the ray transfer matrix for the anterior segment of the eye (cornea to pupil)."""
        return self._matrix_cornea() * uniform_medium(self.D_ACD)

    def _matrix_posterior_segment(self) -> sp.Matrix:
        """Compute the ray transfer matrix for the posterior segment of the eye (pupil to retina)."""
        return self._matrix_lens() * uniform_medium(self.D_vitr)

    def calculate_ray_transfer_matrix(self) -> sp.Matrix:
        """Calculate the eye's ray transfer matrix.

        Returns
        -------
        sp.Matrix
            Ray transfer matrix of the eye.
        """
        # pIOL
        if self.pIOL:  # (Diol,tiol,Niol,d_iollens)
            phakic_iol = calculate_piol_matrix(
                self.pIOL[0],
                self.pIOL[1],
                self.pIOL[2],
                self.pIOL[3],
                self.refractive_indices["aq"],
            )
        else:  # Identity matrix
            phakic_iol = sp.Matrix([[1, 0], [0, 1]])

        # Retina to cornea
        return (
            self._matrix_anterior_segment()
            * phakic_iol
            * self._matrix_posterior_segment()
        )

    def entrance_pupil_position(self) -> float:
        """Compute the position of the entrance pupil relative to the cornea.

        Note: since this is a reversed eye, the calculation is similar to the exit pupil position,
        but using the anterior segment matrix instead of the posterior segment matrix.

        The entrance pupil is calculated from the anterior ray transfer matrix as L = -B / D.
        """
        m_anterior = self._matrix_anterior_segment().evalf(subs=self.geometry)

        return float(-m_anterior[0, 1] / m_anterior[1, 1])

    def nodal_points(self) -> tuple[float, float]:
        """Compute the positions of the nodal points relative to the cornea.

        The nodal points are calculated from the ray transfer matrix as:
        N1 = (A - n_vit) / C
        N2 = AL - (D - 1) / C

        where A, C, D are elements of the ray transfer matrix, n_vit is the refractive index of the vitreous,
        and AL is the axial length.
        """
        matrix = self.evaluate_matrix()

        n1 = float(matrix[0, 0]) - self.refractive_indices["vit"] / float(matrix[1, 0])
        n2 = self.axial_length - float(matrix[1, 1] - 1) / float(matrix[1, 0])

        return n1, n2

    def update_geometry(self, **geometry: Unpack[_PartialEyeGeometry]) -> None:
        """Update the eye geometry with new values.

        Parameters
        ----------
        geometry : _PartialEyeGeometry
            Geometry parameters to update as keyword arguments.
        """
        self.geometry.update(geometry)
        self._update_ray_transfer_matrix()

    def update_refractive_indices(
        self, **refractive_indices: Unpack[_PartialRefractiveIndices]
    ) -> None:
        """Update the eye's refractive indices with new values.

        Parameters
        ----------
        refractive_indices : _PartialRefractiveIndices
            Refractive indices to update as keyword arguments.
        """
        self.refractive_indices.update(refractive_indices)
        self._update_ray_transfer_matrix()

    def _update_ray_transfer_matrix(self) -> None:
        """Update the eye's ray transfer matrix."""
        self.ray_transfer_matrix = self.calculate_ray_transfer_matrix()

    def adjust_lens_back(
        self, target_refraction: float, *, update_model: bool = False
    ) -> tuple[float, float]:
        """Fit the lens back curvature to the eye's refraction.

        A corrective lens (glasses) for an eye with `target_refraction` is place in
        front of the eye. The lens back surface is then solved for a focused image on
        the retina. A vertex distance of 1.4 cm between the glasses and the eye is
        assumed.

        Parameters
        ----------
        target_refraction : float
            Desired spherical of refraction of the eye in diopters.
        update_model : bool
            If `True`, the eye's lens back curvature is set to the calculated value.

        Returns
        -------
        lens_back_curvature : float
            Radius of curvature of the lens back surface.
        r_glasses : float
            Radius of curvature of the glasses.
        """
        n_glasses = 1.5
        r_glasses = 2.0 * (n_glasses - 1.0) / (target_refraction + 0.0000000001)

        m_glasses = spherical_interface(n_glasses, 1, -r_glasses) * spherical_interface(
            1, n_glasses, r_glasses
        )

        # Reversed eye with glasses
        # Assuming average vertex distance of 1.4cm
        matrix_glasses_eye = (
            m_glasses * uniform_medium(0.014) * self.ray_transfer_matrix
        )
        temp_geometry = copy.deepcopy(self.geometry)

        temp_geometry.pop("R_lensB")

        solve_output = sp.solveset(
            matrix_glasses_eye.evalf(subs=temp_geometry)[1, 1], self.R_lensB
        )  # parallel rays from glasses

        if len(list(solve_output)) != 1:
            warn(f"Multiple solutions found. {target_refraction=}, {solve_output=}")

        lens_back_curvature = next(iter(solve_output))

        if update_model:
            self.geometry["R_lensB"] = lens_back_curvature

        return lens_back_curvature, r_glasses

    def calculate_refraction(self) -> float:
        """Calculate the refraction of the eye.

        The refraction is calculated as the vergence of a central retinal object at a vertex distance
        of 1.4 cm in front of the cornea. This is equivalent to the power of a corrective thin lens
        placed 1.4 cm in front of the cornea that focuses light on the retina.

        Returns
        -------
        float
            The power of the corrective lens, in diopters.
        """
        matrix = self.evaluate_matrix()

        B = float(matrix[0, 1])
        D = float(matrix[1, 1])

        # Calculate vergence at cornea
        vergence = D / B

        # Calculate refraction 1.4 cm in front of the cornea (vertex distance)
        return vergence / (1 + 0.014 * vergence)

    def evaluate_matrix(self) -> sp.Matrix:
        """Evaluate the eye's ray transfer matrix using its geometrical parameters.

        Returns
        -------
        sympy.Matrix
            Ray transfer matrix for the eye model.
        """
        return self.ray_transfer_matrix.evalf(subs=self.geometry)


class BaseCamera(ABC):
    @abstractmethod
    def calculate_magnification(self, eye: Eye, *args, **kwargs) -> float:
        """Calculate the magnification of the eye-camera system.

        Parameters
        ----------
        eye : Eye
            Eye model.
        *args : tuple
            Additional camera-specific positional arguments.
        **kwargs : dict
            Additional camera-specific keyword arguments.

        Returns
        -------
        float
            Magnification of the eye-camera system, in pixels per millimeter.
        """


_MISSING = cast("float", object())


class Camera(BaseCamera):
    def __init__(
        self,
        F_cond: NumberOrSymbol | None = None,
        a1: NumberOrSymbol | None = None,
        pixel_density: float = _MISSING,
        camera_type: Literal["default"] = "default",
    ) -> None:
        """Create a new camera model.

        Parameters
        ----------
        F_cond : NumberOrSymbol
            Focal length of the condenser lens, in meters.
        a1 : NumberOrSymbol
            First order calibration term.
        camera_type : Literal["default"]
            Type of camera. Currently only "default" is implemented.
        pixel_density : float
            Pixel density of the camera sensor, in pixels per millimeter.
            Defaults to 100 pixels/mm.
        """
        if pixel_density is _MISSING:
            raise ValueError("Pixel density must be specified.")
        if pixel_density <= 0:
            raise ValueError("Pixel density must be a positive float.")

        self.camera_type = "lensTaylor"
        self.F_cond = sp.Symbol("F_cond", real=True)
        self.d_CCD = sp.symbols("d_CCD")
        self.R_foc = sp.Symbol("R_foc", real=True)  # in m
        self.a1 = sp.Symbol("a1", real=True)
        self.pixel_density = pixel_density  # pixels per mm

        if camera_type == "default":
            self.d_CCD = self.F_cond
        else:
            raise NotImplementedError("Custom CCD distances are not implemented.")

        self.n_glas = 1.5
        self.a1_value = a1

        self.condenser_lens = spherical_interface(
            self.n_glas, 1.0, -self.F_cond
        ) * spherical_interface(1.0, self.n_glas, self.F_cond)
        self.focus_lens = spherical_interface(
            self.n_glas, 1.0, -self.R_foc
        ) * spherical_interface(1.0, self.n_glas, self.R_foc)
        self.correction_term = sp.Matrix([
            [1 + self.a1 / self.R_foc, 0],
            [0, 1.0 / (1 + self.a1 / self.R_foc)],
        ])
        self.ray_transfer_matrix = (
            self.correction_term
            * uniform_medium(self.d_CCD)
            * self.focus_lens
            * self.condenser_lens
        )

        if F_cond and (a1 is not None):
            self.ray_transfer_matrix = self.ray_transfer_matrix.evalf(
                subs={self.F_cond: F_cond, self.a1: a1}
            )

    @property
    def M_camera_alg(self):
        return self.ray_transfer_matrix

    def calculate_focus_lens_radius(
        self, eye_matrix: sp.Matrix, distance_eye_camera: float = 0.05
    ) -> float:
        """Calculate the radius of curvature of the focus lens.

        Solves for the radius of curvature of the camera's focus lens, so that the image
        of an object on the retina is in focus.

        Parameters
        ----------
        eye_matrix : sympy.Matrix
            Ray transfer matrix of the eye model.
        distance_eye_camera : float
            Distance between the eye and camera in meters, measured from the cornea
            front to the camera lens front. Defaults to 0.05 m.

        Returns
        -------
        float
            Radius of curvature of the focus lens, in meters.
        """
        system_matrix = (
            self.ray_transfer_matrix * uniform_medium(distance_eye_camera) * eye_matrix
        )

        # For contact cameras such as the Panoret fundus camera, the refractive index (of air) needs to be changed to
        # that of the medium used between the eye and camera. This can be done by changing the use of uniform_medium()
        # to medium_change().
        # The camera should correct for the patients refraction, so the image should be
        # focused, i.e. B = 0
        B = system_matrix[0, 1] + 0.0000000001

        if self.a1 in B.free_symbols:
            possible_curvatures = list(sp.solve(B.evalf(subs={self.a1: 0}), self.R_foc))
        else:
            possible_curvatures = list(sp.solve(B, self.R_foc))

        return np.max(np.abs(possible_curvatures))

    def focused_system_matrix(
        self,
        eye_matrix: sp.Matrix,
        distance_eye_camera: float = 0.05,
        *,
        return_focus_lens_curvature: bool = False,
    ) -> sp.Matrix | tuple[sp.Matrix, float]:
        system_matrix = (
            self.ray_transfer_matrix * uniform_medium(distance_eye_camera) * eye_matrix
        )

        # For contact cameras such as the Panoret fundus camera, the refractive index (of air) needs to be changed to
        # that of the medium used between the eye and camera. This can be done by changing the use of uniform_medium()
        # to medium_change().
        focus_lens_curvature = self.calculate_focus_lens_radius(
            eye_matrix, distance_eye_camera
        )
        if return_focus_lens_curvature:
            return (
                system_matrix.evalf(subs={self.R_foc: focus_lens_curvature}),
                focus_lens_curvature,
            )

        return system_matrix.evalf(subs={self.R_foc: focus_lens_curvature})

    def get_size_on_ccd(self, size_pixels: float) -> float:
        """Convert an image size in pixels to its physical size on the CCD sensor in meters.

        Parameters
        ----------
        size_pixels : float
            Image size on the CCD sensor in pixels.

        Returns
        -------
        float
            Image size on the CCD sensor in meters.
        """
        return size_pixels / (self.pixel_density * 1000)  # convert mm to m

    # Maximum allowed difference between the calculated and specified refraction of the eye model
    _MAXIMUM_REFRACTION_DEVIATION = 0.05

    # Maximum allowed value of the B-element in a ray transfer matrix for a focused system
    _MAXIMUM_FOCUS_DEVIATION = 0.0001

    def calculate_magnification(
        self,
        eye: Eye,
        distance_eye_camera: float = 0.05,
        focus_lens_radius: float | None = None,
        *,
        suppress_warnings: bool = False,
    ) -> float:
        """Calculate the total magnification of the eye - camera system.

        A structure of 1 mm  on the central retina has a size of `magnification` pixels
        on the camera sensor.

        Parameters
        ----------
        eye : Eye
            Eye model.
        distance_eye_camera : float
            Distance between eye and camera in meters, measured from the cornea front to
            the camera lens front.
        focus_lens_radius : float
            Optional radius of curvature of the focus lens. If not specified, the focus lens
            curvature is determined using `Camera.calculate_focus_lens_radius`.
        suppress_warnings : bool
            If `True`, no warning is issued if the calculated glasses power differs
            significantly from `eye.spherical_equivalent`.

        Returns
        -------
        magnification : float
            Central magnification of the eye - camera system, in pixels per millimeter.

        Warns
        --------
        If the calculated glasses power differs significantly from the clinical refraction of the eye model.
        """
        glasses_power = eye.calculate_refraction()

        if (
            eye.spherical_equivalent is not None
            and not suppress_warnings
            and abs(glasses_power - eye.spherical_equivalent)
            > self._MAXIMUM_REFRACTION_DEVIATION
        ):
            warn(
                f"model refraction {glasses_power:.2f} not matching clinical refraction"
                f" {eye.spherical_equivalent:.2f} for {eye.name}"
            )

        # For contact cameras such as the Panoret fundus camera, this refractive index (of air) needs to be changed to that
        # of the medium between the eye and camera. This can be done by changing the use of uniform_medium() to
        # medium_change().
        system_matrix = (
            self.ray_transfer_matrix
            * uniform_medium(distance_eye_camera)
            * eye.evaluate_matrix()
        )

        if focus_lens_radius is None:
            # system is in focus so B=0
            solutions = list(sp.solve(system_matrix[0, 1] + 0.00001, self.R_foc))
            focus_lens_radius = solutions[
                np.argmax(np.abs(np.array(solutions) + self.a1_value))
            ]

        focused_system_matrix = system_matrix.evalf(
            subs=({self.R_foc: focus_lens_radius})
        )

        magnification: float = focused_system_matrix[0, 0] * self.pixel_density

        if abs(focused_system_matrix[0, 1]) > self._MAXIMUM_FOCUS_DEVIATION:
            warn(f"focused_system_matrix not in focus for patient {eye.name}")

        return float(magnification)


class TelecentricCamera(BaseCamera):
    def __init__(self, k: float) -> None:
        """Create a new telecentric camera model.

        The telecentric camera is characterized by a constant `k`, which is the ratio between
        the image size on the camera sensor and ray angles.

        Parameters
        ----------
        k : float
            Angle scaling factor of the camera, in pixels per radian.

        Raises
        ------
        ValueError
            If `k` is not positive.
        """
        if k <= 0:
            raise ValueError("Magnification factor k must be positive.")

        self.k = k

    def calculate_magnification(self, eye: Eye) -> float:
        """Calculate the magnification of the telecentric camera system for a given eye.

        The magnification is calculated in units of pixels per millimeter, i.e. a structure of 1 mm
        on the retina has a size of `magnification` pixels on the camera sensor.

        Parameters
        ----------
        eye : Eye
            Eye model.

        Returns
        -------
        float
            Magnification of the eye-camera system, in pixels per millimeter.
        """
        eye_matrix = eye.evaluate_matrix()

        B = eye_matrix[0, 1]
        D = eye_matrix[1, 1]

        numerator = self.k * eye.refractive_indices["vit"]
        denominator = 1e3 * (B + eye.entrance_pupil_position() * D)

        return numerator / denominator


class FocusDependentCamera(BaseCamera):
    def __init__(self, k0: float, alpha: float) -> None:
        """Create a new focus-dependent camera model.

        The focus-dependent camera is characterized by a constant `k0`, which is the ratio between
        the image size on the camera sensor and ray angles for emmetropic eyes, and a slope `alpha`
        that describes how the magnification changes with the eye's refraction.

        Parameters
        ----------
        k0 : float
            Angle scaling factor of the camera for emmetropic eyes, in pixels per radian.
        alpha : float
            Slope describing how the magnification changes with the eye's refraction, in pixels per radian per diopter.

        Raises
        ------
        ValueError
            If `k0` is not positive.
        """
        if k0 <= 0:
            raise ValueError("Magnification factor k0 must be positive.")

        self.k0 = k0
        self.alpha = alpha

    def k(self, refraction: float) -> float:
        """Calculate the angle scaling factor k for a given eye refraction.

        Parameters
        ----------
        refraction : float
            Spherical equivalent of refraction of the eye model, in diopters.

        Returns
        -------
        float
            Angle scaling factor k for the given eye refraction, in pixels per radian.
        """
        return self.k0 + self.alpha * refraction

    def calculate_magnification(self, eye: Eye) -> float:
        """Calculate the magnification of the focus-dependent camera system for a given eye.

        The magnification is calculated in units of pixels per millimeter, i.e. a structure of 1 mm
        on the retina has a size of `magnification` pixels on the camera sensor.

        Parameters
        ----------
        eye : Eye
            Eye model.

        Returns
        -------
        float
            Magnification of the eye-camera system, in pixels per millimeter.
        """
        eye_matrix = eye.evaluate_matrix()

        B = eye_matrix[0, 1]
        D = eye_matrix[1, 1]

        numerator = self.k(eye.calculate_refraction()) * eye.refractive_indices["vit"]
        denominator = 1e3 * (B + eye.entrance_pupil_position() * D)

        return numerator / denominator
