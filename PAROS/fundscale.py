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
from typing import Literal, NamedTuple, Union
from warnings import warn

import numpy as np
import sympy as sp

sp.init_printing(use_unicode=True)

NumberOrSymbol = Union[int, float, sp.Symbol]
EyeModelType = Literal["Navarro", "VughtIOL"]


def _round_expr(expr: sp.Expr, num_digits: int = 3) -> sp.Expr:
    return expr.xreplace({n: round(n, num_digits) for n in expr.atoms(sp.Number)})


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


def calculate_piol_curvature(piol_power: float, thickness: float = 0.2e-3, n_iol: float = 1.47) -> float:
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

    A = -thickness / n_iol  # noqa: N806
    B = 2  # noqa: N806
    C = -piol_power  # noqa: N806

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


class Eye:
    """Symbolic representation of an eye model."""

    def __init__(
        self,
        name: str = "testEye",
        geometry: dict[str, float] | None = None,
        model_type: EyeModelType = "Navarro",
        NType: EyeModelType = "Navarro",  # noqa: N803
        refraction: float | None = None,
        pIOL: PhakicIOL | None = None,  # noqa: N803
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
        geometry : dict[str, float]
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
        pIOL : PhakicIOL
            Properties of the pIOL, if present. Tuple of power, thickness, refractive
            index, distance to lens.
        """
        self.name = name
        self.spherical_equivalent = refraction
        self.is_pseudophakic = model_type == "VughtIOL"
        self.axial_length = None
        self.pIOL = pIOL

        self.geometry = self._build_geometry_dictionary(model_type, geometry)

        if NType == "Navarro":
            self.refractive_indices = {
                "cor": 1.3777,
                "aq": 1.3391,
                "lens": 1.4222,
                "vit": 1.3377,
            }
        elif NType == "VughtIOL":
            self.refractive_indices = {
                "cor": 1.3777,
                "aq": 1.3391,
                "lens": 1.47,
                "vit": 1.3377,
            }
        else:
            raise ValueError(f"Model type {NType} is undefined.")

        self.R_corF, self.R_corB, self.R_lensF, self.R_lensB = sp.symbols("R_corF R_corB R_lensF R_lensB")
        self.D_cor, self.D_ACD, self.D_lens, self.D_vitr = sp.symbols("D_cor D_ACD D_lens D_vitr")

        self.axial_length = (
            self.geometry["D_cor"] + self.geometry["D_ACD"] + self.geometry["D_lens"] + self.geometry["D_vitr"]
        )
        self.ray_transfer_matrix = self.calculate_ray_transfer_matrix()

    def __str__(self) -> str:
        return f"{self.name}"

    @staticmethod
    def _build_geometry_dictionary(model_type: EyeModelType, partial_geometry: dict[str, float] | None = None):
        if model_type == "Navarro":
            geometry = {
                "R_corF": -7.72e-3,
                "R_corB": -6.50e-3,
                "R_lensF": -10.20e-3,
                "R_lensB": +6.00e-3,
                "D_cor": 0.55e-3,
                "D_ACD": 3.05e-3,
                "D_lens": 4.00e-3,
                "D_vitr": 16.3203e-3,
            }
        elif model_type == "VughtIOL":
            geometry = {
                "R_corF": -7.72e-3,
                "R_corB": -6.50e-3,
                "R_lensF": -8.16e-3,
                "R_lensB": +11.18e-3,
                "D_cor": 0.55e-3,
                "D_ACD": 3.05e-3,
                "D_lens": 0.6896e-3,
                "D_vitr": 19.3203e-3,
            }
        else:
            raise ValueError(f"Model type {model_type} is undefined.")

        if partial_geometry is not None:
            # Estimate the cornea back curvature if it is unspecified and the cornea
            # front curvature is specified
            if "R_corF" in partial_geometry and "R_corB" not in partial_geometry:
                partial_geometry["R_corB"] = 0.81 * partial_geometry["R_corF"]

            # Update the geometry with the parameters specified in partial_geometry
            geometry.update(partial_geometry)

        return geometry

    def calculate_ray_transfer_matrix(self) -> sp.Matrix:
        """Calculate the eye's ray transfer matrix.

        Returns
        -------
        sp.Matrix
            Ray transfer matrix of the eye.
        """

        # Cornea
        cornea = (
            spherical_interface(self.refractive_indices["cor"], 1.0, self.R_corF)
            * uniform_medium(self.D_cor)
            * spherical_interface(
                self.refractive_indices["aq"],
                self.refractive_indices["cor"],
                self.R_corB,
            )
        )

        # Lens
        lens = (
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
        return cornea * uniform_medium(self.D_ACD) * phakic_iol * lens * uniform_medium(self.D_vitr)

    def adjust_lens_back(self, target_refraction: float, *, update_model: bool = False) -> tuple[float, float]:
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

        m_glasses = spherical_interface(n_glasses, 1, -r_glasses) * spherical_interface(1, n_glasses, r_glasses)

        # Reversed eye with glasses
        # Assuming average vertex distance of 1.4cm
        matrix_glasses_eye = m_glasses * uniform_medium(0.014) * self.ray_transfer_matrix
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

    def calculate_refraction(self, *, display=True) -> tuple[float, float]:
        """Calculate the refraction of the eye.

        A corrective lens (glasses) for an eye with `target_refraction` is place in
        front of the eye. Its radius of curvature is then solved for a focused image on
        the retina. A vertex distance of 1.4 cm between the glasses and the eye is
        assumed.

        Parameters
        ----------
        display : bool
            If `True`, the results are printed.

        Returns
        -------
        glasses_curvature : float
            Radius of curvature of the corrective lens, in meters.
        glasses_power : float
            Power of the corrective lens, in diopters.

        Warnings
        --------
        When multiple solutions for the curvature are found, a warning is displayed.
        """
        n_glasses = 1.5
        R = sp.symbols("R")  # noqa: N806
        glasses = spherical_interface(n_glasses, 1, -R * 10**-3) * spherical_interface(1, n_glasses, R * 10**-3)
        eye = self.evaluate_matrix()

        glasses_curvature_solutions = sp.solveset((glasses * uniform_medium(0.014) * eye)[1, 1], R)
        if len(list(glasses_curvature_solutions)) != 1:
            warn(f"Multiple solutions found: {glasses_curvature_solutions}")

        glasses_curvature = next(iter(glasses_curvature_solutions)) * 1e-3
        glasses_power = 2.0 * (n_glasses - 1.0) / (glasses_curvature)
        if display:
            print(f"{glasses_curvature=:.2f} m, {glasses_power=:.2f} D")  # noqa: T201

        return glasses_curvature, glasses_power

    def evaluate_matrix(self) -> sp.Matrix:
        """Evaluate the eye's ray transfer matrix using its geometrical parameters.

        Returns
        -------
        sympy.Matrix
            Ray transfer matrix for the eye model.
        """
        return self.ray_transfer_matrix.evalf(subs=self.geometry)

    def calculate_camera_magnification(
        self, camera: Camera, distance_eye_camera: float = 0.05
    ) -> tuple[float, sp.Matrix, float]:
        """Calculate the total magnification of the eye - camera system.

        A structure on the central retina is depicted `magnification` times larger
        on the camera sensor.

        Parameters
        ----------
        camera : Camera
            Camera model.
        distance_eye_camera : float
            Distance between eye and camera in meters, measured from the cornea front to
            the camera lens front.

        Returns
        -------
        magnification : float
            Central magnification of the eye - camera system.
        focused_system_matrix : sympy.Matrix
            Ray transfer matrix for the full system.
        focus_lens_radius : float
            Radius of curvature of the camera's focal length, in meters.
        """
        eye_matrix = self.evaluate_matrix()
        focused_system_matrix, focus_lens_radius = camera.focused_system_matrix(
            eye_matrix, distance_eye_camera, return_focus_lens_curvature=True
        )

        magnification = focused_system_matrix[0, 0]

        return magnification, focused_system_matrix, focus_lens_radius


class Camera:
    def __init__(
        self,
        F_cond: NumberOrSymbol = None,  # noqa: N803
        a1: NumberOrSymbol = None,
        camera_type: Literal["default"] = "default",
    ) -> None:
        self.camera_type = "lensTaylor"
        self.F_cond = sp.Symbol("F_cond", real=True)
        self.d_CCD = sp.symbols("d_CCD")
        self.R_foc = sp.Symbol("R_foc", real=True)  # in m
        self.a1 = sp.Symbol("a1", real=True)

        if camera_type == "default":
            self.d_CCD = self.F_cond
        else:
            raise NotImplementedError("Custom CCD distances are not implemented.")

        self.n_glas = 1.5
        self.a1_value = a1

        self.condenser_lens = spherical_interface(self.n_glas, 1.0, -self.F_cond) * spherical_interface(
            1.0, self.n_glas, self.F_cond
        )
        self.focus_lens = spherical_interface(self.n_glas, 1.0, -self.R_foc) * spherical_interface(
            1.0, self.n_glas, self.R_foc
        )
        self.correction_term = sp.Matrix([[1 + self.a1 / self.R_foc, 0], [0, 1.0 / (1 + self.a1 / self.R_foc)]])
        self.ray_transfer_matrix = (
            self.correction_term * uniform_medium(self.d_CCD) * self.focus_lens * self.condenser_lens
        )

        self.M_camera_alg = self.ray_transfer_matrix

        if F_cond and (a1 is not None):
            self.ray_transfer_matrix = self.M_camera_alg.evalf(subs={self.F_cond: F_cond, self.a1: a1})

    def calculate_focus_lens_radius(self, eye_matrix: sp.Matrix, distance_eye_camera: float = 0.05) -> float:
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
        system_matrix = self.ray_transfer_matrix * uniform_medium(distance_eye_camera) * eye_matrix

        # Ror contact cameras such as the Panoret fundus camera, the refractive index (of air) needs to be changed to
        # that of the medium used between the eye and camera. This can be done by changing the use of uniform_medium()
        # to medium_change().
        # The camera should correct for the patients refraction, so the image should be
        # focused, i.e. B = 0
        B = system_matrix[0, 1] + 0.0000000001  # noqa: N806

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
        system_matrix = self.ray_transfer_matrix * uniform_medium(distance_eye_camera) * eye_matrix

        # For contact cameras such as the Panoret fundus camera, the refractive index (of air) needs to be changed to
        # that of the medium use dbetween the eye and camera. This can be done by changing the use of uniform_medium()
        # to medium_change().
        focus_lens_curvature = self.calculate_focus_lens_radius(eye_matrix, distance_eye_camera)
        if return_focus_lens_curvature:
            return (
                system_matrix.evalf(subs={self.R_foc: focus_lens_curvature}),
                focus_lens_curvature,
            )

        return system_matrix.evalf(subs={self.R_foc: focus_lens_curvature})


# Maximum allowed difference between the calculated and specified refraction of the eye model
_MAXIMUM_REFRACTION_DEVIATION = 0.05

# Maximum allowed value of the B-element in a ray transfer matrix for a focused system
_MAXIMUM_FOCUS_DEVIATION = 0.0001


def calculate_magnification(
    eye: Eye,
    camera: Camera,
    distance_eye_camera: float = 0.05,
    focus_lens_radius: float | None = None,
    *,
    suppress_warnings: bool = False,
) -> tuple[float, float, Eye]:
    """Calculate the total magnification of the eye - camera system.

    A structure on the central retina is depicted `magnification` times larger
    on the camera sensor.

    Parameters
    ----------
    eye : Eye
        Eye model.
    camera : Camera
        Camera model.
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
        Central magnification of the eye - camera system.
    glasses_power : float
        Power of the corrective lens required to correct for the eye's refraction, in
        diopters.
    eye_model : Eye
        Eye model used to calculate the magnification.

    Warnings
    --------
    When the calculated glasses power differs significantly from
    `eye.spherical_equivalent`, a warning is displayed.
    """
    glasses_curvature, glasses_power = eye.calculate_refraction(display=False)

    if not suppress_warnings and abs(glasses_power - eye.spherical_equivalent) > _MAXIMUM_REFRACTION_DEVIATION:
        warn(
            f"model refraction {glasses_power:.2f} not matching clinical refraction"
            f" {eye.spherical_equivalent:.2f} for {eye.name}"
        )

    # For contact cameras such as the Panoret fundus camera, this refractive index (of air) needs to be changed to that
    # of the medium between the eye and camera. This can be done by changing the use of uniform_medium() to
    # medium_change().
    system_matrix = camera.ray_transfer_matrix * uniform_medium(distance_eye_camera) * eye.evaluate_matrix()

    if focus_lens_radius is None:
        # system is in focus so B=0
        solutions = list(sp.solve(system_matrix[0, 1] + 0.00001, camera.R_foc))
        focus_lens_radius = solutions[np.argmax(np.abs(np.array(solutions) + camera.a1_value))]

    focused_system_matrix = system_matrix.evalf(subs=({camera.R_foc: focus_lens_radius}))

    magnification: float = focused_system_matrix[0, 0]

    if abs(focused_system_matrix[0, 1]) > _MAXIMUM_FOCUS_DEVIATION:
        warn(f"focused_system_matrix not in focus for patient {eye.name}")

    return magnification, glasses_power, eye
