import pytest
import sympy as sp

from PAROS import fundscale


@pytest.fixture
def navarro_eye():
    return fundscale.Eye()


@pytest.fixture
def topcon_camera():
    return fundscale.Camera(F_cond=0.02657, a1=0.03481, pixel_density=100)


def test_navarro_magnification(navarro_eye, topcon_camera):
    magnification = topcon_camera.calculate_magnification(navarro_eye)

    assert pytest.approx(magnification, rel=1e-3) == -161.343


@pytest.mark.parametrize(
    "geometry, expected_refraction",
    [
        ({}, 0.000),
        ({"R_corB": -6.25e-3, "R_lensB": 5.555e-3, "D_vitr": 18.3203e-3}, -6.0001),
        ({"R_corB": -6.25e-3, "R_lensB": 6.283e-3, "D_vitr": 14.3203e-3}, 6.0003),
    ],
)
def test_calculate_refraction(
    geometry: fundscale._PartialEyeGeometry, expected_refraction: float
):
    eye = fundscale.Eye(geometry=geometry)
    refraction = eye.calculate_refraction()

    matrix = eye.evaluate_matrix()
    P = sp.symbols("P")
    lens = sp.Matrix([[1, 0], [-P, 1]])
    vertex = sp.Matrix([[1, 0.014], [0, 1]])
    system_matrix = lens * vertex * matrix

    solutions = sp.solve(system_matrix[1, 1], P)
    glasses_power = next(iter(solutions))

    assert pytest.approx(refraction, abs=1e-4) == expected_refraction
    assert pytest.approx(glasses_power, abs=1e-4) == refraction
