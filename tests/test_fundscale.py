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


def test_calculate_refraction(navarro_eye):
    refraction = navarro_eye.calculate_refraction()

    matrix = navarro_eye.evaluate_matrix()
    P = sp.symbols("P")
    lens = sp.Matrix([[1, 0], [-P, 1]])
    vertex = sp.Matrix([[1, 0.014], [0, 1]])
    system_matrix = lens * vertex * matrix

    solutions = sp.solveset(system_matrix[1, 1], P)
    glasses_power = next(iter(solutions))

    assert pytest.approx(refraction, abs=1e-4) == -0.000
    assert pytest.approx(glasses_power, abs=1e-4) == refraction
