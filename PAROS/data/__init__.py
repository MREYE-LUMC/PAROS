"""Calibration and lens data files."""

import importlib.resources
from pathlib import Path
from typing import cast

import pandas as pd

__all__ = ("get_camera_calibration_data", "get_lens_data")

calibration = importlib.resources.files("PAROS.data.calibration")
lenses = importlib.resources.files("PAROS.data.lenses")


def _find_available_manufacturers():
    return [i.name for i in calibration.iterdir() if i.is_dir()]


_camera_manufacturers = _find_available_manufacturers()


def get_camera_calibration_data(manufacturer: str, camera: str) -> pd.DataFrame:
    """Get camera calibration data for a given manufacturer and camera.

    Reads the data file at `PAROS/data/calibration/{manufacturer}/{camera}.csv` and
    returns it as a pandas DataFrame.

    Parameters
    ----------
    manufacturer : str
        Camera manufacturer name.
    camera : str
        Camera model name.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing the camera calibration data.
    """
    if manufacturer not in _camera_manufacturers:
        raise KeyError(f"Manufacturer {manufacturer} does not exist.")

    try:
        camera_data_file = cast(Path, calibration.joinpath(manufacturer, camera + ".csv"))

        return pd.read_csv(camera_data_file)
    except FileNotFoundError as e:
        raise KeyError(f"Camera {camera} does not exist for manufacturer {manufacturer}.") from e


def get_lens_data(name: str) -> pd.DataFrame:
    """Get lens data.

    Reads the data file at `PAROS/data/lenses/{name}.csv` and returns it as a pandas DataFrame.

    Parameters
    ----------
    name : str
        Name of the lens data set.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing the lens data.
    """
    try:
        lens_data_file = cast(Path, lenses.joinpath(name + ".csv"))
        return pd.read_csv(lens_data_file)
    except FileNotFoundError as e:
        raise KeyError(f"Lens data does not exist for name {name}.") from e
