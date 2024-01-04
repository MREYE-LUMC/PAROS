"""PARaxial Optical fundus Scaling (PAROS)

Paros is a method to calculate the magnification of fundus images based on the optical characteristics of the patient's
eye. The full method and validation are described in (article reference to be added upon acceptance).
"""

from PAROS import data, eyephantom, fundscale

__all__ = ("data", "eyephantom", "fundscale")
__version__ = "1.0.0"
