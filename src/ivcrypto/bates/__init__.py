"""Bates: Heston stochastic variance plus lognormal jumps, priced with the Heston machinery."""

from ivcrypto.bates.calibrate import calibrate_bates, calibrate_variants
from ivcrypto.bates.charfunc import BatesParams, characteristic_function
from ivcrypto.bates.mc import mc_price

__all__ = [
    "BatesParams",
    "calibrate_bates",
    "calibrate_variants",
    "characteristic_function",
    "mc_price",
]
