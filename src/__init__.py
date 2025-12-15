"""
Energy-Aware Perception Scheduling for Autonomous Vehicles

A machine learning framework for optimizing energy consumption in AV perception systems.
"""

__version__ = "1.0.0"
__author__ = "Cepher Onano"
__email__ = "rconano@wm.edu"

from . import data_processing
from . import models
from . import perception
from . import scheduling
from . import evaluation
from . import utils

__all__ = [
    "data_processing",
    "models",
    "perception",
    "scheduling",
    "evaluation",
    "utils"
]
