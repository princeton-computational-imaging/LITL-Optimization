# NOTE: The contents of _version.py are generated automatically during the package building process.
# This is configured in pyproject.toml [tool.setuptools_scm]
from ._version import version

__version__ = version


from .simulation.simulator import LiDARSim
