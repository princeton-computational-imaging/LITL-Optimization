"""Base classes for all the package."""
import logging


logging.basicConfig()


class BaseUtility:
    """Base class for all objects within this simulation package."""

    def __init__(self, **kwargs):
        """Base class's init method."""
        self._set_logger(**kwargs)

    def _set_logger(self, loglevel=logging.INFO):
        """Sets the logger object."""
        if loglevel is None:
            loglevel = logging.INFO
        self._da_logger = logging.getLogger(type(self).__name__)
        self._da_logger.setLevel(loglevel)

    @property
    def _logger(self):
        """Returns the logger for this object."""
        if hasattr(self, "_da_logger"):
            return self._da_logger
        # will default to INFO
        self._set_logger()
        return self._da_logger

    @property
    def _loglevel(self):
        """The loglevel."""
        return self._logger.level
