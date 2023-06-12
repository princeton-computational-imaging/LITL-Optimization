"""Main app module."""
from PyQt5.QtWidgets import (
        QMainWindow,
        )
from PyQt5.QtGui import QGuiApplication
from ..bases import BaseUtility


def kill_app():
    """Kills the main application."""
    LiDARGui.kill()


# Only one instance allowed.
class LiDARGui(QMainWindow, BaseUtility):
    """Main Window for lidar viewer app and accessories."""
    app = None

    def __init__(self, app, root, **kwargs):
        """Main GUI app init method."""
        QMainWindow.__init__(self)
        BaseUtility.__init__(self, **kwargs)
        if self.app is not None:
            raise RuntimeError("Only one app instance allowed!")
        LiDARGui.app = app

        # make window full screen
        self.screen = QGuiApplication.screens()[0]
        self.monitor = self.screen.availableGeometry()
        self.monitor.setHeight(self.monitor.height())
        self.setGeometry(self.monitor)
        self.setAcceptDrops(True)

        # main widget (contains viewer)
        # import here to avoid import loops
        from .main_widget import MainWidget
        self.centerWidget = MainWidget(self, root=root, loglevel=self._loglevel)
        self.setCentralWidget(self.centerWidget)

    @classmethod
    def kill(cls):
        """Kills the application."""
        if cls.app is None:
            raise RuntimeError("No app to kill...")
        cls.app.quit()
