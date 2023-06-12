"""Utils for GUI module."""
import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette


LOADING_ICON = os.path.join(os.path.dirname(__file__), "loading_icon.gif")


def enable_chkbox(chkbox, enable=True):
    """Enable or Disable a QCheckBox instance."""
    if enable is True or enable == Qt.Checked:
        enable = True
    else:
        enable = False
    chkbox.setEnabled(enable)


def enable_lineeditor(lineeditor, enable=True):
    """Enable or disable a lineeditor."""
    palette = QPalette()
    palette.setColor(QPalette.Text, Qt.black)
    if enable is True or enable == Qt.Checked:
        readonly = False
        palette.setColor(QPalette.Base, Qt.white)
    else:
        readonly = True
        palette.setColor(QPalette.Base, Qt.gray)
    lineeditor.setReadOnly(readonly)
    lineeditor.setPalette(palette)
