"""Custom widgets module."""
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtGui import QFont, QFontMetrics, QPainter
from PyQt5.QtWidgets import (
        QFrame, QGridLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget,
        )


class DynamicLabel(QLabel):
    """Creates a dynamic label that changes its content over time."""
    def __init__(self, texts, delta):
        """Dynamic Label init method.

        Args:
            texts: list
                The list of texts to alternate from.
            delta: float
                Delta time between each text (in sec).
        """
        super().__init__(texts[0])
        self.texts = texts
        self.index = 0
        self.delta = delta * 1000  # in ms
        self._stop = False
        self.flicker(first_time=True)

    def stop(self):
        """Stops the flickering."""
        self._stop = True

    def flicker(self, first_time=False):
        """Flicker the dynamic texts."""
        if self._stop:
            return
        if first_time:
            QTimer.singleShot(self.delta, self.flicker)
            return
        self.index = (self.index + 1) % len(self.texts)
        self.setText(self.texts[self.index])
        QTimer.singleShot(self.delta, self.flicker)


class LineWidget(QFrame):
    """Simple gray line widget to be added in layouts."""

    def __init__(self, alignment="horizontal"):
        """Line widget init method."""
        super().__init__()
        if alignment == "horizontal":
            self.setFixedHeight(2)
            self.setFrameShape(QFrame.HLine)
        elif alignment == "vertical":
            self.setFixedWidth(2)
            self.setFrameShape(QFrame.VLine)
        else:
            raise ValueError(alignment)
        self.setFrameShadow(QFrame.Sunken)
        # self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet("background-color: #2C2C2C;")


# taken from:
# https://stackoverflow.com/a/44593428/6362595
class MathTextLabel(QWidget):
    """Label-like widget to display math equations."""
    def __init__(self, mathText=None, parent=None, **kwargs):
        """Math text label init method."""
        super().__init__(parent, **kwargs)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        r, g, b, a = self.palette().base().color().getRgbF()
        self._figure = Figure(alpha=1, edgecolor=(r, g, b), facecolor=(r, g, b))
        self._canvas = FigureCanvas(self._figure)
        self.layout.addWidget(self._canvas)
        if mathText is not None:
            self.setText(mathText)

    def setText(self, text):
        """Sets the LaTeX equation to be displayed."""
        self._figure.clear()
        text = self._figure.suptitle(
                text, x=0.0, y=1.0,
                horizontalalignment="left",
                verticalalignment="top",
                size=QFont().pointSize(),
                )
        self._canvas.draw()
        (x0, y0), (x1, y1) = text.get_window_extent().get_points()
        w, h = x1 - x0, y1 - y0
        self._figure.set_size_inches(w / 80, h / 80)
        self.setFixedSize(w, h)


class SliderWithIncrementButtons(QWidget):
    """Slider with increment buttons."""

    def __init__(
            self, *args, decrement_symbol="<", increment_symbol=">",
            orientation="horizontal", name=None,
            **kwargs):
        """Slider with increment buttons init method."""
        super().__init__(*args, **kwargs)
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.decrement_button = QPushButton(decrement_symbol)
        self.increment_button = QPushButton(increment_symbol)
        if orientation == "horizontal":
            if name is not None:
                self.name_label = QLabel(name)
                self.layout.addWidget(self.name_label, 0, 0, 1, 1)
                starting_col = 1
            else:
                starting_col = 0
            self.layout.addWidget(self.decrement_button, 0, starting_col, 1, 1)
            self.slider = QSlider(Qt.Horizontal)
            self.layout.addWidget(self.slider, 0, starting_col + 1, 1, 2)
            self.layout.addWidget(self.increment_button, 0, starting_col + 3, 1, 1)
        elif orientation == "vertical":
            if name is not None:
                self.name_label = VerticalLabel(name)
                self.layout.addWidget(self.name_label, 0, 0, 1, 1)
                starting_row = 1
            else:
                starting_row = 0
            self.layout.addWidget(self.decrement_button, starting_row, 0, 1, 1)
            self.slider = QSlider()
            self.layout.addWidget(self.slider, starting_row + 1, 0, 2, 1)
            self.layout.addWidget(self.increment_button, starting_row + 3, 0, 1, 1)
        else:
            raise ValueError(f"Invalid orientation: '{orientation}'.")

    def maximum(self, *args, **kwargs):
        """The maximum value for the slider."""
        return self.slider.maximum(*args, **kwargs)

    def minimum(self, *args, **kwargs):
        """The minimum value for the slider."""
        return self.slider.minimum(*args, **kwargs)

    def setMaximum(self, *args, **kwargs):
        """Sets the maximum slider value."""
        return self.slider.setMaximum(*args, **kwargs)

    def setMinimum(self, *args, **kwargs):
        """Sets the minimum slider value."""
        return self.slider.setMinimum(*args, **kwargs)

    def setValue(self, *args, **kwargs):
        """Sets the slider value."""
        return self.slider.setValue(*args, **kwargs)

    def value(self, *args, **kwargs):
        """Return the slider value."""
        return self.slider.value(*args, **kwargs)


# taken from:
# https://stackoverflow.com/a/67515822/6362595
class VerticalLabel(QLabel):
    """Vertical Label custom class."""

    def paintEvent(self, event):
        """Paint event callback function override."""
        painter = QPainter(self)
        painter.translate(0, self.height())
        painter.rotate(-90)
        fm = QFontMetrics(painter.font())
        xoffset = int(fm.boundingRect(self.text()).width()/2)
        yoffset = int(fm.boundingRect(self.text()).height()/2)
        x = int(self.width()/2) + yoffset
        y = int(self.height()/2) - xoffset
        # because we rotated the label, x affects the vertical placement, and y affects the horizontal
        painter.drawText(y, x, self.text())
        painter.end()

    def minimumSizeHint(self):
        """Override of minimumSizeHint."""
        size = QLabel.minimumSizeHint(self)
        return QSize(size.height(), size.width())

    def sizeHint(self):
        """Override of sizeHint."""
        size = QLabel.sizeHint(self)
        return QSize(size.height(), size.width())
