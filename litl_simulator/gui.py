"""The main lidar gui script."""
import argparse
import logging
import os
import traceback

from PyQt5.QtWidgets import QMessageBox
from pyqtgraph.Qt.QtWidgets import QApplication

from .lidargui import LiDARGui


def main():
    """Main GUI function."""
    logging.basicConfig()
    parser = argparse.ArgumentParser(
            description="GUI to visualize lidar point cloud with corresponding img.")
    parser.add_argument(
            "frame_dir", action="store", default=os.getcwd(), nargs="?",
            help=("The directory containing the frames."),
            )
    parser.add_argument(
            "--loglevel", action="store", default=logging.INFO,
            help=("The logging level (defaults to INFO)."))
    args = parser.parse_args()
    print("Starting GUI app.")
    try:
        app = QApplication([])
        window = LiDARGui(
                app=app,
                root=args.frame_dir,
                loglevel=args.loglevel,
                )
        window.show()
        app.exec_()
    except KeyboardInterrupt:
        print("App killed :(")
    except Exception as exc:
        errorbox = QMessageBox()
        errorbox.setIcon(QMessageBox.Critical)
        errorbox.setStandardButtons(QMessageBox.Close)
        errorbox.setText("An error occured!")
        errorbox.setWindowTitle("Error")
        # taken from:
        # https://stackoverflow.com/a/35712784/6362595
        stack = ''.join(traceback.format_exception(etype=type(exc), value=exc, tb=exc.__traceback__))
        print("An error occured:")
        print(stack)
        errorbox.setInformativeText(stack)
        errorbox.buttonClicked.connect(errorbox.close)
        errorbox.exec_()
    else:
        print("App exited normally.")


if __name__ == "__main__":
    main()
