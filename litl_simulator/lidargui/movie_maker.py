"""Movie maker module to make a movie out of the GUI."""
import concurrent.futures
import os
import subprocess

from PyQt5.QtCore import pyqtSignal, QRect
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import (
        QCheckBox, QDialog, QGridLayout, QLabel, QLineEdit,
        QMessageBox, QProgressBar, QPushButton,
        )

from .bases import BaseTaskManager
from .custom_widgets import DynamicLabel
from .utils import LOADING_ICON
from ..bases import BaseUtility
from ..settings import MOVIE_FRAMES_DIRECTORY


class MovieMakerProgressDialog(QDialog, BaseUtility):
    """Movie Maker progress dialog."""

    def __init__(self, task_manager, movie_maker_viewport, *args, **kwargs):
        """Movie maker progress dialog init method."""
        QDialog.__init__(self)
        BaseUtility.__init__(self, *args, **kwargs)
        self.movie_maker_viewport = movie_maker_viewport
        self.task_manager = task_manager
        self.build_gui()

    def build_gui(self):
        """Builds the gui."""
        self.setModal(True)
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.setWindowTitle("Processing movie...")

        self.desc_label = QLabel()
        self.desc_label.setText("Creating movie... please wait")
        self.layout.addWidget(self.desc_label, 0, 0, 1, 1)

        self.gif_label = QLabel()
        self.layout.addWidget(self.gif_label, 1, 0, 1, 1)
        self.gif_movie = QMovie(LOADING_ICON)
        if not self.gif_movie.isValid():
            self.gif_label.setText("U DONUT")
        else:
            self.gif_label.setMovie(self.gif_movie)
            self.gif_movie.start()

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(self.movie_maker_viewport.mainhub.nframes)
        self.layout.addWidget(self.progress_bar, 2, 0, 1, 1)

        self.abort_btn = QPushButton("Abort")
        self.abort_btn.clicked.connect(self.cancel)
        self.layout.addWidget(self.abort_btn, 3, 0, 1, 1)

    def cancel(self):
        """Abort creating movie."""
        self.task_manager.cancel()
        # stop playing movie if needed
        self.movie_maker_viewport.mainhub.stop()
        self.done(0)

    def finished_movie(self, signal):
        """Callback when movie is finished."""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Movie successfully created!")
        msg.setWindowTitle("MAN'S INTERNATIONAL FAM!")
        msg.setStandardButtons(QMessageBox.Ok)
        self.close()
        self.movie_maker_viewport.close()
        msg.exec()

    def compiling_movie(self):
        """Set the progress bar to max and replace it with message."""
        msgs = ["Compiling movie " + x for x in (["." * i for i in range(1, 4)] + ["..."] * 5)]
        self.compiling_label = DynamicLabel(msgs, delta=0.1)
        self.layout.replaceWidget(
                self.progress_bar, self.compiling_label)
        self.layout.addWidget(self.compiling_label, 2, 0, 1, 1)
        self.progress_bar.setParent(None)

    def increment_progress_bar(self):
        """Increment the progress bar."""
        self.progress_bar.setValue(self.progress_bar.value() + 1)


class MovieMakerTaskManager(BaseTaskManager):
    """Movie task manager."""
    finished_movie = pyqtSignal(object)

    def submit(self, fn, *args, **kwargs):
        """Submit job."""
        future = self.executor.submit(fn, *args, **kwargs)
        future.add_done_callback(self.done_callback)
        self.futures["finished_movie"].append(future)

    def done_callback(self, future):
        """Method called once task is finished."""
        # emit signal
        try:
            exception = future.exception()
        except concurrent.futures.CancelledError:
            self.finished_movie.emit("cancel")
        if type(exception) is KeyboardInterrupt:
            self.finished_movie.emit("cancel")
            return
        if exception:
            self.finished_movie.emit(exception)
        else:
            self.finished_movie.emit("done")


class MovieMaker(BaseUtility, QDialog):
    """Movie maker dialog."""

    def __init__(self, mainhub, **kwargs):
        """Movie maker dialog init method."""
        self.mainhub = mainhub
        QDialog.__init__(self, mainhub)
        BaseUtility.__init__(self, **kwargs)
        self.build_gui()
        # other widgets
        self.progress_dialog = None
        self.task_manager = MovieMakerTaskManager()

    def build_gui(self):
        """Builds the gui."""
        self.setWindowTitle("Make Movie")
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.framerate_desc = QLabel()
        self.framerate_desc.setText("Movie Framerate (FPS):")
        self.layout.addWidget(self.framerate_desc, 0, 0, 1, 1)

        self.framerate_lineeditor = QLineEdit()
        self.framerate_lineeditor.setText("10")
        self.layout.addWidget(self.framerate_lineeditor, 0, 1, 1, 1)

        self.camera_chkbox = QCheckBox("Add camera viewport to movie")
        self.layout.addWidget(self.camera_chkbox, 1, 0, 1, 2)

        self.overwrite_chkbox = QCheckBox("Overwrite movie frames: ")
        self.layout.addWidget(self.overwrite_chkbox, 2, 0, 1, 2)

        self.make_movie_btn = QPushButton("Make Movie")
        self.make_movie_btn.clicked.connect(self.make_movie)
        self.layout.addWidget(self.make_movie_btn, 3, 0, 1, 1)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.close)
        self.layout.addWidget(self.cancel_btn, 3, 1, 1, 1)

    @property
    def framerate(self):
        """The framerate for the movie."""
        return int(self.framerate_lineeditor.text())

    @property
    def current_frame(self):
        """Return the current frame index."""
        return self.mainhub.current_frame

    @property
    def nframes(self):
        """Return the maximal number of frames."""
        return self.mainhub.nframes

    def make_movie(self):
        """Make the movie."""
        # show progress dialog
        self.progress_dialog = MovieMakerProgressDialog(
                self.task_manager, self,
                loglevel=self._loglevel,
                )
        self.progress_dialog.show()
        # first generate movie images by playing all the frames and making screenshots
        if not self.overwrite_chkbox.isChecked():
            # go to the first frame not generated
            for iframe in range(self.nframes):
                savepath = os.path.join(
                        MOVIE_FRAMES_DIRECTORY, f"frame{iframe}.jpg")
                if os.path.exists(savepath):
                    self.progress_dialog.increment_progress_bar()
                    continue
                self.mainhub.go_to_frame(iframe)
                break
        else:
            iframe = 0
            self.mainhub.go_to_beginning()
        if iframe != self.nframes - 1:
            self.mainhub.play(
                callback=self.screenshot_callback,
                callback_kwargs={
                    "overwrite": self.overwrite_chkbox.isChecked(),
                    "camera": self.camera_chkbox.isChecked(),
                    },
                loop=False,
                )
        else:
            self.start_compiling()

    def screenshot_callback(self, overwrite=False, camera=False):
        """Screenshot callback when playing whole movie."""
        savedir = os.path.join(self.mainhub.root_dir, MOVIE_FRAMES_DIRECTORY)
        os.makedirs(savedir, exist_ok=True)
        width, height = self.grab_widget(
                self.mainhub.pc_viewport.pc_viewer,
                os.path.join(savedir, f"pc_frame{self.current_frame}.jpg"),
                overwrite=overwrite)
        if camera:
            if not self.mainhub.cam_viewport.isVisible():
                self.mainhub.show_cam_viewport()
            self.grab_widget(
                    self.mainhub.cam_viewport.cam_viewport,
                    os.path.join(savedir, f"cam_frame{self.current_frame}.jpg"),
                    overwrite,
                    )
        if self.current_frame == self.nframes - 1:
            self.start_compiling(camera, width, height)
        else:
            self.progress_dialog.increment_progress_bar()

    def grab_widget(self, widget, savepath, overwrite):
        """Take a screenshot of the given widget."""
        if os.path.exists(savepath):
            if overwrite:
                os.remove(savepath)
            else:
                return
        w = widget.width()
        if w % 2:
            w -= 1
        h = widget.height()
        if h % 2:
            h -= 1
        rect = QRect(0, 0, w, h)
        self._logger.info(f"Saving screenshot at: {savepath} with w x h = {w} x {h}")
        pixmap = widget.grab(rect)
        pixmap.save(savepath)
        return w, h

    def start_compiling(self, *args):
        """Start compiling movie."""
        self.progress_dialog.compiling_movie()
        self.task_manager.reset()
        self.task_manager.finished_movie.connect(self.progress_dialog.finished_movie)
        self.task_manager.submit(self._compile_movie, self.framerate, *args)

    def _compile_movie(self, framerate, camera, width, height):
        # this method should be executed within a thread
        movie_dir = os.path.join(self.mainhub.root_dir, MOVIE_FRAMES_DIRECTORY)
        pc_movie_path = os.path.join(movie_dir, "pc_movie.mp4")
        self._logger.info(f"Compiling PC movie: {pc_movie_path}")
        if os.path.exists(pc_movie_path):
            os.remove(pc_movie_path)
        frames = os.path.join(movie_dir, "pc_frame%d.jpg")
        command = f"ffmpeg -r {framerate} -i {frames} {pc_movie_path}"
        subprocess.run(command.split(" "))
        if camera:
            cam_movie_path = os.path.join(movie_dir, "cam_movie.mp4")
            if os.path.exists(cam_movie_path):
                os.remove(cam_movie_path)
            self._logger.info(f"Compiling Cam movie {cam_movie_path}")
            frames = os.path.join(movie_dir, "cam_frame%d.jpg")
            command = f"ffmpeg -r {framerate} -i {frames} {cam_movie_path}"
            subprocess.run(command.split(" "))
            # # combine movies
            # final_path = os.path.join(movie_dir, "final_movie.mp4")
            # self._logger.info(f"Combining cam + pc movies: {final_path}")
            # # need to scale cam to match pc movie
            # command = (
            #         f'ffmpeg -i {cam_movie_path} -i {pc_movie_path} -filter_complex '
            #         f'"[0:v]scale={width}:{height}[v0];[v0][1:v]hstack=inputs=2" {final_path}'
            #         )
            # subprocess.run(command.split(" "))
