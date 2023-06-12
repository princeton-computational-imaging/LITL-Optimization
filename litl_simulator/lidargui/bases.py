"""Base classes for GUI."""
import concurrent.futures
import os

from PyQt5.QtCore import QObject, pyqtBoundSignal

from ..bases import BaseUtility


class BaseTaskManager(QObject, BaseUtility):
    """Base class for task managers."""
    _executor_type = "thread"

    def __init__(self, *args, loglevel=None, **kwargs):
        """Base task manager init method."""
        QObject.__init__(self, *args, **kwargs)
        BaseUtility.__init__(self, loglevel=loglevel)
        self._executor = None
        self._create_executor()
        self._set_futures_dict()

    def _set_futures_dict(self):
        self.futures = {}  # dict of all futures. keys are the signal names
        for attr in dir(self.__class__):
            if attr == "all_signals":
                continue
            if isinstance(getattr(self, attr), pyqtBoundSignal):
                self.futures[attr] = []

    @property
    def executor(self):
        """The executor object."""
        return self._executor

    @property
    def all_futures(self):
        """The list of all futures object."""
        all_futures = []
        for futures in self.futures.values():
            all_futures += futures
        return all_futures

    @property
    def all_signals(self):
        """The list of all signals."""
        signals = []
        for attr in dir(self.__class__):
            if attr == "all_signals":
                continue
            obj = getattr(self, attr)
            if isinstance(obj, pyqtBoundSignal):
                signals.append(obj)
        return signals

    def cancel(self):
        """Cancel all jobs."""
        for future in self.all_futures:
            future.cancel()
        self.shutdown()

    def join(self):
        """Wait for all futures to finish."""
        for completed in concurrent.futures.as_completed(self.all_futures):
            continue

    def reset(self, **kwargs):
        """Reset the task manager."""
        self.cancel()
        for key in self.futures:
            self.futures[key] = []
        self.shutdown()
        if self._executor is not None:
            del self._executor
        self._create_executor(**kwargs)
        for signal in self.all_signals:
            try:
                signal.disconnect()
            except Exception:
                pass

    def submit(self, fn, *args, done_callback=None, futures_name=None, **kwargs):
        """Submits a job to the executor.

        Arguments:
            fn: callable
                The function for the job.
            done_callback: callable
                A callable for the 'add_done_callback' method.
            futures_name: str
                The futures list key (for the futures attribute dict).
            other args and kwargs are passed to the fn.
        """
        if futures_name is None:
            raise ValueError("Need to specify 'futures_name'.")
        if futures_name not in self.futures:
            raise KeyError(f"Invalid future_name '{futures_name}'.")
        future = self.executor.submit(fn, *args, **kwargs)
        if done_callback is not None:
            future.add_done_callback(done_callback)
        self.futures[futures_name].append(future)

    def shutdown(self):
        """Shutsdown task manager."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)  # , cancel_futures=True)
            # cancel futures only for python >= 3.9
            for futures_name, futures in self.futures.items():
                for future in futures:
                    future.cancel()
            del self.futures
            self._set_futures_dict()
            del self._executor
            self._executor = None

    def _create_executor(self, nworkers=None):
        # leave 2 cpus for other tasks
        if nworkers is None:
            nworkers = os.cpu_count() - 2
        if nworkers > os.cpu_count() - 2:
            nworkers = os.cpu_count() - 2  # maximize amount of workers
        if self._executor_type == "thread":
            self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=nworkers)
        elif self._executor_type == "process":
            self._executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=nworkers)
        else:
            raise ValueError(self._executor_type)
