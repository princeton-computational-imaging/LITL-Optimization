"""Module for an error dialog wrapper function."""
from functools import wraps
import inspect
import traceback

from PyQt5.QtWidgets import QMessageBox


def decorator_check_arguments(decorator):
    """Decorator for decorators.

    Checks if the decorater have been called with args or not.
    Returns a decorated decorator that handles either case.
    """
    # taken from:
    # https://stackoverflow.com/a/14412901/6362595
    @wraps(decorator)
    def new_decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # decorator called without any arguments
            # i.e.: directly applied to a function.
            return decorator(args[0])
        # arguments used
        # return the decorator called with the arguments
        return lambda true_function: decorator(true_function, *args, **kwargs)
    return new_decorator


@decorator_check_arguments
def error_dialog(fn, text=None, title=None, callback=None):
    # fn, text=None, title=None, callback=None):
    """Decorator for functions we wish to have an error display if something happens."""

    @wraps(fn)
    def wrapped(*args, callback=callback, **kwargs):
        """Final wrapped function."""
        try:
            to_return = fn(*args, **kwargs)
        except Exception as err:
            # create error dialog
            errorbox = QMessageBox()
            errorbox.setIcon(QMessageBox.Critical)
            errorbox.setStandardButtons(QMessageBox.Close)
            msg = "U DONUT! An error occured:" if text is None else text
            # add fn name
            msg += "\nfunc:\n" + fn.__qualname__ + str(inspect.signature(fn))
            errorbox.setText(msg)
            errorbox.setWindowTitle("U PIG!" if title is None else title)
            # taken from:
            # https://stackoverflow.com/a/35712784/6362595
            stack = ''.join(traceback.format_exception(
                etype=type(err), value=err, tb=err.__traceback__))
            print(text)
            print(stack)
            errorbox.setInformativeText(stack)
            errorbox.buttonClicked.connect(errorbox.close)
            errorbox.exec_()
            if callback is not None:
                callback()
        else:
            return to_return
    return wrapped
