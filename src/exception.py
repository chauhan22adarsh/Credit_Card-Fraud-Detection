import sys


def error_message_detail(error, error_detail: sys):
    """
    Builds a detailed error message including the file name and line
    number where the exception actually occurred — much more useful for
    debugging a pipeline than Python's default traceback alone, especially
    once code is spread across several files like this project is.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = (
        f"Error occurred in python script [{file_name}] "
        f"at line number [{line_number}]: {str(error)}"
    )
    return error_message

class CustomException(Exception):
    """
    Wraps any caught exception with the extra file/line detail above.
    Every component in this project raises this instead of letting a bare
    exception propagate, so a failure anywhere in the pipeline is easy to
    trace back to its exact source.
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
    