import sys


def error_massage_detail(error, error_detail: sys):
    """
    error_massage_detail : str
    This variable holds the detailed error message when an exception occurs.

    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error: {str(error)} in {file_name} at line {exc_tb.tb_lineno}"

    return error_message


class CustomException(Exception):
    """
    A class used to handle custom exceptions and log the error message in the log file.

    ...

    Attributes
    ----------
    error_message : str
        the error message to be displayed when the exception is raised

    Methods
    -------
    __init__(error_message, error_detail)
        Constructs all the necessary attributes for the CustomException object.
    __str__()
        Returns the error message when the exception is raised.
    """

    def __init__(self, error_message, error_detail: sys):
        """
        Constructs all the necessary attributes for the CustomException object.

        Parameters
        ----------
            error_message : str
                the error message to be displayed when the exception is raised
            error_detail : sys
                system-specific parameters and functions
        """

        super().__init__(error_message)
        self.error_message = error_massage_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        """
        Returns the error message when the exception is raised.

        Returns
        -------
        error_message : str
            the error message to be displayed when the exception is raised
        """

        return f"{self.error_message}"
