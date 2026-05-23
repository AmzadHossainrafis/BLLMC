"""
licence : mit
author : amzad hossain rafi
email : [EMAIL_ADDRESS]

change log :
    17-5-2026 : start
    17-5-2026 : implement gelu activation function

to do :
    1. implement test cases

referance :
    this implementaion of gelu is same as nn.GELU(approximate='tanh')

"""

import torch
import torch.nn as nn


class GELU_taylor(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    This module implements the GELU activation function, which is a smooth approximation
    of the Rectified Linear Unit (ReLU).

    The formula is: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Args:
        None
    """

    def __init__(self):
        """
        Initializes the GELU module.
        """
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


def GELU():
    return nn.GELU(approximate="tanh")
