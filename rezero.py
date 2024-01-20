import torch
import torch.nn as nn
import torch.nn.functional as F


class ReZero(nn.Module):
    """ReZero.

    Implements a ReZero layer (Bachlechner et al., 2020). ReZero initializes
    residual layers to identity functions using a simple gating mechanism.

    Parameters
    ----------
    >>> module = ReZero(
    ...     embedding_dimension=256,
    ...     module=TransformerBlock(...),
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(self, embedding_dimension: int, module: nn.Module) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        module : nn.Module
            The target module.
        """

        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(1e-3))
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

        return x + self.alpha * self.module(x)
