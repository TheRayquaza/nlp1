from abc import ABC, abstractmethod
from typing import Any, Self
import numpy as np
from exchange.logger import get_logger

class AbstractModel(ABC):
    """Abstract base class for machine learning models."""

    def __init__(self: Self, config: Any) -> None:
        self.config = config
        self.logger = get_logger(self.config.name)
        self.model = None

    @abstractmethod
    def fit(self: Self, X: np.ndarray, y: np.ndarray) -> Self:
        """Train the model using the provided features and target."""
        pass

    @abstractmethod
    def predict(self: Self, X: np.ndarray) -> np.ndarray:
        """Generate predictions from the input features."""
        pass

    @abstractmethod
    def save(self: Self, path: str) -> None:
        """Save the model parameters to a file."""
        pass

    @abstractmethod
    def load(self: Self, path: str) -> Self:
        """Load model parameters from a file."""
        pass
