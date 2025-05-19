from abc import ABC, abstractmethod
from typing import Any, Tuple, Self
from pydantic import BaseModel
from exchange.logger import get_logger
import pandas as pd
import numpy as np

class AbstractPipeline(ABC):
    """Abstract base class for a data processing pipeline."""

    def __init__(self, config: BaseModel) -> None:
        self.config = config
        self.logger = get_logger(self.config.name)

    @abstractmethod
    def load(self: Self) -> Any:
        """Load raw data."""
        pass

    @abstractmethod
    def transform(self: Self, data: pd.DataFrame) -> np.ndarray:
        """Transform raw data into features>"""
        pass

    @abstractmethod
    def fit(self: Self, X: Any, y: Any) -> Any:
        """Train model or pipeline components."""
        pass

    @abstractmethod
    def save(self: Self, path: str) -> None:
        """Save the model or outputs."""
        pass
