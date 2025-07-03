import numpy as np
import typing


class MinMaxScaler:
    def __init__(self):
        self.min_columns = None
        self.max_columns = None

    def fit(self, data: np.ndarray) -> None:
        self.min_columns = np.min(data, axis=0)
        self.max_columns = np.max(data, axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.min_columns) / (self.max_columns - self.min_columns)


class StandardScaler:
    def __init__(self):
        self.mean_columns = None
        self.std_columns = None

    def fit(self, data: np.ndarray) -> None:
        self.mean_columns = np.mean(data, axis=0)
        self.std_columns = np.std(data, axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean_columns) / self.std_columns
