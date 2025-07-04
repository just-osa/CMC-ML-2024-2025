import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype
        self.unique_values = {}

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        self.unique_values = {col: sorted(X[col].dropna().unique()) for col in X.columns}

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        X_copy = X.copy().dropna()

        for column_name, unique_vals in self.unique_values.items():
            for value in unique_vals:
                X_copy[f"{column_name}_{value}"] = np.where(X_copy[column_name] == value, 1, 0)

        X_copy.drop(columns=self.unique_values.keys(), inplace=True)
        return X_copy.to_numpy()

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype
        self.encoded = {}

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """

        for column_name in X.columns:
            unique_values = X[column_name].unique()
            for value in unique_values:
                self.encoded[f"{column_name}_{value}"] = [
                    Y[X[column_name] == value].mean(),
                    (X[column_name] == value).mean()
                ]

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        # your code here
        transformed = []

        for column_name in X.columns:
            column_transformed = []
            for value in X[column_name]:
                mean, proportion = self.encoded[f"{column_name}_{value}"]
                relation = (mean + a) / (proportion + b)
                column_transformed.append([mean, proportion, relation])

            transformed.append(column_transformed)

        return np.hstack([np.array(t) for t in transformed])

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds
        self.statistics = []
        """
        statistics = [
            (
                fold_idx,
                current_fold_statistics = {
                    column1: {
                        value1: [mean1, proportion1],
                        value2: [mean2, proportion2],
                        ...
                    }
                    column2: {
                        value1: [mean1, proportion1],
                        value2: [mean2, proportion2],
                        ...
                    }
                    ...
                }
            ),
            (...)
             ...
        ]
        """
    def fit(self, X, Y, seed=1):
        for fold_idx, rest_idx in group_k_fold(X.shape[0], self.n_folds, seed):
            current_fold_statistics = {}
            X_fold, Y_fold = X.iloc[rest_idx], Y.iloc[rest_idx]
            for column in X.columns:
                unique_values = X_fold[column].unique()
                current_fold_statistics[column] = {}
                for value in unique_values:
                    current_fold_statistics[column][value] = [Y_fold[X_fold[column] == value].mean(), np.mean(X_fold[column] == value)]
            self.statistics.append((fold_idx, current_fold_statistics))

    def transform(self, X, a=1e-5, b=1e-5):
        n_objects, n_features = X.shape
        result = np.zeros((n_objects, 3 * n_features), dtype=self.dtype)
        for fold_idx, current_fold_statistics in self.statistics:
            for column_idx, column in enumerate(X.columns):
                for row_idx in fold_idx:
                    value = X.iloc[row_idx, column_idx]
                    mean, proportion = current_fold_statistics[column][value]
                    result[row_idx, 3 * column_idx] = mean
                    result[row_idx, 3 * column_idx + 1] = proportion
                    result[row_idx, 3 * column_idx + 2] = (mean + a) / (proportion + b)
        return result

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
        param x: training set of one feature, numpy-array, shape [n_objects,]
        param y: target for training objects, numpy-array, shape [n_objects,]
        returns: optimal weights, numpy-array, shape [|x unique values|,]
        """
    unique_values = np.unique(x)
    enc_x = np.eye(unique_values.shape[0])[x]
    weights = np.zeros(enc_x.shape[1])
    lr = 1e-2

    for i in range(1000):
        p = np.dot(enc_x, weights)
        grad = np.dot(enc_x.T, (p - y))
        weights -= grad * lr

    return weights
