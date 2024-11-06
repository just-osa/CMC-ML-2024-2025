import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:

    fold_size = num_objects // num_folds

    numbers = np.arange(num_objects)
    folds = []
    cur = 0
    for i in range(num_folds - 1):
        folds.append(numbers[cur:cur + fold_size])
        cur += fold_size
    folds.append(numbers[cur:])

    res = []
    for i in range(num_folds):
        test = folds[i]
        train = np.hstack(folds[:i] + folds[i + 1:])
        res.append((train, test))

    return res


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:

    res = defaultdict(float)

    for n_neighbors in parameters['n_neighbors']:
        for metric in parameters['metrics']:
            for weight in parameters['weights']:
                for normalizer, normalizer_name in parameters['normalizers']:
                    fold_res = []

                    for train_idx, test_idx in folds:
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]
                        if normalizer is not None:
                            X_train = normalizer.fit_transform(X_train)
                            X_test = normalizer.transform(X_test)

                        model = knn_class(n_neighbors=n_neighbors, metric=metric, weights=weight)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        fold_score = score_function(y_test, y_pred)
                        fold_res.append(fold_score)
                    folds_mean = np.mean(fold_res)
                    res[(normalizer_name, n_neighbors, metric, weight)] = folds_mean

    return dict(res)
