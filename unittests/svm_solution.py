import numpy as np
from sklearn.svm import SVC


def train_svm_and_predict(train_features, train_target, test_features):
    """
    train_features: np.array, (num_elements_train x num_features) - train data description, the same features and the same order as in train data
    train_target: np.array, (num_elements_train) - train data target
    test_features: np.array, (num_elements_test x num_features) -- some test data, features are in the same order as train features

    return: np.array, (num_elements_test) - test data predicted target, 1d array
    """

    model_svc = SVC(kernel='rbf', C=10)
    model_svc.fit(train_features[:, [0, 1, 3, 4]], train_target)
    return model_svc.predict(test_features[:, [0, 1, 3, 4]])
