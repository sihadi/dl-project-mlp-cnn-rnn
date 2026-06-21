import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(test_size=0.2, random_state=42):
    ds = load_breast_cancer()
    X = ds.data.astype(np.float32)
    y = ds.target.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, ds.feature_names.tolist(), [str(c) for c in ds.target_names.tolist()]
