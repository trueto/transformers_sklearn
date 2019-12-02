import numpy as np
def to_numpy(X):
    """
    Convert input to numpy ndarray
    """
    if hasattr(X, 'iloc'):              # pandas
        return X.values
    elif isinstance(X, list):           # list
        return np.array(X)
    elif isinstance(X, np.ndarray):     # ndarray
        return X
    else:
        raise ValueError("Unable to handle input type %s"%str(type(X)))


def unpack_text_pairs(X):
    """
    Unpack text pairs
    """
    if X.ndim == 1:
        texts_a = X
        texts_b = None
    else:
        texts_a = X[:, 0]
        texts_b = X[:, 1]

    return texts_a, texts_b


def unpack_data(X, y=None):
    """
    Prepare data
    """
    X = to_numpy(X)
    texts_a, texts_b = unpack_text_pairs(X)

    if y is not None:
        y = to_numpy(y)
        labels = y
        return texts_a, texts_b, labels
    else:
        return texts_a, texts_b, None