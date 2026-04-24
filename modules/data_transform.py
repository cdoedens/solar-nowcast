import numpy as np

def log_transform(y, eps=0.1):
    """
    Log-transform cloud optical depth safely.
    
    Parameters:
        y (array-like): original COD values
        eps (float): small constant to avoid log(0)
    
    Returns:
        transformed y
    """
    return np.log(y + eps)


def inverse_log_transform(y_log, eps=1e-6):
    """
    Inverse transform back to original COD scale.
    """
    return np.exp(y_log) - eps