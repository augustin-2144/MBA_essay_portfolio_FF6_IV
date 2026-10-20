import numpy as np


# Converts a pandas Series/DataFrame to its underlying numpy array (via `.values`), or passes
# numpy arrays / lists through `np.asarray` unchanged. Used throughout the optimization and
# metrics code so every function accepts either pandas or numpy inputs.
def to_array(x):
    return np.asarray(x.values if hasattr(x, "values") else x)
