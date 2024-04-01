'''

'''
from plumpy.ml.get_moments import get_moments

def zscore(x, y=None, xmin=None, xmax=None, duration=None, units='seconds', sr=None):
    '''
    Z-score the data using its own mean and scale or mean and scale of another array (y).
    Mean and scale can also be calculated using a subset of data, e.g. rest period between xmin and xmax.
    Xmin and xmax can be timestamps in seconds or data samples.
    Note that sklearn uses ddof=0, whereas get_moments uses ddof=1. This results in very small differences.

    :param x: ndarray: data to zscore, time x channels
    :param y: ndarray: to use as a base for calculating mean/scale, if different from x
    :param xmin: float for timestamp or int for sample for the start of baseline period
    :param xmax: float for timestamp or int for sample for the end of baseline period
    :param duration: float for timestamp or int for sample for the duration of baseline period
    :param units: units: str: 'seconds' or 'samples'
    :param sr:
    :return:
    '''
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    if y is not None:
        z = y.copy()
    else:
        z = x.copy()

    if xmin is not None:
        scaler.mean_, scaler.scale_ = get_moments(z, xmin, xmax, duration, units, sr)
    else:
        scaler.fit(z)

    return scaler.transform(x), scaler