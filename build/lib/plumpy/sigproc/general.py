import numpy as np
from scipy.stats import rankdata
from fractions import Fraction
from scipy.signal import resample_poly
from decimal import Decimal, ROUND_HALF_UP


def cross_correlate(x, y, type='pearson'):
    '''
    Cross-correlate 2 signals, return values in [-1 to 1]

    :param x: ndarray
    :param y: ndarray
    :param type: str: type of correlation
    :return: ndarray of cross-correlation
    '''
    if type == 'spearman':
        x, y = rankdata(x), rankdata(y)
    x = (x - np.mean(x)) / (np.std(x) * x.shape[0])
    y = (y - np.mean(y)) / np.std(y)
    return np.correlate(x, y, mode='full')


def resample(x, sr1, sr2, axis=0):
    '''
    Resample signal

    :param x: ndarray, time x channels
    :param sr1: float: target sampling rate
    :param sr2: float: source sampling rate
    :param axis: axis of array to apply function to
    :return:
    '''
    a, b = Fraction(sr1, sr2)._numerator, Fraction(sr1, sr2)._denominator
    return resample_poly(x, a, b, axis).astype(np.float32)

def calculate_rms(x, axis=-1):
    '''
    Calculate RMS of the signal

    :param x: ndarray, channels x time
    :param axis: axis of array to apply function to
    :return: ndarray: RMS values
    '''
    get_rms = lambda d: np.sqrt(np.sum(d.astype(float)**2)/len(d))
    return np.apply_along_axis(get_rms, axis, x)

def smooth_signal_1d(y, n):
    '''
    Smooth 1d signal

    :param y: ndarray: signal
    :param n: size of smoothing
    :return: smoothed signal
    '''
    #box = np.ones(n)/n
    #ys = np.convolve(y, box, mode='same')
    #return ys
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(y, axis=0, size=n)
    
def sec2ind(s, sr):
    '''
    Convert timestamp to sample

    :param s: float, timestamp in seconds
    :param sr: float, sampling rate
    :return: int: sample that corresponds to the timestamp
    '''
    return int(Decimal(s * sr).quantize(0, ROUND_HALF_UP))

def ind2sec(ind, sr):
    '''
    Convert timestamp to sample

    :param ind: int, sample index
    :param sr: float, sampling rate
    :return: float: timestamp in seconds that corresponds to the sample index
    '''
    return float(ind)/sr