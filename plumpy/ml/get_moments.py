from plumpy.sigproc.general import sec2ind
import numpy as np
import argparse

def get_moments(source, xmin, xmax=None, duration=None, units='seconds', sr=None):
    '''
    If using timestamps, units are seconds. Specify either xmax or duration

    :param source: str for filename or 2d ndarray of data, where 1st dim is time
    :param xmin: float for timestamp or int for sample for the start of baseline period
    :param xmax: float for timestamp or int for sample for the end of baseline period
    :param duration: float for timestamp or int for sample for the duration of baseline period
    :param units: str: 'seconds' or 'samples'
    :param sr: float, sampling rate
    :return: mean and std over the baseline period in the data
    '''
    if type(source) is str:
        x = np.load(source)
    elif type(source) is np.ndarray:
        x = source
    else:
        raise NotImplementedError

    if xmax and duration:
        raise Exception('Cannot have both xmax and duration, choose one')
    if xmax:
        xmax_ = xmax
    elif duration:
        xmax_ = xmin + duration
    else:
        raise NotImplementedError

    if units == 'seconds':
        assert sr is not None, 'Sr needs to be set for seconds'
        beg = sec2ind(xmin, sr)
        end = sec2ind(xmax_, sr)
    elif units == 'samples':
        beg = xmin
        end = xmax_
    else:
        raise NotImplementedError
    temp = x[beg:end]
    return np.mean(temp, 0), np.std(temp, 0, ddof=1)

##
if __name__ == '__main__':
    # TODO: this needs to be tested: xmax vs duration and default values for params
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--source', '-i', type=str, help='Input file', default='')
    parser.add_argument('--xmin', help='Float for timestamp or int for sample for the start of baseline period', default='')
    parser.add_argument('--xmax', help='Float for timestamp or int for sample for the end of baseline period', nargs='?', const=1, default=1)
    parser.add_argument('--duration', help='Float for timestamp or int for sample for the duration of baseline period', nargs='?')
    parser.add_argument('--units', type=str, help='Units', default='', choices=['seconds', 'samples'])
    parser.add_argument('--sr', type=float, help='Sampling rate', nargs='?')
    args = parser.parse_args()

    get_moments(args.source, args.xmin, args.xmax, args.duration, args.units, args.sr)
