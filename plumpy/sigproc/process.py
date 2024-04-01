'''

'''
import mne
import numpy as np
import typing
from plumpy.utils.plots import plot_psd, save_plot
from plumpy.sigproc.general import resample, smooth_signal_1d

def process_mne(data: np.ndarray,
                channels: list,
                sr: float,
                ch_types: str = 'ecog',
                plot_path: str = None,
                data_name: str = None,
                bad: list = None,
                sr_post: float = 100.,
                n_smooth: int = 1,
                freqs: typing.Dict = None):
    '''
    Perform preprocessing on raw ECoG data: channels x time
    TODO: turn to pipeline with processors as for online preprocessing

    :param data: ndarray of raw data: channels x time
    :param channels: list of str: channel names, length == number of channels
    :param sr: float: sampling date
    :param ch_types: str: channel type
    :param plot_path: str: path to save plots, no plots are made is None
    :param data_name: name for plots, e.g. "s001_run002_14nav"
    :param bad: list of int: indices of bad channels
    :param sr_post: float: target sr for preprocessed data
    :param n_smooth: int: size for smoothing
    :param freqs: dict of frequencies to extract with a wavelet transform
    :return:
        d_out: dict with processed data per frequency band
    '''

    if not freqs:
        freqs = dict(hfb=range(60, 90))

    info = mne.create_info(ch_names=[str(i) for i in channels], sfreq=sr, ch_types=ch_types, verbose=None)
    d_ = mne.io.RawArray(data, info, first_samp=0, copy='auto', verbose=None)
    if plot_path:
        plot_psd(signal=d_, fmax=sr/2)
        save_plot(plot_path, name=data_name + '_raw_psd')

    d_.info['bads'].extend([str(i) for i in bad + 1])
    # d_.drop_channels(d_.info['bads'])

    ##
    d_.notch_filter(freqs=np.arange(50, 1000, 50))
    plot_psd(signal=d_, fmax=sr/2)
    save_plot(plot_path, name=data_name + '_notch_psd')

    ## ignores bad by default
    d_ref, ref_par = mne.set_eeg_reference(d_.copy(), 'average')
    plot_psd(signal=d_ref, fmax=sr/2)
    save_plot(plot_path, name=data_name + '_notch_car_psd')

    ##
    d_res = d_ref.copy().resample(sfreq=500)
    plot_psd(signal=d_res, fmax=250)
    save_plot(plot_path, name=data_name + '_notch_car_500Hz_psd')

    ##
    d_out = {}
    # for band, freqs in zip(['hfb', 'beta', 'alpha'], (np.arange(60, 180), np.arange(13, 30), np.arange(8, 12))):
    # for band, freqs in zip(['hfb'], [np.arange(60, 90)]):
    for band, fq in freqs.items():
        processed = mne.time_frequency.tfr_array_morlet(np.expand_dims(d_res._data, 0),
                                                        # (n_epochs, n_channels, n_times)
                                                        sfreq=500,
                                                        freqs=fq,
                                                        verbose=True,
                                                        n_cycles=4. * 2 * np.pi,
                                                        n_jobs=1)
        processed = np.mean(np.abs(processed), 2).squeeze().T
        d_out[band] = smooth_signal_1d(np.log(resample(processed, sr_post, 500)), n=n_smooth)

    return d_out