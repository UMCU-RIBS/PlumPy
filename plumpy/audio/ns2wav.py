#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

from wonambi import Dataset
from numpy import abs, nanmax
from scipy.io.wavfile import write


def main():
    parser = ArgumentParser(
        prog='ns2wav',
        description="convert blackrock recordings to wav recording. Use as: python3 ns2wav.py /path/to/recording.ns6 /path/to/wavfile.wav")

    parser.add_argument(
        'ns_file',
        help='Input in blackrock (it should have only one channel)')
    parser.add_argument(
        'wav_file',
        help='Output wav file')
    parser.add_argument(
        '--chan', default='ainp1',
        help='Specify which channel should be converted to wav (default: ainp1)')
    args = parser.parse_args()

    if Path(args.wav_file).exists():
        raise FileExistsError(f'{args.wav_file} exists. Not overwriting')

    d = Dataset(args.ns_file)
    data = d.read_data()
    x = data(trial=0, chan=args.chan)
    x /= nanmax(abs(x))
    write(args.wav_file, data.s_freq, x)


if __name__ == '__main__':
    main()