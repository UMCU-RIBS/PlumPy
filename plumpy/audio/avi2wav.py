import subprocess
import glob
import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Path')
    parser.add_argument('--path', '-p', type=str, help='Path with avi files')
    args = parser.parse_args()
    fnames = []
    datadir = args.path
    for file in glob.glob(datadir + "/*.AVI"):
        fnames.append(file)
        print(file)

    ##
    for fname in fnames:
        command = "ffmpeg -i "+ fname+ " -ab 160k -ac 2 -ar 22050 -vn " + fname.replace('.AVI', '.wav')
        subprocess.call(command, shell=True)
