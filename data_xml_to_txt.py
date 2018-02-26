import regex as re
from preprocessing import Vocab, SOS, EOS
import argparse

from os import listdir

def main(args):
    prefix = args.dirname  #path to dir to convert, ie data/en-{cs,de}
    files = [f for f in listdir(prefix)]
    files.remove("README")
    for f in files:
        if f.endswith(".txt"):
            continue
        if f == 'train.de' or f == 'train.cs':
            continue
        outfile = open(prefix + f.replace(".xml", "") + ".txt", "w+", encoding='utf-8')
        convert(prefix + f, outfile)
        outfile.close()


def convert(file, output):
    print("Reading files...")
    with open(file, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            if line.startswith('<'):
                if line.startswith('<seg id'):
                    line = re.sub('<[^>]*>', '', line).strip()
                    output.write(line + "\n")
            else:
                output.write(line)
            line = f.readline()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dirname", required=True, help='data dir (eg data/en-cs)')
    args = parser.parse_args()
    main(args)
