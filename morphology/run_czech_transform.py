import argparse
from os import listdir
from morphology.analyze_czech import CzechMorphologyTransformer

import regex as re


def main(args):
    prefix = args.dirname  #path to dir to convert, ie data/en-{cs,de}
    files = [f for f in listdir(prefix)]
    files.remove("README")
    transformer = CzechMorphologyTransformer(args.dictionary, args.tagger)
    for f in files:
        if f.endswith(".cs.txt"):
            outfile = open(prefix + f.replace(".txt", "") + "-morph.txt", "w+", encoding='utf-8')
            convert(prefix + f, outfile, transformer)
            outfile.close()


def convert(file, output, transformer):
    print("Reading file {}".format(file))
    with open(file, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            encoded = transformer.morph_enc(line)
            output.write(encoded + "\n")
            line = f.readline()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dirname", required=True, help='data dir (eg data/en-cs)')
    parser.add_argument("-y", "--dictionary", required=True, help='path to czech morphodita dict file')
    parser.add_argument("-t", "--tagger", required=True, help='path to czech morphodita parser file')
    args = parser.parse_args()
    main(args)
