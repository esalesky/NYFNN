import argparse
from os import listdir
from morphology.analyze_czech import CzechMorphologyTransformer

import regex as re


def main(args):
    prefix = args.dirname  #path to dir to convert, ie data/en-{cs,de}
    files = [f for f in listdir(prefix)]
    files.remove("README")
    transformer = CzechMorphologyTransformer(args.dictionary, args.tagger, mode=args.mode)
    if args.mode == 'tokenize':
        file_suffix = '-tokenized.txt'
    elif args.mode == 'lemma':
        file_suffix = '-lemma.txt'
    elif args.mode == 'tags-only':
        file_suffix = '-tags.txt'
    else:
        file_suffix = '-morph.txt'
    # file_suffix = '-tokenized.txt' if args.mode == 'tokenize' else '-morph.txt'
    for f in files:
        if f.endswith(".cs.txt"):
            outfile = open(prefix + f.replace(".txt", "") + file_suffix, "w+", encoding='utf-8')
            if args.mode == 'tags-only':
                get_tags(prefix + f, outfile, transformer)
            else:
                convert(prefix + f, outfile, transformer)
            outfile.close()


def get_tags(file, output, transformer):
    print("Reading file {}".format(file))
    file_tags = set()
    with open(file, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            tags = transformer.get_forms(line)
            for t in tags:
                file_tags.add(t)
            line = f.readline()
    for t in file_tags:
        output.write(t + "\n")


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
    parser.add_argument("-m", "--mode", required=True, help='Mode (tokenize or morph)')

    args = parser.parse_args()
    main(args)
