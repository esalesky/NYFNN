import argparse
from analyze_czech import CzechMorphologyTransformer


def main(args):
    transformer = CzechMorphologyTransformer(args.dictionary, args.tagger)
    lines = [line for line in open(args.file, encoding='utf-8')]
    outfile = open(args.file + ".demorph", "w+", encoding='utf-8')
    for sent_num, line in enumerate(lines):
        decoded = transformer.morph_dec(line.strip(), sent_num)
        outfile.write(decoded + "\n")
    outfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help='File to decode')
    parser.add_argument("-y", "--dictionary", required=True, help='path to czech morphodita dict file')
    parser.add_argument("-t", "--tagger", required=True, help='path to czech morphodita parser file')

    args = parser.parse_args()
    main(args)
