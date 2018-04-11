#!/bin/bash

usage() { echo "Usage: $0 [-s <num_splits, eg 500>] [-l <english|czech>]" 1>&2; exit 1; }

while getopts ":s:l:" o; do
    case "${o}" in
        s)  NUM_SPLITS=${OPTARG};;
        l)  LNG=${OPTARG};;
        *)  usage;;
    esac
done
shift $((OPTIND-1))

if [ -z "${NUM_SPLITS}" ] || [ -z "${LNG}" ]; then
    usage
fi


#base to keep separate from morphgen data
OUTDIR=./data/en-cs/bped_base_$NUM_SPLITS

#set language abbreviation
if [[ "$LNG" == "english" ]]; then
    LG="en"
else
    LG="cs"
fi

#make outdir if doesn't yet exist
if [[ ! -e $OUTDIR ]]; then
    mkdir $OUTDIR
fi

./subword-nmt/learn_bpe.py -s $NUM_SPLITS < ./data/en-cs/tokenized-$LNG/train.tags.en-cs.$LG.tok.txt > $OUTDIR/${LG}_codes.$NUM_SPLITS

for X in ./data/en-cs/tokenized-$LNG/*; do FILE=${X##*/}; ./subword-nmt/apply_bpe.py -c $OUTDIR/${LG}_codes.$NUM_SPLITS -i $X -o $OUTDIR/${FILE%.txt}.bpe; done
