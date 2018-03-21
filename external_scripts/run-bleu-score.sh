#!/bin/bash

#How to know directory a bash script is installed in.
#https://stackoverflow.com/questions/59895/getting-the-source-directory-of-a-bash-script-from-within?rq=1
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

# From Google shell style guide for handling flags
detok=false
while getopts ':dh' flag; do
  case "${flag}" in
    d) detok=true; shift;;
	h) printf "Usage: run-bleu-score.sh [-d] [-h] test ref\r\nOptions:\r\n\t-d Detokenize the test file after removing bpe splits\r\n\t-h Display this help"; exit 1;;
    \?) printf "Unknown flag: ${flag}"; exit 1;;
  esac
done

echo $DIR
if [ $# -lt 2 ]
then
	echo 'Usage: run-bleu-score.sh infile ref'
	exit 1
fi

DEBPE_FILE="$1.debpe"
REF_FILE=$2

echo $DEBPE_FILE
echo $REF_FILE

sed -r 's/(@@ )|(@@ ?$)//g' $1 > $DEBPE_FILE
if [ "$detok" = "true" ]
then
	DETOK_FILE="$DEBPE_FILE.detok"
	perl $DIR/detokenizer.perl -l cs < $DEBPE_FILE > $DETOK_FILE
	perl $DIR/multi-bleu.perl -lc $2 < $DETOK_FILE
else
	perl $DIR/multi-bleu.perl -lc $2 < $DEBPE_FILE
fi
