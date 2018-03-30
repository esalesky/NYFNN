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
MORPHDIR=$DIR/../morphology

# From Google shell style guide for handling flags
detok=false
while getopts ':dh' flag; do
  case "${flag}" in
    d) detok=true; shift;;
	h) printf "Usage: run-bleu-score.sh [-d] [-h] output ref.txt ref.xml src.xml tgt_lang
                  Options:
                  \t-d Detokenize the test file after removing bpe splits
                  \t-h Display this help
                  Note: tgt_lang should be written out, eg czech\n"; exit 1;;
    \?) printf "Unknown flag: ${flag}"; exit 1;;
  esac
done



echo $DIR
if [ $# -lt 5 ]
then
	echo 'Usage: run-bleu-score.sh outputfile ref.txt ref.xml src.xml tgt_lang'
	exit 1
fi

DEBPE_FILE="$1.debpe"
REF_FILE=$2

echo $DEBPE_FILE
echo $REF_FILE

gsed -r 's/(@@ )|(@@ ?$)//g' $1 > $DEBPE_FILE
DEMORPH_FILE="$DEBPE_FILE.demorph"
python3 $MORPHDIR/czech_decode.py -f $DEBPE_FILE -y $MORPHDIR/czech-morfflex-161115.dict -t $MORPHDIR/czech-morfflex-pdt-161115.tagger

if [ "$detok" = "true" ]
then
    DETOK_FILE="$DEMORPH_FILE.detok"
    perl $DIR/detokenizer.perl -l cs < $DEMORPH_FILE > $DETOK_FILE
    perl $DIR/multi-bleu.perl -lc $2 < $DETOK_FILE
else
    perl $DIR/multi-bleu.perl -lc $2 < $DEMORPH_FILE
fi

#-----------------------------------------------

REF_XML=$3
SRC_XML=$4
LANG=$5
SYSTEM_ID="system_id"
TGT_FILE=""
if [ "$detok" = "true" ]
then
    TGT_FILE="$DEMORPH_FILE.detok"
else
    TGT_FILE="$DEMORPH_FILE"
fi
TGT_XML="$TGT_FILE.sgm"

#example formats:
#~/747/external_scripts/wrap-xml.perl czech ~/747/data/en-cs/IWSLT16.TED.tst2012.en-cs.en.xml init < dev_output_e29.txt.debpe.detok > dev29.debpe.detok.sgm
#~/747/external_scripts/mteval-v13a.pl -r ~/747/data/en-cs/IWSLT16.TED.tst2012.en-cs.cs.xml -s ~/747/data/en-cs/IWSLT16.TED.tst2012.en-cs.en.xml -t dev29.debpe.detok.sgm

echo ""
echo "scoring file $TGT_XML with mteval-v13a:"
echo ""
perl $DIR/wrap-xml.perl $LANG $SRC_XML $SYSTEM_ID < $TGT_FILE > $TGT_XML
perl $DIR/mteval-v13a.pl -r $REF_XML -s $SRC_XML -t $TGT_XML
