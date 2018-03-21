## project code for 11-747

#### Single step postprocessing

./external_scripts/run-bleu-score.sh 
Usage: run-bleu-score.sh [-d] [-h] test ref
Options:
        -d Detokenize the test file after removing bpe splits
        -h Display this help


#### useful preprocessing/postprocessing commands

to de-xml data directory (runs on all files in a dir, currently):

    python3 data_xml_to_txt.py -d dir

to tokenize/detokenize English:

    perl external_scripts/tokenizer.perl (-l [en|cs|...]) (-threads 4) < textfile > tokenizedfile
    perl external_scripts/detokenizer.perl (-l [en|cs|...]) < tokenizedfile > detokenizedfile

to tokenize Czech (runs on all files in a dir, currently):

    python3 morphology/run_czech_transform.py -d dir -y morphodita_dict -t morphodita_parser_file

to generate bpe:

    ./subword-nmt/learn_bpe.py -s {num_operations} < {train_file} > {codes_file}
    ./subword-nmt/apply_bpe.py -c {codes_file} < {test_file}

to de-bpe:

    sed -r 's/(@@ )|(@@ ?$)//g'

to score with BLEU: 

    perl external_scripts/multi-bleu.perl -lc ref < hyp

