#--------------------------------------------------------
# language settings
#--------------------------------------------------------
src_lang = 'en'
tgt_lang = 'cs'  #cs or de
pair = "en-" + tgt_lang

#--------------------------------------------------------
# model settings
#--------------------------------------------------------
fixed_seeds=True
batch_size = 60
max_num_sents = 1000000
max_sent_length = 50  #paper: 50 for baseline, 100 for morphgen
max_gen_length  = 100 #100 for baseline, 200 for morphgen to be safe
num_epochs = 30
beam_size  = 5
bi_enc = True
cond_gru_dec = True
embed_size = 500      #paper: 500
# Encoder and decoder hidden size must change together
enc_hidden_size = 1024
dec_hidden_size = 1024  #paper: 1024
if bi_enc:
    enc_hidden_size = int(enc_hidden_size / 2)

#--------------------------------------------------------
# data settings (overwritten if debug arg passed to main)
#--------------------------------------------------------
src_dir = "wmt/bped_base_50000"
tgt_dir = "wmt/bped_base_10000"
#tgt_dir = "morphgen_bpe"
src_suffix = ".tok.bpe"
tgt_suffix = ".tok.bpe"
#tgt_suffix = "-morph.bpe"

train_src = 'data/{}/{}/train1M.{}{}'.format(pair, src_dir, src_lang, src_suffix)
train_tgt = 'data/{}/{}/train1M.{}{}'.format(pair, tgt_dir, tgt_lang, tgt_suffix)
dev_src   = 'data/{}/{}/newstest2013.{}{}'.format(pair, src_dir, src_lang, src_suffix)
dev_tgt   = 'data/{}/{}/newstest2013.{}{}'.format(pair, tgt_dir, tgt_lang, tgt_suffix)
tst_src   = 'data/{}/{}/newstest2014.{}{}'.format(pair, src_dir, src_lang, src_suffix)
tst_tgt   = 'data/{}/{}/newstest2014.{}{}'.format(pair, tgt_dir, tgt_lang, tgt_suffix)

#--------------------------------------------------------
# output settings
#--------------------------------------------------------
OUTPUT_PATH = 'output-cz-1m/'
MODEL_PATH  = 'models-cz-1m/'
print_every = 50
plot_every  = 50
model_every = 1  #not used w/early stopping
checkpoint_every = 50000  #for intermediate dev loss/output. set high enough to not happen
patience = 10

src_vocab = '{}src-vocab_{}_maxnum{}_maxlen{}.pkl'.format(MODEL_PATH, pair, max_num_sents, max_sent_length)
tgt_vocab = '{}tgt-vocab_{}_maxnum{}_maxlen{}.pkl'.format(MODEL_PATH, pair, max_num_sents, max_sent_length)

#--------------------------------------------------------
# incremental bpe settings (overwritten if debug arg passed to main)
#--------------------------------------------------------
use_incremental_bpe = False
