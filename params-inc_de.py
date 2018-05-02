#--------------------------------------------------------
# language settings
#--------------------------------------------------------
src_lang = 'en'
tgt_lang = 'de'  #cs or de
pair = "en-" + tgt_lang

#--------------------------------------------------------
# model settings
#--------------------------------------------------------
fixed_seeds=True
max_num_sents = 1000000
batch_size = 60
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
src_dir = "wmt/words"
tgt_dir = "wmt/10k"
tag = "words"
suffix = ".bpe"

train_src = 'data/{}/{}/train.{}'.format(pair, src_dir, src_lang)
train_tgt = 'data/{}/{}/train.{}{}'.format(pair, tgt_dir, tgt_lang, suffix)
dev_src   = 'data/{}/{}/newstest2013.{}'.format(pair, src_dir, src_lang)
dev_tgt   = 'data/{}/{}/newstest2013.{}{}'.format(pair, tgt_dir, tgt_lang, suffix)
tst_src   = 'data/{}/{}/newstest2014.{}'.format(pair, src_dir, src_lang)
tst_tgt   = 'data/{}/{}/newstest2014.{}{}'.format(pair, tgt_dir, tgt_lang, suffix)

#--------------------------------------------------------
# incremental bpe settings (overwritten if debug arg passed to main)
#--------------------------------------------------------
use_incremental_bpe = True
inc_bpe_dir = 'data/{}/wmt'.format(pair)
burn_in_iters = 3 # Iterations to run before evaluating the loss threshold
dev_loss_threshold = 0.05 # Load new bpe splits if dev loss fails to decrease by this threshold for a number of epochs
bpe_patience = 1 # Number of iterations to allow dev loss to decrease less than the threshold
embed_merge = 'avg' # One of avg, max, ae

train_bpe_tgt = inc_bpe_dir + '/{}/' + 'train.' + tgt_lang + suffix
dev_bpe_tgt   = inc_bpe_dir + '/{}/' + 'newstest2013.' + tgt_lang + suffix
tst_bpe_tgt   = inc_bpe_dir + '/{}/' + 'newstest2014.' + tgt_lang + suffix
code_paths    = inc_bpe_dir + '/{}/' + tgt_lang + '_codes.{}'

#--------------------------------------------------------
# output settings
#--------------------------------------------------------
OUTPUT_PATH = 'output-inc-de/'
MODEL_PATH  = 'models-inc-de/'
print_every = 50
plot_every  = 50
model_every = 1  #not used w/early stopping
checkpoint_every = 50000  #for intermediate dev loss/output. set high enough to not happen
patience = 10
src_vocab = '{}src-vocab_{}_maxnum{}_maxlen{}.pkl'.format(MODEL_PATH, pair, max_num_sents, max_sent_length)
tgt_vocab = '{}tgt-vocab_{}_maxnum{}_maxlen{}.pkl'.format(MODEL_PATH, pair, max_num_sents, max_sent_length)

