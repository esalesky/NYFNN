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
max_num_sents = 1000000
batch_size = 60
max_sent_length = 50  #paper: 50 for baseline, 100 for morphgen
max_gen_length  = 100 #100 for baseline, 200 for morphgen to be safe
num_epochs = 30 # Specifically set very high to allow for full set of bpe splits
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
# incremental bpe settings (overwritten if debug arg passed to main)
#--------------------------------------------------------
use_incremental_bpe = True
inc_bpe_dir = 'data/{}/bpe_tune'.format(pair)
burn_in_iters = 3 # Iterations to run before evaluating the loss threshold
dev_loss_threshold = 0.05 # Load new bpe splits if dev loss fails to decrease by this threshold for a number of epochs
bpe_patience = 1 # Number of iterations to allow dev loss to decrease less than the threshold
embed_merge = 'avg' # One of avg, max, ae

train_bpe_tgt = inc_bpe_dir + '/{}/' + 'train.tags.{}.{}.tok.bpe'.format(pair, tgt_lang)
dev_bpe_tgt   = inc_bpe_dir + '/{}/' + 'IWSLT16.TED.tst2012.{}.{}.tok.bpe'.format(pair, tgt_lang)
tst_bpe_tgt   = inc_bpe_dir + '/{}/' + 'IWSLT16.TED.tst2013.{}.{}.tok.bpe'.format(pair, tgt_lang)
code_paths    = inc_bpe_dir + '/{}/' + tgt_lang + '_codes.{}'


#--------------------------------------------------------
# data settings (overwritten if debug arg passed to main)
#--------------------------------------------------------
src_dir = "bped"
tgt_dir = "bpe_tune/20k"
src_suffix = ".tok.bpe"
tgt_suffix = ".tok.bpe"

train_src = 'data/{}/{}/train.tags.{}.{}{}'.format(pair, src_dir, pair, src_lang, src_suffix)
train_tgt = 'data/{}/{}/train.tags.{}.{}{}'.format(pair, tgt_dir, pair, tgt_lang, tgt_suffix)
dev_src   = 'data/{}/{}/IWSLT16.TED.tst2012.{}.{}{}'.format(pair, src_dir, pair, src_lang, src_suffix)
dev_tgt   = 'data/{}/{}/IWSLT16.TED.tst2012.{}.{}{}'.format(pair, tgt_dir, pair, tgt_lang, tgt_suffix)
tst_src   = 'data/{}/{}/IWSLT16.TED.tst2013.{}.{}{}'.format(pair, src_dir, pair, src_lang, src_suffix)
tst_tgt   = 'data/{}/{}/IWSLT16.TED.tst2013.{}.{}{}'.format(pair, tgt_dir, pair, tgt_lang, tgt_suffix)


#--------------------------------------------------------
# output settings
#--------------------------------------------------------
OUTPUT_PATH = 'output-shuff/'
MODEL_PATH  = 'models-shuff/'
print_every = 50
plot_every  = 50
model_every = 1  #not used w/early stopping
checkpoint_every = 50000  #for intermediate dev loss/output. set high enough to not happen
patience = 100 # Set very high so it won't kick off

src_vocab = '{}src-vocab_{}_maxnum{}_maxlen{}.pkl'.format(MODEL_PATH, pair, max_num_sents, max_sent_length)
tgt_vocab = '{}tgt-vocab_{}_maxnum{}_maxlen{}.pkl'.format(MODEL_PATH, pair, max_num_sents, max_sent_length)
