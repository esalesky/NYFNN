"""read data, create vocab, and preprocess"""
import unicodedata
import regex as re

SOS = 0
EOS = 1

SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"


#todo: tokenization, bpe, iwslt xml format
#todo: import vocabs (so we don't have to do this every time for the same data) ?

class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2idx = {SOS_TOKEN : SOS, EOS_TOKEN : EOS}
        self.idx2word = [SOS_TOKEN, EOS_TOKEN]
        
        self.unk_token  = None     #default None, set below after vocab frozen
        self.vocab_frozen = False  #impt for decoding, where we should not add new vocab

    def vocab_size(self):
        return len(self.idx2word)

    # maps words to vocab idx, adds if not already present
    def map2idx(self, word):
        if word not in self.word2idx:
            if self.vocab_frozen:
                assert self.unk_token != None, 'No unk_token set but tried to map OOV word to idx with frozen vocab'
                return self.unk_token
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]

    def freeze_vocab(self):
        self.vocab_frozen = True

    def set_unk(self, unk_word):
        assert self.vocab_frozen, 'Tried to set unk with vocab not frozen'
        if unk_word not in self.word2idx:
            self.word2idx[unk_word] = len(self.idx2word)
            self.idx2word.append(unk_word)
        self.unk_token = self.word2idx[unk_word]
        

# reads parallel data where format is one sentence per line, filename prefix.lang
# expectation is file does not have SOS/EOS symbols
def read_corpus(data_prefix, src_lang, tgt_lang):
    src_file = data_prefix + "." + src_lang
    tgt_file = data_prefix + "." + tgt_lang

    src_vocab = Vocab(src_lang)
    tgt_vocab = Vocab(tgt_lang)
    
    print("Reading files...")

    # read file, create vocab, and maps words to idxs
    src_sents = []
    with open(src_file, 'r', encoding='utf-8') as f:
        line = f.readline().strip().split()
        while line:
            sent = [ src_vocab.map2idx(w) for w in line ]
            src_sents.append([SOS] + sent + [EOS])
            line = f.readline().strip().split()

    # read file, create vocab, and maps words to idxs
    tgt_sents = []
    with open(tgt_file, 'r', encoding='utf-8') as f:
        line = f.readline().strip().split()
        while line:
            sent = [ tgt_vocab.map2idx(w) for w in line ]
            tgt_sents.append([SOS] + sent + [EOS])
            line = f.readline().strip().split()

    if len(src_sents) != len(tgt_sents): raise RuntimeError(f"different number of src and tgt sentences!! {len(src_sents)} != {len(tgt_sents)}")

    src_vocab.freeze_vocab()
    src_vocab.set_unk(UNK_TOKEN)

    tgt_vocab.freeze_vocab()
    tgt_vocab.set_unk(UNK_TOKEN)

    return src_vocab, tgt_vocab, list(zip(src_sents, tgt_sents))


def input_reader(data_prefix, src, tgt, max_sent_length, max_num_sents):
    src_vocab, tgt_vocab, sents = read_corpus(data_prefix, src, tgt)
    print("Read %s sentences" % len(sents))
    sents = filter_sents(sents, max_sent_length, max_num_sents)
    print("Filtered to %s sentences" % len(sents))

    print("Vocab sizes: %s %d, %s %d" % (src_vocab.name, src_vocab.vocab_size(), tgt_vocab.name, tgt_vocab.vocab_size()))
    return src_vocab, tgt_vocab, sents


#true if sent should not be filtered
def keep_pair(p, max_sent_length):
    return len(p[0]) < max_sent_length and len(p[1]) < max_sent_length

#return sents filtered by max sent length, max num sents
def filter_sents(sents, max_sent_length, max_num_sents):
    return [pair for pair in sents[:max_num_sents] if keep_pair(pair, max_sent_length)]
