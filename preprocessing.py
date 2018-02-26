"""read data, create vocab, and preprocess"""
import unicodedata
import regex as re
import pickle

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

    def save(self, fname):
        """Save the vocabulary to a pickle file."""
        with open(fname, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)
        

# reads parallel data where format is one sentence per line, filename prefix.lang
# expectation is file does not have SOS/EOS symbols
def read_corpus(file_prefix, file_suffix, src_lang, tgt_lang, max_num_sents, src_vocab, tgt_vocab, max_sent_length, min_sent_length):
    src_file = file_prefix + "." + src_lang + file_suffix
    tgt_file = file_prefix + "." + tgt_lang + file_suffix

    if src_vocab is None:
        src_vocab = Vocab(src_lang)
    if tgt_vocab is None:
        tgt_vocab = Vocab(tgt_lang)
    
    print("Reading files... src: {}, tgt: {}".format(src_file, tgt_file))

    sents = []
    with open(src_file, encoding='utf-8') as src_sents, open(tgt_file, encoding='utf-8') as tgt_sents:
        for src_sent, tgt_sent in zip(src_sents, tgt_sents):
            src_sent = clean(src_sent)
            tgt_sent = clean(tgt_sent)
            if keep_pair((src_sent, tgt_sent), max_sent_length, min_sent_length):
                src_sent = [SOS] + [src_vocab.map2idx(w) for w in src_sent] + [EOS]
                tgt_sent = [SOS] + [tgt_vocab.map2idx(w) for w in tgt_sent] + [EOS]
                sents.append((src_sent, tgt_sent))
                if len(sents) >= max_num_sents:
                    break

    print("Read {} sentences.".format(len(sents)))
    src_vocab.freeze_vocab()
    src_vocab.set_unk(UNK_TOKEN)

    tgt_vocab.freeze_vocab()
    tgt_vocab.set_unk(UNK_TOKEN)

    return src_vocab, tgt_vocab, sents


def clean(line):
    if line.startswith('<'):  # Handles the IWSLT XML formats
        if line.startswith('<seg id'):
            line = re.sub('<[^>]*>', '', line).strip()
        else:
            line = ""
    else:
        line = line.strip()  # Supports both txt and the pseudo IWSLT XML
    return line.split()


def input_reader(file_prefix, src, tgt, max_num_sents,
                 max_sent_length=100, src_vocab=None, tgt_vocab=None, file_suffix=''):

    src_vocab, tgt_vocab, sents = read_corpus(file_prefix, file_suffix, src, tgt, max_num_sents,
                                              src_vocab, tgt_vocab, max_sent_length, min_sent_length=1)

    if src_vocab is None:
        print("Vocab sizes: %s %d, %s %d" % (src_vocab.name, src_vocab.vocab_size(),
                                             tgt_vocab.name, tgt_vocab.vocab_size()))
    return src_vocab, tgt_vocab, sents


#true if sent should not be filtered
def keep_pair(p, max_sent_length, min_sent_length):
    return min_sent_length <= len(p[min_sent_length]) <= max_sent_length and min_sent_length <= len(p[1]) <= max_sent_length
