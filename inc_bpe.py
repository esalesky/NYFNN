from preprocessing import input_reader, create_vocab
import logging
import mod_embed as me


logger = logging.getLogger(__name__)

class BPEIncrementer:

    def __init__(self, params):
        bpe_set = ["5k","10k","15k"]
        self.code_sets = ['{}/{}/cs_codes.{}'.format(params.inc_bpe_dir, x, x) for x in bpe_set]
        self.tgt_train_sets = ['{}/{}/train.tags.{}.{}.tok.bpe'
                                   .format(params.inc_bpe_dir, x, params.pair, params.tgt_lang) for x in bpe_set]
        self.tgt_dev_sets = ['{}/{}/IWSLT16.TED.tst2012.{}.{}.tok.bpe'
                                 .format(params.inc_bpe_dir, x, params.pair, params.tgt_lang) for x in bpe_set]
        self.tgt_tst_sets = ['{}/{}/IWSLT16.TED.tst2013.{}.{}.tok.bpe'
                                 .format(params.inc_bpe_dir, x, params.pair, params.tgt_lang) for x in bpe_set]
        self.bpe_step = 0
        self.bpe_inc  = 5000
        self.src_lang = params.src_lang
        self.tgt_lang = params.tgt_lang
        self.train_src = params.train_src
        self.dev_src = params.dev_src
        self.tst_src = params.tst_src
        self.max_num_sents = params.max_num_sents
        self.max_sent_length = params.max_sent_length


    def _load_sentences(self, src_vocab, tgt_vocab):
        train_sents = input_reader(self.train_src, self.tgt_train_sets[self.bpe_step], self.src_lang, self.tgt_lang,
                                   self.max_num_sents, self.max_sent_length, src_vocab, tgt_vocab, sort=True)
        dev_sents_unsorted = input_reader(self.dev_src, self.tgt_dev_sets[self.bpe_step], self.src_lang, self.tgt_lang,
                                          self.max_num_sents, self.max_sent_length, src_vocab, tgt_vocab, filt=False)
        dev_sents_sorted = input_reader(self.dev_src, self.tgt_dev_sets[self.bpe_step], self.src_lang, self.tgt_lang,
                                        self.max_num_sents, self.max_sent_length, src_vocab, tgt_vocab, sort=True, filt=False)
        tst_sents = input_reader(self.tst_src, self.tgt_tst_sets[self.bpe_step], self.src_lang, self.tgt_lang, self.max_num_sents,
                                 self.max_sent_length, src_vocab, tgt_vocab, filt=False)

        return train_sents, dev_sents_unsorted, dev_sents_sorted, tst_sents

    # todo: add to optimizer
    def load_next_bpe(self, model, src_vocab, tgt_vocab):
        print(tgt_vocab.map2idx('&a@@'))
        # add one to current_bpe_step
        self.bpe_step += 1
        logger.info("Moving to next bpe increment: {}".format(self.bpe_step))
        # unfreeze target vocab
        tgt_vocab.thaw_vocab()
        # get new vocab words
        with open(self.code_sets[self.bpe_step], 'r') as code_file:
            code_file.readline().strip()  # version header
            # skip lines in current vocab (prev bpe set)
            past_codes = {}
            for _ in range(self.bpe_inc * self.bpe_step):
                line = code_file.readline().strip()
                combined = line.replace(' ', '')
                if combined in past_codes:
                    print('combined codes are in dictionary multiple times, should this happen?')
                    raise Exception
                past_codes[combined] = line
            print("Number of past codes: {}".format(len(past_codes)))
            print("Vocabulary size: {}".format(tgt_vocab.vocab_size()))
            # first real line
            line = code_file.readline().strip().split()
            while line:
                # add line codes to vocab and embeddings
                # print(line[0], line[1])
                self.update_embeddings(line[0], line[1], tgt_vocab, model.decoder.embed, past_codes)
                line = code_file.readline().strip().split()
        # freeze target vocab
        tgt_vocab.freeze_vocab()
        return self._load_sentences(src_vocab, tgt_vocab)

    def update_embeddings(self, w1, w2, vocab, embedding, past_codes):
        w1_bpe, w2_bpe = me.get_bpe_forms(w1, w2)
        if w1_bpe == '&a@@':
            print(vocab.vocab_size())
            print("AAAAAAH")
            print(vocab.map2idx('&a@@'))
        print('({}, {})'.format(w1_bpe, w2_bpe))
        if w1_bpe not in vocab:
            if w1 in past_codes:
                print('{} not in vocab, but {} is in past codes.'.format(w1_bpe, w1))
                past_code = past_codes[w1].split(' ')
                self.update_embeddings(past_code[0], past_code[1], vocab, embedding, past_codes)
            # self.find_splits(line[0], tgt_vocab, model.decoder.embed)
        if w2_bpe not in vocab:
            if w2 in past_codes:
                print('{} not in vocab, but {} is in past codes.'.format(w2_bpe, w2))
                past_code = past_codes[w2].split(' ')
                self.update_embeddings(past_code[0], past_code[1], vocab, embedding, past_codes)
            # self.find_splits(line[1], tgt_vocab, model.decoder.embed)
        self.update_pair_embedding(w1, w2, vocab, embedding)

    """Update the embedding layer with some combination of the embeddings of the provided two words."""
    def update_pair_embedding(self, w1, w2, vocab, embedding):
        print(w1, w2)
        idx, idy, w = me.merge_bpe(w1, w2, vocab)
        # add to vocab
        prev_vocab_size = vocab.vocab_size()
        widx = vocab.map2idx(w)
        if widx != prev_vocab_size:
            logger.error("! word already in vocab: {}".format(w))
            raise Exception
        # add to embeddings
        me.update_embed(embedding, idx, idy, operation="avg")
        print(embedding)
