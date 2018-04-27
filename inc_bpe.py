from preprocessing import input_reader, create_vocab
import logging
import torch
import pdb
import mod_embed as me
from utils import use_cuda


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
    def load_next_bpe(self, model, optimizer, src_vocab, tgt_vocab):
        print(tgt_vocab.map2idx('&a@@'))
        # add one to current_bpe_step
        self.bpe_step += 1
        logger.info("Moving to next bpe increment: {}".format(self.bpe_step))
        # Save the original embeddings before we modify them in any way
        original_embedding = model.decoder.embed.weight
        n_updates = 0  # Keep track of the number of updates made to the embeddings
        pdb.set_trace()
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
                n_updates += 1
                # add line codes to vocab and embeddings
                # print(line[0], line[1])
                self.update_embeddings(line[0], line[1], tgt_vocab, model.decoder.embed, past_codes)
                line = code_file.readline().strip().split()
        # Update the optimizer
        pdb.set_trace()
        self.update_optimizer(optimizer, model, original_embedding, n_updates)
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

    def update_optimizer(self, optimizer, model, original_embedding, n_updates, fill_val=0.0):
        """Update the optimizer with new embeddings."""
        # Re-initialize all the param groups with the new model params
        # We just do this over all params for certainty
        param_groups = list(model.parameters())        
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        optimizer.param_groups = []
        for param_group in param_groups:
            optimizer.add_param_group(param_group)

        # Get the shape difference for padding new tensors
        new_embedding = model.decoder.embed.weight
        original_shape = original_embedding.shape
        new_shape = model.decoder.embed.weight.shape
        pad_shape = torch.Size([new_shape[0] - original_shape[0], new_shape[1]])

        # Modify the state tensors to the new size
        original_state = optimizer.state[original_embedding]
        new_state_pad = torch.zeros(pad_shape)
        if use_cuda:
            new_state_pad = new_state_pad.cuda()

        new_exp_avg = torch.cat((original_state['exp_avg'], new_state_pad), 0)
        new_exp_avg_sq = torch.cat((original_state['exp_avg_sq'], new_state_pad), 0)
        new_step = original_state['step']
        new_state = {'step': new_step, 'exp_avg': new_exp_avg, 'exp_avg_sq': new_exp_avg_sq}

        # Actually modify the state
        optimizer.state[new_embedding] = new_state
        del(optimizer.state[original_embedding])