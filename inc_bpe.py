from preprocessing import input_reader, create_vocab
import logging
import torch
import pdb
import mod_embed as me
from utils import use_cuda
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)

class BPEIncrementer:

    def __init__(self, params, vocab):
        bpe_set = ["5k","10k","15k", "20k", "25k", "30k", "35k", "40k", "45k"]
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
        self._load_init(vocab)

    """Load the initial set of bpe splits into the vocab to ensure that all subwords are present for future
       combination."""
    def _load_init(self, tgt_vocab):
        tgt_vocab.thaw_vocab()
        with open(self.code_sets[self.bpe_step], 'r') as code_file:
            for line in code_file:
                line = line.strip().split()
                left, right = self._get_bpe_forms(line[0], line[1])
                tgt_vocab.map2idx(left)
                tgt_vocab.map2idx(right)
        tgt_vocab.freeze_vocab()

    """Load the next set of bpe'd sentences. This method should be called after update_bpe_vocab"""
    def load_next_bpe(self, src_vocab, tgt_vocab):
        train_sents = input_reader(self.train_src, self.tgt_train_sets[self.bpe_step], self.src_lang, self.tgt_lang,
                                   self.max_num_sents, self.max_sent_length, src_vocab, tgt_vocab, sort=True)
        dev_sents_unsorted = input_reader(self.dev_src, self.tgt_dev_sets[self.bpe_step], self.src_lang, self.tgt_lang,
                                          self.max_num_sents, self.max_sent_length, src_vocab, tgt_vocab, filt=False)
        dev_sents_sorted = input_reader(self.dev_src, self.tgt_dev_sets[self.bpe_step], self.src_lang, self.tgt_lang,
                                        self.max_num_sents, self.max_sent_length, src_vocab, tgt_vocab, sort=True, filt=False)
        tst_sents = input_reader(self.tst_src, self.tgt_tst_sets[self.bpe_step], self.src_lang, self.tgt_lang, self.max_num_sents,
                                 self.max_sent_length, src_vocab, tgt_vocab, filt=False)

        return train_sents, dev_sents_unsorted, dev_sents_sorted, tst_sents

    def update_bpe_vocab(self, model, optimizer, tgt_vocab):
        # add one to current_bpe_step
        print(tgt_vocab.vocab_size(), model.decoder.embed.weight.shape[0])
        self.bpe_step += 1
        logger.info("Moving to next bpe increment: {}".format(self.bpe_step))
        # Save the original embeddings before we modify them in any way
        original_embedding = model.decoder.embed.weight
        n_updates = 0  # Keep track of the number of updates made to the embeddings
        # unfreeze target vocab
        tgt_vocab.thaw_vocab()
        # get new vocab words
        with open(self.code_sets[self.bpe_step], 'r') as code_file:
            code_file.readline().strip()  # version header
            # Retrieve the set of previously BPE-merged subword to its component subwords
            for _ in range(self.bpe_inc * self.bpe_step):
                line = code_file.readline().strip().split()
                left, right = self._get_bpe_forms(line[0], line[1])
                # Expect that all subwords should already have been in the vocabulary
                # todo: If they aren't, we'll need to add them separately
                if left not in tgt_vocab:
                    logger.error("Word is not already present in vocab :(")
                    raise Exception
                if right not in tgt_vocab:
                    logger.error("Word is not already present in vocab :(")
                    raise Exception
                idx, idy, w = self._merge_bpe(line[0], line[1], tgt_vocab)
                prev_vocab_size = tgt_vocab.vocab_size()
                widx = tgt_vocab.map2idx(w)
                if widx != prev_vocab_size:
                    continue
                # add to embeddings
                me.update_embed(model.decoder.embed, idx, idy, operation="avg")
            # first real line
            line = code_file.readline().strip().split()
            while line:
                # add line codes to vocab and embeddings
                left, right = self._get_bpe_forms(line[0], line[1])
                # Expect that all subwords should already have been in the vocabulary
                # todo: If they aren't, we'll need to add them separately
                if left not in tgt_vocab:
                    logger.error("Word is not already present in vocab :(")
                    raise Exception
                if right not in tgt_vocab:
                    logger.error("Word is not already present in vocab :(")
                    raise Exception
                self.update_pair_embedding(line[0], line[1], tgt_vocab, model.decoder.embed, model.decoder.out)
                n_updates += 1
                line = code_file.readline().strip().split()
            print(tgt_vocab.vocab_size(), model.decoder.embed.weight.shape[0])
        self.update_optimizer(optimizer, model, original_embedding, n_updates)
        # freeze target vocab
        tgt_vocab.freeze_vocab()

    """Update the embedding layer with some combination of the embeddings of the provided two words."""
    def update_pair_embedding(self, w1, w2, vocab, embedding, output):
        idx, idy, w = self._merge_bpe(w1, w2, vocab)
        # add to vocab
        prev_vocab_size = vocab.vocab_size()
        widx = vocab.map2idx(w)
        if widx != prev_vocab_size:
            logger.error("! word already in vocab: {}".format(w))
            raise Exception
        # add to embeddings
        me.update_embed(embedding, idx, idy, operation="avg")
        me.update_linear(output, idx, idy, operation="avg")
        if embedding.weight.shape[0] != vocab.vocab_size() != output.weight.shape[0]:
            logger.error("Embedding, vocab, and linear sizes do not match! Vocab size: {}, Embeding size: {}, Output size: {}"
                         .format(vocab.vocab_size(), embedding.weight.shape[0], output.weight.shape[0]))
            raise Exception
        # print(embedding.weight[idx], embedding.weight[idy], embedding.weight[widx])

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
        print(optimizer.state[new_embedding])
        del(optimizer.state[original_embedding])

    def _merge_bpe(self, first, second, vocab):
        left = first+"@@"   #a@@
        if second.endswith("</w>"):
            right  = second.replace('</w>','') #b
            merged = first+right #ab
        else:
            right  = second+"@@" #b@@
            merged = first+right #ab@@
        return vocab.map2idx(left), vocab.map2idx(right), merged


    def _get_bpe_forms(self, first, second):
        left = first+"@@"
        if second.endswith("</w>"):
            right  = second.replace('</w>','')
        else:
            right = second+"@@"
        return left, right