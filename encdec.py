import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.autograd import Variable
from utils import use_cuda
from preprocessing import SOS, EOS
from beam_search import Beam

import logging
logger = logging.getLogger(__name__)


def rnn_factory(rnn_type, **kwargs):
    assert rnn_type in ['LSTM','GRU'], 'rnn_type not one of currently supported options'
    rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn


class RNNEncoder(nn.Module):
    """simple initial encoder"""
    def __init__(self, vocab_size, embed_size, hidden_size, rnn_type='GRU', num_layers=1, bidirectional=False):
        super(RNNEncoder, self).__init__()
        self.vocab_size  = vocab_size  #source vocab size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # The layers of the NN
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed_linear = nn.Linear(embed_size, hidden_size)
        self.rnn = rnn_factory(rnn_type, input_size=hidden_size, hidden_size=hidden_size, batch_first=True,
                               num_layers=num_layers, bidirectional=bidirectional)
        self.hidden = None

    def embedding(self, src):
        embedded = self.embed(src)
        output   = self.embed_linear(embedded)
        return output
        
   # src is a batch of sentences
    def forward(self, src):
        embedded = self.embedding(src)  # 3D Tensor of size [batch_size x num_hist x emb_size]
        output, self.hidden = self.rnn(embedded, self.hidden)
        return output

    def save(self, fname):
        """Save the model using pytorch's format"""
        logger.info("Saving at: {}".format(fname))
        torch.save(self, fname)


class RNNDecoder(nn.Module):
    """simple initial decoder"""
    def __init__(self, vocab_size, embed_size, hidden_size, rnn_type='GRU',  num_layers=1, bidirectional=False):
        super(RNNDecoder, self).__init__()
        self.vocab_size = vocab_size  #target vocab size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed_linear = nn.Linear(embed_size, hidden_size)
        self.num_layers  = num_layers
        # nn.rnn internally makes input_size = hidden_size for >1 layer
        self.rnn = rnn_factory(rnn_type, input_size=hidden_size, hidden_size=hidden_size, batch_first=True,
                               num_layers=num_layers, bidirectional=bidirectional)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)  #dim corresponding to vocab
        self.hidden = None

    def embedding(self, src):
        embedded = self.embed(src)
        output   = self.embed_linear(embedded)
        return output

    # Performs forward pass for a batch of sentences through the decoder using teacher forcing.
    # Note that this decoder does not use the encoder_outputs at all
    def forward(self, init_hidden, encoder_outputs, tgt):
        self.hidden = init_hidden # init hidden state with last encoder hidden
        # The hidden state in RNNs in Pytorch is always (seq_length, batch_size, emb_size) - even if you use batch_first
        batch_size = init_hidden.shape[1]
        decoder_input = Variable(torch.LongTensor(batch_size * [[SOS]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        outputs = []
        tgt_len = tgt.shape[1]
        tgt = tgt.transpose(0, 1)
        for i in range(tgt_len):
            decoder_outputs = self.__forward_one_word(decoder_input)
            _, top_idx = decoder_outputs.data.topk(1)
            # todo: potentially make teacher forcing optional/stochastic
            decoder_input = tgt[i]
            outputs.append(decoder_outputs.squeeze(1))
        return outputs

    # Generates entire sequence, up to max_gen_length, conditioned on initial hidden state.
    # Note that this decoder does not use the encoder outputs at all
    def generate(self, init_hidden, encoder_outputs, max_gen_length, beam_size):
        self.hidden = init_hidden # init hidden state with last encoder hidden
        # The hidden state in RNNs in Pytorch is always (seq_length, batch_size, emb_size) - even if you use batch_first
        # Note that during generation, the batch size should always be 1
        decoder_input = Variable(torch.LongTensor(init_hidden.shape[1] * [[SOS]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        outputs = []
        words = []
        for _ in range(max_gen_length):
            decoder_outputs = self.__forward_one_word(decoder_input)
            _, top_idx = decoder_outputs.data.topk(1)
            # Recover the value of the top index and wrap in a new variable to break backprop chain
            word_idx = top_idx.cpu().numpy()
            # Results in a new decoder input of dims (batch_size, 1)
            decoder_input = Variable(torch.LongTensor(torch.from_numpy(word_idx))).squeeze(2)
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            outputs.append(decoder_outputs.squeeze(1))
            words.append(word_idx[0][0][0])
            if word_idx == EOS:
                break

        return outputs, words

    # Passes a single word through the decoder network
    def __forward_one_word(self, tgt):
        # Map to embeddings - dims are (batch size, seq length, emb size)
        batch_size = tgt.shape[0]
        output = self.embedding(tgt).view(batch_size, 1, -1)
        # Non-linear activation over embeddings
        output = F.tanh(output)
        # Pass representation through RNN
        output, self.hidden = self.rnn(output, self.hidden)
        # Softmax over the final output state
        output = self.softmax(self.out(output))

        return output

    def save(self, fname):
        """Save the model to a pickle file."""
        with open(fname, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)


class Attn(nn.Module):
    def __init__(self, input_size, hidden_size, attn_type):
        super(Attn, self).__init__()
        self.attn_type = attn_type  #bilinear, h(src)T * W * h(tgt)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(self.input_size, self.hidden_size)

    def forward(self, hidden, attn_scores):
        batch_size = attn_scores.shape[0]

        # calculate attn_score against each output in encoder_outputs
        attn_weights = attn_scores.bmm(hidden.view(batch_size, -1, 1)).squeeze(2)

        # Resulting dims are (batch size, seq len)
        attn_weights = self.softmax(attn_weights)
        # normalize weights and resize to batch_size x 1 x src len
        return attn_weights.unsqueeze(1)

    def calc_attn_scores(self, encoder_outputs):
        return self.linear(encoder_outputs)


class AttnDecoder(nn.Module):
    """attention layer on top of basic decoder"""
    def __init__(self, enc_size, vocab_size, embed_size, hidden_size, rnn_type='GRU',
                 num_layers=1, bidirectional_enc=False, tgt_vocab=None):
        super(AttnDecoder, self).__init__()
        self.vocab_size = vocab_size  #target vocab size
        self.hidden_size = hidden_size
        #self.max_src_length = max_src_length  #todo: preallocate mem for longest src sent? would this actually be helpful?
        self.num_layers  = num_layers
        self.bidirectional_enc = bidirectional_enc
        self.enc_size = enc_size
        if bidirectional_enc:
            self.enc_size = enc_size * 2
        self.attn = Attn(self.enc_size, hidden_size, attn_type="bilinear")

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed_linear = nn.Linear(embed_size, hidden_size)
        self.linear = nn.Linear(enc_size, hidden_size)
        # nn.rnn internally makes input_size=hidden_size for >1 layer
        self.rnn = rnn_factory(rnn_type, input_size=2*hidden_size, hidden_size=hidden_size, batch_first=True,
                               num_layers=num_layers, bidirectional=False)
        self.out = nn.Linear(2*hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)  #dim corresponding to vocab
        self.hidden = None
        # TODO: remove saving the tgt vocab in this class
        self.tgt_vocab = tgt_vocab

    def embedding(self, src):
        embedded = self.embed(src)
        output   = self.embed_linear(embedded)
        return output

    # Generates sequence, up to tgt_len, conditioned on the initial hidden state
    def forward(self, init_hidden, encoder_outputs, tgt, generate=False):
        # The hidden state in RNNs in Pytorch is always (seq_length, batch_size, emb_size) - even if you use batch_first
        batch_size = init_hidden.shape[1]
        if self.bidirectional_enc:
            self.hidden = torch.cat((init_hidden[0], init_hidden[1]), 1).view(1, batch_size, -1)
        else:
            self.hidden = init_hidden
        decoder_input = Variable(torch.LongTensor(batch_size * [[SOS]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        decoder_contexts = Variable(torch.zeros(batch_size, 1, self.hidden_size))
        decoder_contexts = decoder_contexts.cuda() if use_cuda else decoder_contexts
        outputs = []
        tgt_len = tgt.shape[1]
        tgt = tgt.transpose(0, 1)
        attn_scores = self.attn.calc_attn_scores(encoder_outputs)
        for i in range(tgt_len):
            decoder_outputs, decoder_contexts, attn_weights = self.__forward_one_word(decoder_input, decoder_contexts,
                                                                                      encoder_outputs, attn_scores)
            # todo: potentially make teacher forcing optional/stochastic
            decoder_input = tgt[i]
            outputs.append(decoder_outputs.squeeze(1))

        return outputs

    # Generates entire sequence, up to max_gen_length, conditioned on initial hidden state.
    def generate(self, init_hidden, encoder_outputs, max_gen_length, beam_size=1):
        # The hidden state in RNNs in Pytorch is always (seq_length, batch_size, emb_size) - even if you use batch_first
        # Note that during generation, the batch size should always be 1
        if self.bidirectional_enc:
            self.hidden = torch.cat((init_hidden[0], init_hidden[1]), 1).view(1, 1, -1)
        else:
            self.hidden = init_hidden
        decoder_contexts = Variable(torch.zeros(1, 1, self.hidden_size))
        decoder_contexts = decoder_contexts.cuda() if use_cuda else decoder_contexts
        attn_scores = self.attn.calc_attn_scores(encoder_outputs)

        #Accumulate the output scores and words generated by the model
        beam = Beam(beam_size)
        beam.add_initial_path(decoder_contexts)
        for _ in range(max_gen_length):

            # Get paths up front since the dict changes size while iterating
            all_paths = [p for p in beam]
            for path in all_paths:
                if path[-1] == EOS:
                    pass # Keep the current path as a candidate beam path
                else:
                    decoder_input, decoder_contexts = beam.get_decoder_params(path)
                    decoder_outputs, decoder_contexts, attn_weights = self.__forward_one_word(
                        decoder_input, decoder_contexts, encoder_outputs, attn_scores)
                    # Add the potential next steps for the current beam path
                    beam.add_paths(path, decoder_outputs, decoder_contexts, attn_weights)
            beam.select_best_paths()

            # Break if all beam paths have ended
            if beam.is_ended():
                break

        outputs, words, attn_weights_matrix = beam.get_best_path_results()
        return outputs, words, attn_weights_matrix

        # return outputs, words, torch.cat(attn_weights_matrix, dim=1)

    # Passes a single word through the decoder network
    def __forward_one_word(self, tgt_word, prev_context, encoder_outputs, attn_scores):

        batch_size = tgt_word.shape[0]
        embedded = self.embedding(tgt_word).view(batch_size, 1, -1)  #dims are (batch size, num words (1), emb size)

        rnn_input = torch.cat((embedded, prev_context), 2)  #concat tgt seed word + context to input to rnn
        rnn_output, self.hidden = self.rnn(rnn_input, self.hidden)

        # calculate attn against each encoder outputs
        attn_weights = self.attn(rnn_output, attn_scores)

        context = attn_weights.bmm(encoder_outputs)  # src_len x batch_size x hidden_size -> batch_size x src_len x hidden_size

        # predict next word using rnn output and context vector
        output = torch.cat((rnn_output, context), 2)  #concat rnn output + context to input to output layer
        output = self.out(output)
        output = self.softmax(output)

        return output, context, attn_weights

    def save(self, fname):
        """Save the model using pytorch's format"""
        logger.info("Saving at: {}".format(fname))
        torch.save(self, fname)


class EncDec(nn.Module):
    """initial encoder + decoder model"""
    def __init__(self, encoder, decoder):
        super(EncDec, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        self.encoder.hidden = None  # self.encoder.init_hidden(batch_size)
        encoder_outputs = self.encoder(src)
        decoder_outputs = self.decoder(self.encoder.hidden, encoder_outputs, tgt)
        return decoder_outputs

    def generate(self, src, max_length, beam_size):
        self.encoder.hidden = None  # self.encoder.init_hidden(batch_size)
        encoder_outputs = self.encoder(src)
        outputs, words, attn = self.decoder.generate(self.encoder.hidden, encoder_outputs, max_gen_length=max_length,
                                                     beam_size=beam_size)
        return outputs, words, attn


    def save(self, fname):
        """Save the model using pytorch's format"""
        logger.info("Saving at: {}".format(fname))
        torch.save(self, fname)
