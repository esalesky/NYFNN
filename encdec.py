import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.autograd import Variable
from utils import use_cuda
from preprocessing import SOS, EOS

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
        self.embedding   = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn_factory(rnn_type, input_size=embed_size, hidden_size=hidden_size, batch_first=True,
                               num_layers=num_layers, bidirectional=bidirectional)  #input_size will need to change if num_layers>1 !
        self.hidden = None

   # src is a batch of sentences
    def forward(self, src):
        embedded = self.embedding(src)  # 3D Tensor of size [batch_size x num_hist x emb_size]
        output, self.hidden = self.rnn(embedded, self.hidden)
        return output

    def save(self, fname):
        """Save the model to a pickle file."""
        with open(fname, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)


class RNNDecoder(nn.Module):
    """simple initial decoder"""
    def __init__(self, vocab_size, embed_size, hidden_size, rnn_type='GRU',  num_layers=1, bidirectional=False):
        super(RNNDecoder, self).__init__()
        self.vocab_size = vocab_size  #target vocab size
        self.hidden_size = hidden_size
        self.embedding   = nn.Embedding(vocab_size, embed_size)
        self.num_layers  = num_layers
        # nn.rnn internally makes input_size = hidden_size for >1 layer
        self.rnn = rnn_factory(rnn_type, input_size=embed_size, hidden_size=hidden_size, batch_first=True,
                               num_layers=num_layers, bidirectional=bidirectional)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)  #dim corresponding to vocab
        self.hidden = None

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
    def generate(self, init_hidden, encoder_outputs, max_gen_length):
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
    def __init__(self, hidden_size, attn_type):
        super(Attn, self).__init__()
        self.attn_type = attn_type  #bilinear, h(src)T * W * h(tgt)
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(dim=0)
        self.linear = nn.Linear(self.hidden_size, hidden_size)

    def forward(self, hidden, encoder_outputs):
        src_len = len(encoder_outputs)
        attn_weights = Variable(torch.zeros(src_len))  #dims are (batch size (1), num words (1), src len)

        if use_cuda:
            attn_weights = attn_weights.cuda()

        # calculate attn_score against each output in encoder_outputs
        # todo: make more efficient by computing attention score once over the concatenated matrix H(src)
        for i in range(src_len):
            attn_score = self.linear(encoder_outputs[i])  #h(src)T * W
            attn_weights[i] = hidden.dot(attn_score)    #[h(src)T*W] * h(tgt)

        # normalize weights and resize to 1 x 1 x src len
        return self.softmax(attn_weights).unsqueeze(0).unsqueeze(0)


class AttnDecoder(nn.Module):
    """attention layer on top of basic decoder"""
    def __init__(self, vocab_size, embed_size, hidden_size, rnn_type='GRU', num_layers=1, bidirectional=False):
        super(AttnDecoder, self).__init__()
        self.vocab_size = vocab_size  #target vocab size
        self.hidden_size = hidden_size
        #self.max_src_length = max_src_length  #todo: preallocate mem for longest src sent? would this actually be helpful?
        self.num_layers  = num_layers

        self.attn = Attn(hidden_size, attn_type="bilinear")

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn_factory(rnn_type, input_size=embed_size+hidden_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)  #nn.rnn internally makes input_size=hidden_size for >1 layer
        self.out = nn.Linear(2*hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)  #dim corresponding to vocab
        self.hidden = self.init_hidden()

    # Generates sequence, up to tgt_len, conditioned on the initial hidden state
    def forward(self, init_hidden, encoder_outputs, tgt_len, generate=False):
        self.hidden = init_hidden  #init hidden with last encoder hidden
        decoder_context = Variable(torch.zeros(1, self.hidden_size))

        decoder_input = Variable(torch.LongTensor([[SOS]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        outputs = []
        words = []
        for _ in range(tgt_len): #todo: unless teacher forcing, shouldn't this just be until EOS?
            decoder_output, decoder_context, attn_weights = self.__forward_one_word(decoder_input, decoder_context, encoder_outputs)
            _, top_idx = decoder_output.data.topk(1)
            # Recover the value of the top index and wrap in a new variable to break backprop chain
            word_idx = top_idx[0][0]
            decoder_input = Variable(torch.LongTensor([[word_idx]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            if generate:
                words.append(word_idx)
            outputs.append(decoder_output)
            if word_idx == EOS:
                break

        return outputs, words

    # Passes a single word through the decoder network
    def __forward_one_word(self, tgt_word, prev_context, encoder_outputs):
        embedded = self.embedding(tgt_word).view(1, 1, -1)  #dims are (num words (1), batch size (1), emb size)

        rnn_input = torch.cat((embedded, prev_context.unsqueeze(0)), 2)  #concat tgt seed word + context to input to rnn
        rnn_output, self.hidden = self.rnn(rnn_input, self.hidden)
        rnn_output = rnn_output.squeeze(0)  #collapse num words dim

        # calculate attn against each encoder outputs
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  #swaps encoder dims for multiply: src_len x batch_size x hidden_size -> batch_size x src_len x hidden_size
        context = context.squeeze(1)  #collapse num words dim

        # predict next word using rnn output and context vector
        output = torch.cat((rnn_output, context), 1)  #concat rnn output + context to input to output layer
        output = self.out(output)
        output = self.softmax(output)

        return output, context, attn_weights

    #Initialize hidden state to a pair of variables for context/hidden
    def init_hidden(self):
        result = (Variable(torch.zeros(1, 1, self.hidden_size)),
                  Variable(torch.zeros(1, 1, self.hidden_size)))
        if use_cuda:
            return result[0].cuda(), result[1].cuda()
        else:
            return result

    def save(self, fname):
        """Save the model to a pickle file."""
        with open(fname, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)


class EncDec(nn.Module):
    """initial encoder + decoder model"""
    def __init__(self, encoder, decoder):
        super(EncDec, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        # TODO tgt.shape[0] may be wrong in this call below
        self.encoder.hidden = None  # self.encoder.init_hidden(batch_size)
        encoder_outputs = self.encoder(src)
        decoder_outputs = self.decoder(self.encoder.hidden, encoder_outputs, tgt)
        return decoder_outputs

    def generate(self, src, max_length):
        self.encoder.hidden = None  # self.encoder.init_hidden(batch_size)
        encoder_outputs = self.encoder(src)
        words = self.decoder.generate(self.encoder.hidden, encoder_outputs, max_gen_length=max_length)
        return words


    def save(self, fname):
        """Save the model to a pickle file."""
        logger.info("Saving at: {}".format(fname))
        with open(fname, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)
