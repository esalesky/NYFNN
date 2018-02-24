import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.autograd import Variable
from utils import use_cuda
from preprocessing import SOS, EOS

#todo: minibatching

def rnn_factory(rnn_type, **kwargs):
    assert rnn_type in ['LSTM','GRU'], 'rnn_type not one of currently supported options'
    rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn

class RNNEncoder(nn.Module):
    """simple initial encoder"""
    def __init__(self, vocab_size, embed_size, hidden_size, rnn_type='LSTM', num_layers=1, bidirectional=False):
        super(RNNEncoder, self).__init__()
        self.vocab_size  = vocab_size  #source vocab size
        self.hidden_size = hidden_size
        self.embedding   = nn.Embedding(vocab_size, embed_size)  #currently embed_size = hidden_size
        self.num_layers  = num_layers
        self.rnn = rnn_factory(rnn_type, input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)  #requires embed_size = hidden_size
        self.hidden = self.init_hidden()

    #src is currently single sentence
    def forward(self, src):
        embedded = self.embedding(src)
        output, self.hidden = self.rnn(embedded.view(len(src), 1, -1), self.hidden)
        return output, self.hidden

    def init_hidden(self):
        #dimensions: num_layers, minibatch_size, hidden_dim
        # LSTM needs one variable, GRU needs two - defaults take care of it
        # result = (Variable(torch.zeros(1, 1, self.hidden_size)),
        #                    Variable(torch.zeros(1, 1, self.hidden_size)))
        result = None
        if use_cuda:
            return result[0].cuda(), result[1].cuda()
        else:
            return result


    def save(self, fname):
        """Save the model to a pickle file."""
        with open(fname, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)


class RNNDecoder(nn.Module):
    """simple initial decoder"""
    def __init__(self, vocab_size, embed_size, hidden_size, rnn_type='LSTM',  num_layers=1, bidirectional=False):
        super(RNNDecoder, self).__init__()
        self.vocab_size = vocab_size  #target vocab size
        self.hidden_size = hidden_size
        self.embedding   = nn.Embedding(vocab_size, embed_size)
        self.num_layers  = num_layers
        self.rnn = rnn_factory(rnn_type, input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)  #requires embed_size = hidden_size
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)  #dim corresponding to vocab
        self.hidden = self.init_hidden()

    # Generates entire sequence, up to tgt_len, conditioned on the initial hidden state
    def forward(self, init_hidden, tgt_len, generate=False):
        self.hidden = init_hidden
        decoder_input = Variable(torch.LongTensor([[SOS]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        outputs = []
        for _ in range(tgt_len):
            decoder_output = self.__forward_one_word(decoder_input)
            _, top_idx = decoder_output.data.topk(1)
            # Recover the value of the top index and wrap in a new variable to break backprop chain
            word_idx = top_idx[0][0]
            decoder_input = Variable(torch.LongTensor([[word_idx]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            if generate:
                outputs.append(word_idx)
            else:
                outputs.append(decoder_output)
            if word_idx == EOS:
                break
        return outputs

    # Passes a single word through the decoder network
    def __forward_one_word(self, tgt):
        # Map to embeddings - dims are (seq length, batch size, emb size)
        output = self.embedding(tgt).view(1, 1, -1)
        # Non-linear activation over embeddings
        output = F.relu(output)
        # Pass representation through RNN
        output, self.hidden = self.rnn(output, self.hidden)
        # Softmax over the final output state
        output = self.softmax(self.out(output[0]))

        return output

    def init_hidden(self):
        #Initialize hidden state to a pair of variables for context/hidden
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
        return self._forward(src, tgt.shape[0])

    def generate(self, src, max_length):
        return self._forward(src, max_length, generate=True)

    # src,tgt currently single sentences
    def _forward(self, src, tgt_len, generate=False):
        self.encoder.hidden = self.encoder.init_hidden()
        self.encoder(src)
        decoder_outputs = self.decoder(self.encoder.hidden, tgt_len, generate=generate)
        return decoder_outputs

    def save(self, fname):
        """Save the model to a pickle file."""
        print("Saving at: {}".format(fname))
        with open(fname, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)
