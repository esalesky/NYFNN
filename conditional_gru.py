import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.autograd import Variable
from utils import use_cuda

import logging
logger = logging.getLogger(__name__)

class Attn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attn, self).__init__()
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


class TanhAttn(nn.Module):

    """Tanh Attention from Nematus Paper computes attention score as w * tanh(U * t + V * S)
       S = source encoding matrix, t = target word, U, V, w are weight tensors"""
    def __init__(self, enc_size, hidden_size):
        super(TanhAttn, self).__init__()
        self.enc_size = enc_size
        self.hidden_size = hidden_size
        self.src_linear = nn.Linear(self.enc_size, self.hidden_size)
        self.tgt_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.tanh = nn.Tanh()
        self.tanh_linear = nn.Linear(self.hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, attn_scores):
        batch_size = attn_scores.shape[0]

        # print("Src encodings shape", attn_scores.shape)
        output = self.tgt_linear(hidden)
        output = output.unsqueeze(1)
        # print("Output after tgt linear", output.shape)
        # print(output)
        output = output.expand((batch_size, attn_scores.shape[1], self.hidden_size))
        # print("Output after tgt linear", output.shape)
        # print(output)
        # U * t + V * s
        output = output + attn_scores
        # print(output)
        # Tanh non-linearity
        output = self.tanh_linear(self.tanh(output))
        # print("After dot product", output.shape)
        # Dot product to generate a single score for each source word
        output = self.softmax(output).transpose(1, 2)
        return output

    def calc_attn_scores(self, encoder_outputs):
        return self.src_linear(encoder_outputs)


class ConditionalGRUAttn(nn.Module):

    """A module that uses an attention layer between 2 GRU layers."""

    def __init__(self, input_size, hidden_size, context_size, batch_first=False):
        super(ConditionalGRUAttn, self).__init__()
        # self.attn_type = attn_type  #bilinear, h(src)T * W * h(tgt)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.batch_first = batch_first
        self.first_cell = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)
        self.second_cell = nn.GRUCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
        # todo: They use a tanh layer for attention
        self.attn = Attn(self.context_size * 2, self.hidden_size)
        self.first_cell.reset_parameters()
        self.second_cell.reset_parameters()


    def calc_attn_scores(self, encoder_outputs):
        return self.attn.calc_attn_scores(encoder_outputs)

    """Input is a sequence, hidden is the initial hidden state, context is the sequence of hidden states"""
    def forward(self, input_, hidden, context, attn_scores):

        if self.batch_first:
            batch_size = input_.shape[0]
            # Reorder dimensions as (seq. length, batch size, hidden size)
            input_ = input_.transpose(0, 1)
            # context = context.transpose(0, 1)
        else:
            batch_size = input_.shape[1]

        # print(attn_scores.shape)

        num_steps = input_.shape[0]

        outputs = []
        attention = []
        hidden = hidden.squeeze(1)
        # print("Attention scores shape", attn_scores.shape)
        for i in range(num_steps):
            #First GRU input is input[i], hidden from previous full cell run --> (batch_size, 1, hidden_size)
            # print("Input shape", input_[i].shape)
            # print("Hidden shape", hidden.shape)
            interim_hidden = self.first_cell(input_[i], hidden)
            # print("Interim hidden shape", interim_hidden.shape)
            # calculate attn against each encoder outputs - dims are (batch_size, 1, src_len)
            # The 1 is necessary for the batch matrix multiple in the next step
            attn_weights = self.attn(interim_hidden, attn_scores)
            attention.append(attn_weights)
            # print("Attention weights shape: ", attn_weights.shape)
            # print("Attention weights", attn_weights)
            # (batch_size, 1, src_len) * (batch_size, src_len, hidden_size) --> (batch_size, 1, hidden_size)
            context = attn_weights.bmm(context)
            # print("Context shape: ", context.shape)
            output = self.second_cell(interim_hidden, context.squeeze(1))
            # print("Output shape: ", output.shape)
            outputs.append(output)
            hidden = output

        output = torch.stack(outputs, 0)
        attention = torch.stack(attention, 0).squeeze(0)
        # print("Output", output.shape)
        if self.batch_first:
            output = output.transpose(0, 1)
        #Return the compacted outputs and the final hidden state
        return output, attention, hidden