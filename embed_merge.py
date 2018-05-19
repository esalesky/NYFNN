import logging
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import numpy as np
from sklearn.model_selection import train_test_split
from training import optimizer_factory
from utils import use_cuda
import pdb

logger = logging.getLogger(__name__)

"""Abstract class for implementing training callbacks."""


def get_merger(merge_type, **kwargs):
    if merge_type == 'avg':
        return AvgEmbeddingMerger(**kwargs)
    elif merge_type == 'max':
        return MaxEmbeddingMerger(**kwargs)
    elif merge_type == 'ae':
        return AutoencoderEmbeddingMerger(**kwargs)
    elif merge_type == 'rand':
        return RandomEmbeddingMerger(**kwargs)
    else:
        logger.error("Unknown merger type: {}".format(merge_type))
        raise Exception


class Autoencoder(nn.Module):

    def __init__(self, embed_size, hidden_size=None):
        super(Autoencoder, self).__init__()
        self.embed_size = embed_size
        if hidden_size is None:
            self.hidden_size = int(self.embed_size * 1.5)
        else:
            self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            nn.Linear(self.embed_size*2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.embed_size)
        )

        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.embed_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.embed_size*2)
        )

    def forward(self, embed_pair):
        return self.decoder(self.encoder(embed_pair))

    def combine_embeddings(self, embed_pair):
        return self.encoder(embed_pair)


class EmbeddingMerger(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def generate_embeddings(self, embedding_indices, model):
        pass

    def check_indexes(self, idx, idy, size):
        if idx > size:
            logger.error("Left embedding is not in weights")
            raise Exception
        elif idy > size:
            logger.error("Right embedding is not in weights")
            raise Exception

    def update_linear(self, linear, idx, idy, operation="avg"):
        # add one to num embeddings
        linear.out_features += 1
        # new embedding created from the two at indices idx and idy
        if idx > linear.weight.shape[0] or idy > linear.weight.shape[0]:
            logger.error("Left embedding index is out of bounds.")
            raise Exception
        elif idy > linear.weight.shape[0]:
            logger.error("Right embedding index is out of bounds.")
            raise Exception
        elif operation == "avg":
            new_lin_weight = torch.div(torch.add(linear.weight[idx], linear.weight[idy]), 2).view(1, -1)
            new_lin_bias = torch.div(torch.add(linear.bias[idx], linear.bias[idy]), 2).view(1)
        elif operation == "max":
            new_lin_weight = torch.max(linear.weight[idx], linear.weight[idy]).view(1, -1)
            new_lin_bias = torch.max(linear.bias[idx], linear.bias[idy]).view(1)
        # add to weights (if the new weights are initialized correctly, softmax will just follow)
        linear.weight = Parameter(torch.cat((linear.weight[:].data, new_lin_weight.data), 0))
        linear.bias = Parameter(torch.cat((linear.bias[:].data, new_lin_bias.data), 0))
        return


class AvgEmbeddingMerger(EmbeddingMerger):

    def __init__(self, **kwargs):
        super(AvgEmbeddingMerger, self).__init__()

    def generate_embeddings(self, embedding_indices, layers):
        # add one to num embeddings
        embedding = layers['embed']
        linear = layers['out']
        for (idx, idy) in embedding_indices:
            self.check_indexes(idx, idy, embedding.weight.shape[0])
            embedding.num_embeddings += 1
            # new embedding created from the two at indices idx and idy
            new_embedding = torch.div(torch.add(embedding.weight[idx], embedding.weight[idy]), 2).view(1, -1)
            embedding.weight = Parameter(torch.cat((embedding.weight[:].data, new_embedding.data), 0))
            self.update_linear(linear, idx, idy, operation="avg")


class MaxEmbeddingMerger(EmbeddingMerger):

    def __init__(self, **kwargs):
        super(MaxEmbeddingMerger, self).__init__()

    def generate_embeddings(self, embedding_indices, layers):
        # add one to num embeddings
        embedding = layers['embed']
        linear = layers['out']
        for (idx, idy) in embedding_indices:
            self.check_indexes(idx, idy, embedding.weight.shape[0])
            embedding.num_embeddings += 1
            new_embedding = torch.max(embedding.weight[idx], embedding.weight[idy]).view(1, -1)
            embedding.weight = Parameter(torch.cat((embedding.weight[:].data, new_embedding.data), 0))
            self.update_linear(linear, idx, idy, operation="avg")


class RandomEmbeddingMerger(EmbeddingMerger):

    def __init__(self, **kwargs):
        super(RandomEmbeddingMerger, self).__init__()

    def generate_embeddings(self, embedding_indices, layers):
        # add one to num embeddings
        embedding = layers['embed']
        linear = layers['out']
        for (idx, idy) in embedding_indices:
            self.check_indexes(idx, idy, embedding.weight.shape[0])
            embedding.num_embeddings += 1
            # new embedding is random of same size as embed[idx]
            new_embedding = Variable(torch.rand(1,len(embedding.weight[idx]))).view(1,-1)
            if use_cuda:
                new_embedding = new_embedding.cuda()            
            embedding.weight = Parameter(torch.cat((embedding.weight[:].data, new_embedding.data), 0))
            self.update_linear(linear, idx, idy, operation="avg")


class AutoencoderEmbeddingMerger(EmbeddingMerger):

    def __init__(self, **kwargs):
        super(AutoencoderEmbeddingMerger, self).__init__()
        self.autoenc = Autoencoder(kwargs['embed_size'])
        if use_cuda:
            self.autoenc = self.autoenc.cuda()
        self.trained = False
        self.patience = 3
        self.curr_patience = 0

    def generate_embeddings(self, embedding_indices, layers):
        embedding_layer = layers['embed']
        linear = layers['out']
        logger.info("There are {} embeddings to train on.".format(len(embedding_indices)))
        # Map indices to concatenation of embeddings
        embeddings = [*map(lambda x: self._idx_to_embed(x, embedding_layer),
                                                           embedding_indices)]
        filtered_embeddings = [*filter(lambda y: y is not None, embeddings)]
        logger.info("There are {} filtered embeddings to train on.".format(len(filtered_embeddings)))
        # Train Autoencoder on new embedding pairs
        self.train_embeddings(filtered_embeddings, embedding_layer)
        for (idx, idy) in embedding_indices:
            self.check_indexes(idx, idy, embedding_layer.weight.shape[0])
            concat_embedding = self._idx_to_embed((idx, idy), embedding_layer)
            if concat_embedding is None:
                logger.info("Alex was right.")
                raise Exception
            concat_embedding = Variable(torch.FloatTensor(concat_embedding)).view(1, -1)
            if use_cuda:
                concat_embedding = concat_embedding.cuda()
            new_embedding = self.autoenc.combine_embeddings(concat_embedding)
            embedding_layer.weight = Parameter(torch.cat((embedding_layer.weight[:].data, new_embedding.data), 0))
            embedding_layer.num_embeddings += 1
            self.update_linear(linear, idx, idy, operation="avg")


    def train_embeddings(self, embeddings, embedding_layer):
        train, dev = train_test_split(embeddings, shuffle=True, random_state=69, test_size=.1)
        batch_size = 64
        train_batches = self.make_batches(train, batch_size)
        dev_batches = self.make_batches(dev, batch_size)
        optimizer = optimizer_factory('Adam', self.autoenc, lr=0.0001)
        loss_fn = nn.MSELoss()
        best_dev_loss = float('inf')
        num_epochs = 50 if not self.trained else 50
        for epoch in range(num_epochs):
            total_train_loss = 0
            for i in range(len(train_batches)):
                optimizer.zero_grad()
                batch = Variable(torch.FloatTensor(train_batches[i]))
                if use_cuda:
                    batch = batch.cuda()
                output = self.autoenc(batch)
                loss = loss_fn(output, batch) / batch_size
                total_train_loss += loss.data[0]
                loss.backward()
                optimizer.step()
            total_train_loss /= len(train_batches)
            if epoch % 5 == 0:
                logger.info("Avg Autoencoder Loss at Epoch {}: {:.6f}".format(epoch, total_train_loss))
            total_dev_loss = 0
            for i in range(len(dev_batches)):
                batch = Variable(torch.FloatTensor(dev_batches[i]))
                if use_cuda:
                    batch = batch.cuda()
                output = self.autoenc(batch)
                loss = loss_fn(output, batch) / batch_size
                total_dev_loss += loss.data[0]
            total_dev_loss /= len(dev_batches)

            if total_dev_loss < best_dev_loss:
                best_dev_loss = total_dev_loss
                self.curr_patience = 0
            else:
                self.curr_patience += 1
                if self.curr_patience == self.patience:
                    break
            if epoch % 5 == 0:
                logger.info("Avg Autoencoder Dev Loss at Epoch {}: {:.6f}".format(epoch, total_dev_loss))
        self.trained = True

    def _idx_to_embed(self, indexes, embedding):
        if indexes[0] >= embedding.weight.shape[0] or indexes[1] >= embedding.weight.shape[0]:
            return None
        return torch.cat((embedding.weight[indexes[0]], embedding.weight[indexes[1]]), 0).view(-1).data.cpu().numpy()


    def make_batches(self, train_instances, batch_size=64):
        batches = []
        curr_batch = []
        for i in range(len(train_instances)):
            curr_batch.append(train_instances[i])
            if len(curr_batch) == batch_size or i == (len(train_instances) - 1):
                # batch = torch.cat(curr_batch)
                batch = np.array([b for b in curr_batch])
                batches.append(batch)
                curr_batch.clear()

        return batches
