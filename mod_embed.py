import torch
from torch.nn.parameter import Parameter
import logging
#adds a new embedding based on the embeddings at vocab indices idx and idy


logger = logging.getLogger(__name__)

def update_embed(embed, idx, idy, operation="avg"):
    #add one to num embeddings
    embed.num_embeddings += 1
    #new embedding created from the two at indices idx and idy
    if idx > embed.weight.shape[0]:
        print("Left embedding is not in weights")
        new_embedding = embed.weight[idy].view(1, -1)
    elif idy > embed.weight.shape[0]:
        print("Right embedding is not in weights")
        new_embedding = embed.weight[idx].view(1, -1)
    elif operation == "avg":
        new_embedding = torch.div(torch.add(embed.weight[idx], embed.weight[idy]), 2).view(1,-1)
    elif operation == "max":
        new_embedding = torch.max(embed.weight[idx], embed.weight[idy]).view(1,-1)
    embed.weight = Parameter(torch.cat((embed.weight[:].data, new_embedding.data), 0))
    return

def update_linear(linear, idx, idy, operation="avg"):
    #add one to num embeddings
    linear.out_features += 1
    #new embedding created from the two at indices idx and idy
    if idx > linear.weight.shape[0] or idy > linear.weight.shape[0]:
        logger.error("Left embedding index is out of bounds.")
        raise Exception
    elif idy > linear.weight.shape[0]:
        logger.error("Right embedding index is out of bounds.")
        raise Exception
    elif operation == "avg":
        new_lin_weight = torch.div(torch.add(linear.weight[idx], linear.weight[idy]), 2).view(1,-1)
        new_lin_bias = torch.div(torch.add(linear.bias[idx], linear.bias[idy]), 2).view(1)
    elif operation == "max":
        new_lin_weight = torch.max(linear.weight[idx], linear.weight[idy]).view(1,-1)
        new_lin_bias = torch.max(linear.bias[idx], linear.bias[idy]).view(1)
    #add to weights (if the new weights are initialized correctly, softmax will just follow)
    linear.weight = Parameter(torch.cat((linear.weight[:].data, new_lin_weight.data), 0))
    linear.bias = Parameter(torch.cat((linear.bias[:].data, new_lin_bias.data), 0))
    return

    #optimizer will not be aware of new parameters
    #  - could reinitialize optimizer, but will lose momentum
    #  - could clone optimizer and somehow add momentum for new params?
    #may want to do this operation for all new embeddings at once?

