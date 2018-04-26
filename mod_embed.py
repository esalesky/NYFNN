import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn
import params

#adds a new embedding based on the embeddings at vocab indices idx and idy
def update_embed(embed, idx, idy, operation="avg"):
    #add one to num embeddings
    embed.num_embeddings += 1
    #new embedding created from the two at indices idx and idy
    print(idx, idy)
    if operation == "avg":
        new_embedding = torch.div(torch.add(embed.weight[idx], embed.weight[idy]), 2).view(1,-1)
    elif operation == "max":
        new_embedding = torch.max(embed.weight[idx], embed.weight[idy]).view(1,-1)
    #add to weights (if the new weights are initialized correctly, softmax will just follow)
    embed.weight = Parameter(torch.cat((embed.weight[:].data, new_embedding.data), 0))
    
    return

    #optimizer will not be aware of new parameters
    #  - could reinitialize optimizer, but will lose momentum
    #  - could clone optimizer and somehow add momentum for new params?
    #may want to do this operation for all new embeddings at once?


def merge_bpe(first, second, vocab):
    left = first+"@@"   #a@@
    if second.endswith("</w>"):
        right  = second.replace('</w>','') #b
        merged = first+right #ab
    else:
        right  = second+"@@" #b@@
        merged = first+right #ab@@

    return vocab.map2idx(left), vocab.map2idx(right), merged

def get_bpe_splits(word, vocab):
    no_eow = word.replace('</w>', '')
    for i in range(1, len(no_eow)):
        left = word[:i] + '@@'
        right = no_eow[i:]
        right = right + ('@@' if not word.endswith('</w>') else '')
        if left in vocab and right in vocab:
            print(left, right)

def get_bpe_forms(first, second):
    left = first+"@@"
    if second.endswith("</w>"):
        right  = second.replace('</w>','')
    else:
        right = second+"@@"
    return left, right