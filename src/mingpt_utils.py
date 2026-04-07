import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def top_p_logits(logits, top_p, verbose=False):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    out = logits.clone()
    indices_to_remove = torch.zeros_like(out,dtype=torch.bool).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove )
    out[indices_to_remove] = -float('Inf')#filter_value
    if verbose: 
        print('|top_p:',(out > -float('Inf')).sum().item(),'\n')
    return out

@torch.no_grad()
def sample(model, x, m, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
       
        y = None
        logits, _ = model(x_cond, y , m)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)
    return x

def sample_new(model, x, m, steps, temperature=1.0, sample=True, top_k=None, top_p=None):

    block_size = model.get_block_size()
    model.eval()
    #print(x.shape)
    l = torch.zeros((1,model.tok_emb.weight.size()[0])).to('cuda')
    p = torch.zeros((1,model.tok_emb.weight.size()[0])).to('cuda')

    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        y = None
        logits, _ = model(x_cond, y , m)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None and top_k > 0.0:
            logits = top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
        
        # optionally crop probabilities to only the top p probability mass
        if top_p is not None and top_p > 0.0:
            logits = top_p_logits(logits, top_p)
            probs = F.softmax(logits, dim=-1)
        
        else: probs = F.softmax(logits, dim=-1)
        
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        
        # append to the sequence and continue
        #print(l.shape)
        l = torch.cat((l, logits), dim=0)
        p = torch.cat((p, probs), dim=0)
        x = torch.cat((x, ix), dim=1)
   
    x = x[0]
    return x, l, p

def sample_p(model, x, steps, temperature=1.0, sample=True, top_k=None, top_p=None):

    block_size = model.get_block_size()
    model.eval()
    #print(x.shape)
    l = torch.zeros((1,model.tok_emb.weight.size()[0])).to('cuda')
    p = torch.zeros((1,model.tok_emb.weight.size()[0])).to('cuda')

    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None and top_k > 0.0:
            logits = top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
        
        # optionally crop probabilities to only the top p probability mass
        if top_p is not None and top_p > 0.0:
            logits = top_p_logits(logits, top_p)
            probs = F.softmax(logits, dim=-1)
        
        else: probs = F.softmax(logits, dim=-1)
        
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        
        # append to the sequence and continue
        #print(l.shape)
        l = torch.cat((l, logits), dim=0)
        p = torch.cat((p, probs), dim=0)
        x = torch.cat((x, ix), dim=1)
   
    x = x[0]
    return x, l, p