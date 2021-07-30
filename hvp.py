import torch
from torch.autograd import grad
from utility_general import flatten_grad

def hvp(y, w, v):   # it's really just a second derivative of y(w) multiplied by 1D tensor v
    first_grads = grad(y, w, create_graph=True)
    # now we need make first_grads a 1D tensor, otherwise multiplying and summing get pairs wrong
    first_grads = flatten_grad(first_grads)
    #v = torch.cat([v[0].contiguous().view(-1)])
    # be careful so your vector v doesn't depend on w, otherwise we'll get a double derivative
    grad_v = 0
    for g, v in zip(first_grads, v): # zip: merges [1,2,3] i [4,5,6] w [(1,4),(2,5),(3,6)]
        grad_v += torch.sum(g * v)
    return grad(grad_v, w) # I don't need to create graph here, 2nd derivative is the deepest I want