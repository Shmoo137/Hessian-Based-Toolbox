import torch
import random
import copy
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Sequential, CrossEntropyLoss

from utility_general import flatten_grad, create_progressbar, save_to_file

from influence_function import grad_z

def findLocalEnsemble(hessian, treshold):

    model_params_no = np.shape(hessian)[0]
    eigenvalues, local_ensemble = np.linalg.eigh(hessian)
    #print("Hessian has", len(eigenvalues), "eigenvalues and eigenvectors have this shape:", local_ensemble.shape)
    print("Chosen treshold for the eigenvalues is", treshold)
    print("20 Hessian's largest eigenvalues are: ", eigenvalues[-20:])

    for i in np.flip(np.arange(model_params_no)):
        if i==model_params_no-1 or i==model_params_no-2:
            local_ensemble = np.delete(local_ensemble, i, axis=1) # eigenvectors are columns
            #print(i, eigenvalues[i], local_ensemble.shape)
        elif eigenvalues[i] > treshold:
            local_ensemble = np.delete(local_ensemble, i, axis=1)
            #print(i, eigenvalues[i], local_ensemble.shape)
        else:
            continue
    
    print("Final local ensemble has shape of:", local_ensemble.shape)

    m = model_params_no - local_ensemble.shape[1]

    print("Number of 'removed' eigenvectors: ", m)

    return local_ensemble, m

def calculateLEES(input_size, test_loader, model, λ, hessian, treshold, MINIMAL_VERSION = True, training_set_size = 1000, test_set_size = 50, chosen_test_examples = np.arange(50), criterion = nn.CrossEntropyLoss(reduction='none')):
    
    local_ensemble, m = findLocalEnsemble(hessian, treshold)

    # Get a gradient of a single test loss
    lees = []
    for _, (test_eigenvectors, test_labels) in enumerate(test_loader):
        
        test_eigenvectors = test_eigenvectors.reshape(-1, 1, input_size)
            
        for i in create_progressbar(len(chosen_test_examples), desc='calculating local ensemble extrapolation score for chosen test examples'):

            test_example = chosen_test_examples[i]
            #print("Looking at the test example no. ", test_example, " right now.")

            if MINIMAL_VERSION is True:
                grad_test_loss = grad_z(test_eigenvectors[test_example:test_example+1], test_labels[test_example:test_example+1], model, criterion, training_set_size, λ=λ, without_labels=True)
            else:    
                grad_test_loss = grad_z(test_eigenvectors[test_example:test_example+1], test_labels[test_example:test_example+1], model, criterion, training_set_size, λ=λ)
            
            grad_test_loss = flatten_grad(grad_test_loss)

            multiplication = np.dot(local_ensemble.T, grad_test_loss.detach().numpy())
            result = np.linalg.norm(multiplication)

            lees.append(result)
    
    return lees, m