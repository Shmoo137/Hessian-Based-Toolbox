import torch
import datetime
import numpy as np
from hvp import hvp
from torch.autograd import Variable

from utility_general import flatten_grad, create_progressbar, save_to_file
import torch.nn as nn
from torch.autograd import grad
from data_loader import Downloader
import random # to shuffle the data

def grad_z(z, t, model, loss_criterion, n, λ=0, without_labels = False):
    model.eval()
    # initialize
    z, t = Variable(z), Variable(t) # here were two flags: volatile=False. True would mean that autograd shouldn't follow this. Got disabled
    y = model(z)
    
    if without_labels is True:
        # model makes predictions
        _, predicted = torch.max(y.data, 1)
        predicted = Variable(predicted)
        
        # when ground-truth labels are unavailable, we can calculate "minimal" test loss, using predictions as labels
        loss = loss_criterion(y, predicted)
    else:
        loss = loss_criterion(y, t)

    # We manually add L2 regularization
    l2_reg = 0.0
    for param in model.parameters():
        l2_reg += torch.norm(param)**2
    loss += 1/n * λ/2 * l2_reg

    return list(grad(loss, list(model.parameters()), create_graph=True))

def exact_influence_functions(input_size, train_loader, test_loader, model, λ, hessian, damping, folder_influence, COMPUTE_RELATIF = True, num_classes = 2, training_set_size = 1000, test_set_size = 50, chosen_test_examples = np.arange(50), criterion = nn.CrossEntropyLoss(), random_seed_fixed = True):

    if random_seed_fixed is True:
        np.random.seed(0)
        torch.manual_seed(17)

    model_params_no = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model has ", model_params_no, ' parameters.')

    for i, (eigenvectors, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, # of RGB channels, size, size)
        eigenvectors = eigenvectors.reshape(-1, 1, input_size)
        
        # Forward pass
        hessian_outputs = model(eigenvectors)
        hessian_loss = criterion(hessian_outputs, labels)

        # We manually add L2 regularization
        if λ != 0:
            l2_reg = 0.0
            for param in model.parameters():
                l2_reg += torch.norm(param)**2
            hessian_loss += 1/training_set_size * λ/2 * l2_reg

        damped_hessian = hessian + np.identity(model_params_no)*damping
        print("Smallest eigenvalues of the damped hessian: ", np.linalg.eigvalsh(damped_hessian)[:5])
        inv_hessian = torch.inverse(torch.from_numpy(damped_hessian)).float()
        print("The full Hessian got inverted.")

        for _, (test_eigenvectors, test_labels) in enumerate(test_loader):
            
            test_eigenvectors = test_eigenvectors.reshape(-1, 1, input_size)
                
            for i in create_progressbar(len(chosen_test_examples), desc='calculating influence for chosen test examples'):

                test_example = chosen_test_examples[i]
                #print("Looking at the test example no. ", test_example, " right now.")

                grad_test_loss = grad_z(test_eigenvectors[test_example:test_example+1], test_labels[test_example:test_example+1], model, criterion, training_set_size, λ=λ)
                grad_test_loss = flatten_grad(grad_test_loss)
                s_test = torch.mv(inv_hessian, grad_test_loss)

                #print("s_test for the test example no. ", test_example, " will be multiplied by gradients right now.")

                influence = []
                relatIF = []
                for train_example in create_progressbar(training_set_size, desc='calculating influence (and optionally RelatIF)'):

                    grad_train_loss = grad_z(eigenvectors[train_example:train_example+1], labels[train_example:train_example+1], model, criterion, training_set_size, λ=λ)
                    grad_train_loss = flatten_grad(grad_train_loss)

                    influence_function = - torch.dot(s_test, grad_train_loss) * (- 1 / training_set_size)
                    influence.append(influence_function.item())

                    if COMPUTE_RELATIF is True:
                        divider = torch.norm(torch.mv(inv_hessian, grad_train_loss))
                        relat_influence_function = influence_function / divider
                        relatIF.append(relat_influence_function.item())

                save_to_file(influence, 'exact_influence_test' + str(test_example) + '.txt', folder_influence)

                if COMPUTE_RELATIF is True:
                    save_to_file(relatIF, 'exact_relatIF_test' + str(test_example) + '.txt', folder_influence)

    print("IF (and RelatIF) computations completed.")