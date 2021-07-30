import torch
import random
import copy
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Sequential, CrossEntropyLoss

from utility_general import flatten_grad, create_progressbar, save_to_file

from influence_function import grad_z

def flattened_parameters_to_copied_model(model, flattened_parameters):
    i = 0
    new_model = model
    for name, param in new_model.named_parameters():
        #print('name: ', name)
        #print('param.shape: ', param.shape)
        current_shape = np.array(list(param.shape))
        no_params_in_layer = np.prod(current_shape)
        #print(i, i+no_params_in_layer)
        new_params = flattened_parameters[i:(i+no_params_in_layer)].reshape(list(param.shape))
        param.data = torch.Tensor(new_params)
        #print(current_shape, no_params_in_layer, np.shape(new_params))
        i += no_params_in_layer
    return new_model

def modify_model_parameters(model, modification):
    i = 0
    new_model = copy.deepcopy(model)
    for name, param in new_model.named_parameters():
        #print('name: ', name)
        #print('param.shape: ', param.shape)
        #print('Original values: ', param.data)
        current_shape = np.array(list(param.shape))
        no_params_in_layer = np.prod(current_shape)
        #print(i, i+no_params_in_layer)
        #print("Modification: ")
        #print(modification[i:(i+no_params_in_layer)].reshape(current_shape))
        param.data = param.data - torch.Tensor(modification[i:(i+no_params_in_layer)].reshape(current_shape))
        #print('New values: ', param.data)
        i += no_params_in_layer

    return new_model

def resamplingUncertaintyEstimation(chosen_test_examples, test_loader, model, hessian, grad_training_loss, criterion, training_set_size, λ, damping, input_size, number_of_repetitions = 10, MINIMAL_VERSION = True, save = True, folder_rue='D:\\+ ML and me +\\rue'):

    original_model = model

    # Get inverse of the Hessian
    #hessian = np.load(folder_influence + '/' + model_name + '_hessian.npy') #folder_influence + '/' +
    model_params_no = np.shape(hessian)[0]
    damped_hessian = hessian + np.identity(model_params_no)*damping
    #print(np.linalg.eigvalsh(damped_hessian))
    inv_hessian = np.linalg.inv(damped_hessian)
    print("The full Hessian got inverted.")

    # Get A = (damped_H)^(-1) * grad_train_loss
    grad_training_loss = np.array(grad_training_loss).T # to have d x n instead of n x d
    A = np.dot(inv_hessian, grad_training_loss) # dxd dxn -> dxn
    #print("A", A)

    # Initialize vectors and matrices
    w0 = np.ones(training_set_size)
    #print("w0", w0)
    p0 = w0 / training_set_size
    #print("p0", p0)
    true_matrixY = np.zeros((number_of_repetitions, len(chosen_test_examples)))
    minimal_matrixY = np.zeros((number_of_repetitions, len(chosen_test_examples)))

    for i in create_progressbar(number_of_repetitions, desc='Bootstrap sampling'):
        w = np.random.multinomial(training_set_size, p0)
        vec_diff = w - w0
        modification = np.dot(A, vec_diff)
        #print("modification: ", np.shape(modification))

        modified_model = modify_model_parameters(model, modification)

        for eigenvectors, labels in test_loader:
            eigenvectors = eigenvectors.reshape(-1, 1, input_size)

            outputs = modified_model(eigenvectors)
            _, predicted = torch.max(outputs.data, 1)  # classification
            predicted = Variable(predicted)

            true_test_loss = criterion(outputs, labels)
            minimal_test_loss = criterion(outputs, predicted)
            
            # We manually add L2 regularization
            if λ != 0:
                l2_reg = 0.0
                for param in modified_model.parameters():
                    l2_reg += torch.norm(param)**2
                    true_test_loss += 1/training_set_size * λ/2 * l2_reg
                    minimal_test_loss += 1/training_set_size * λ/2 * l2_reg

            if i == 0:
                # Cross-check
                outputs = original_model(eigenvectors)
                _, predicted = torch.max(outputs.data, 1)  # classification
                predicted = Variable(predicted)

                true_original_test_loss = criterion(outputs, labels)
                minimal_original_test_loss = criterion(outputs, predicted)
                
                # We manually add L2 regularization
                if λ != 0:
                    l2_reg = 0.0
                    for param in original_model.parameters():
                        l2_reg += torch.norm(param)**2
                        true_original_test_loss += 1/training_set_size * λ/2 * l2_reg
                        minimal_original_test_loss += 1/training_set_size * λ/2 * l2_reg

                #print("Original true test losses: ", true_original_test_loss)
                #print("Original minimal test losses: ", minimal_original_test_loss)
                #print("New true test losses: ", true_test_loss)
                #print("New minimal test losses: ", minimal_test_loss)
            else:
                continue
                #print("New true test losses: ", true_test_loss)
                #print("New minimal test losses: ", minimal_test_loss)

            true_matrixY[i] = np.array(true_test_loss.detach().numpy())
            minimal_matrixY[i] = np.array(minimal_test_loss.detach().numpy())
    
    true_variance = np.var(true_matrixY, axis=0)
    minimal_variance = np.var(minimal_matrixY, axis=0)
    print("True variances: ", true_variance)
    print("Minimal variances: ", minimal_variance)
    
    return true_variance, minimal_variance

def calculateRUE(input_size, train_loader, test_loader, model, λ, hessian, damping, number_of_repetitions = 10, training_set_size = 1000, test_set_size = 50, criterion = nn.CrossEntropyLoss(reduction='none'), chosen_test_examples = np.arange(50)):
    #folder_influence, 
    # Get gradient of training loss (each vector element corresponds to different training point)
    grad_train_losses = []
    for i, (eigenvectors, labels) in enumerate(train_loader):
        eigenvectors = eigenvectors.reshape(-1, 1, input_size)
        for train_example in create_progressbar(training_set_size, desc='Calculating the gradient of the training loss'):

            grad_train_loss = grad_z(eigenvectors[train_example:train_example+1], labels[train_example:train_example+1], model, criterion, training_set_size, λ=λ)
            grad_train_loss = 1/training_set_size * flatten_grad(grad_train_loss)
            grad_train_losses.append(grad_train_loss.cpu().data.numpy())

    # Calculate RUE for chosen test points
    true_rue, minimal_rue = resamplingUncertaintyEstimation(chosen_test_examples, test_loader, model, hessian, grad_train_losses, criterion, training_set_size, λ, damping, input_size, number_of_repetitions=number_of_repetitions)
    
    return true_rue, minimal_rue