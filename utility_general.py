import os
import torch
import six
import inspect
import collections
import pandas as pd
from tqdm import tqdm, trange
import re

#import pandas as pd
import numpy as np

np.set_printoptions(threshold=np.inf)

def remove_character_from_string(string, character):
    filtered = re.sub('[' + character + ']', '', string)
    return filtered

def replace_character_in_string(string, old_character, new_character):
    return string.replace(old_character, new_character)

def flatten_grad(grad):
    tuple_to_list = []
    for tensor in grad:
        tuple_to_list.append(tensor.reshape(-1))
    all_flattened = torch.cat(tuple_to_list)
    return all_flattened

def find_hessian(loss, model):
    grad1 = torch.autograd.grad(loss, model.parameters(), create_graph=True) #create graph important for the gradients

    grad1 = flatten_grad(grad1)
    list_length = grad1.size(0)
    hessian = torch.zeros(list_length, list_length)

    for idx in trange(list_length, desc="Calculating hessian"):
            grad2rd = torch.autograd.grad(grad1[idx], model.parameters(), create_graph=True)
            cnt = 0
            for g in grad2rd:
                g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
                cnt = 1
            hessian[idx] = g2.detach().cpu()
            del g2

    H = hessian.cpu().data.numpy()
    # calculate every element separately -> detach after calculating all 2ndgrad from this 1grad
    return H

def find_heigenvalues(loss, model):
    H = find_hessian(loss, model)
    eigenvalues = np.linalg.eigvalsh(H)
    return eigenvalues

def biggest_heigenvalue(hessian_or_loss, model = None, negative=None):
    if model is None:
        eigenvalues = np.linalg.eigvalsh(hessian_or_loss)
    else:
        eigenvalues = find_heigenvalues(hessian_or_loss, model)

    eigenvalues = np.sort(eigenvalues)
    big = eigenvalues[-1]
    small = eigenvalues[0]

    if negative is None:
        if abs(small) > abs(big):
            return small
        else:
            return big
    else:
        if small < 0.0:
            return small
        else:
            print("Hessian has no negative eigenvalue!")
            return None

def save_to_file(list, filename, folder_name = None):
    if folder_name is None:
        folder_name = 'tmp'
    
    if isinstance(list, torch.Tensor):
        if list.requires_grad is True:
            list = list.detach().numpy()

    list_to_string = np.array2string(np.asarray(list), separator=' ', max_line_width=np.inf)
    list_wo_brackets = list_to_string.translate({ord(i): None for i in '[]'})
    file = open(folder_name + '/' + filename, 'w')
    file.write(list_wo_brackets)
    file.close()

def append_to_file(list, filename, folder_name = None, delimiter = ' '):
    if folder_name is None:
        folder_name = 'tmp'
    
    if isinstance(list, torch.Tensor):
        if list.requires_grad is True:
            list = list.detach().numpy()

    list_to_string = np.array2string(np.asarray(list), separator=delimiter, max_line_width=np.inf)
    list_wo_brackets = list_to_string.translate({ord(i): None for i in '[]'})

    file = open(folder_name + '/' + filename, 'a')
    file.write("\n")
    file.close()

    file = open(folder_name + '/' + filename, 'a')
    file.write(list_wo_brackets)
    file.close()

def make_dir_one(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_dir(path):
    separated_path = path.split('/')
    tmp_path = ''
    for directory in separated_path:
        tmp_path = tmp_path + directory + '/'
        if directory == '.':
            continue
        make_dir_one(tmp_path)
    return True

def find_files(path, affix_flag=False):
    if path[-1] == '/':
        path = path[:-1]
    if affix_flag is False:
        return [path + '/' + name for name in os.listdir(path)]
    else:
        return [name for name in os.listdir(path)]

def remove_slash(path):
    return path[:-1] if path[-1] == '/' else path

def create_progressbar(end, desc='', stride=1, start=0):
    return tqdm(six.moves.range(int(start), int(end), int(stride)), desc=desc, leave=False)

# store builtin print
old_print = print

def new_print(*args, **kwargs):
    # if tqdm.tqdm.write raises error, use builtin print
    try:
        tqdm.tqdm.write(*args, **kwargs)
    except:
        old_print(*args, ** kwargs)

# globaly replace print with new_print
inspect.builtins.print = new_print

# nD list to 1D list
def flatten(list):
    if isinstance(list, collections.Iterable):
        return [a for i in list for a in flatten(i)]
    else:
        return [list]