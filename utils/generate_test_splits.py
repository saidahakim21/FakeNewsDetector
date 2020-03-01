import random
import os
from collections import defaultdict


def generate_splited_data_ids (dataset, ratio = 0.8):


    article_ids = list(range(len(dataset.trainData))) # get a list of article ids
    return split_ids(article_ids, ratio)


def split_ids(article_ids, training):
    r = random.Random()
    r.seed(1489215)
    r.shuffle(article_ids)  # and shuffle that list
    training_ids = article_ids[:int(training * len(article_ids))]
    hold_out_ids = article_ids[int(training * len(article_ids)):]
    return training_ids, hold_out_ids


def read_ids(file,base):
    ids = []
    with open(base+"/"+file,"r") as f:
        for line in f:
           ids.append(int(line))
        return ids




def kfold_split(folds_ids, n_folds = 10):

    folds = []
    for k in range(n_folds):
        folds.append(folds_ids[int(k*len(folds_ids)/n_folds):int((k+1)*len(folds_ids)/n_folds)])

    return folds

