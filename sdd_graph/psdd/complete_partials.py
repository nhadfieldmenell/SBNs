#!/usr/bin/env python

import math
import time
import glob
import sys
import random

from pypsdd import *

# for printing numbers with commas
import locale
locale.setlocale(locale.LC_ALL, "en_US.UTF8")

def epochs_partial(rows,cols,num_epochs,copy):
    """Randomize the data instances and separate the data instances and partials into equal collections

    Keep relationship between full and partial data.

    Returns:
        A list of num_epochs full datasets.

        A list of num_epochs list of arrays.
            Each array holds multiple incomplete data instances.
            The different epochs have the same data instances as the corresponding epochs in full datasets.
    """

    full_file = open("../datasets/full_data_%d_%d.txt" % (rows,cols), "r")
    partials_file = open("../datasets/partials_%d_%d.txt" % (rows,cols), "r")

    full_lines = full_file.readlines()
    partials_lines = partials_file.readlines()
    """
    full_ints = map(lambda x: map(int,x[:-1].split(',')),full_lines)
    part_ints = map(lambda x: map(int,x[:-1].split(',')),partials_lines)

    full_and_part = zip(full_ints,part_ints)
    """

    full_and_part = zip(full_lines,partials_lines)

    print full_and_part[0]
    
    """
    for i in range(len(full_and_part)):
        print i
        for j in range(len(full_and_part[i][0])):
            if full_and_part[i][1][j] == 1:
                print "%d: 1 %d" % (j+1, full_and_part[i][0][j])
            elif full_and_part[i][1][j] == 0:
                print "%d: 0 %d" % (j+1, full_and_part[i][0][j])
        print ""
    """
    
    random.shuffle(full_and_part)

    epoch_num = 0
    full_epochs = [[] for i in range(num_epochs)]
    partial_epochs = [[] for i in range(num_epochs)]

    copy.uniform_weights()
    bad_models = {}
    unique_bad = 0
    total_bad = 0

    for i in range(len(full_and_part)):
        model = full_and_part[i][0]
        intermediate_name = "intermediate.txt"
        intermediate = open(intermediate_name,"w")
        intermediate.write(model)
        intermediate.close()
        model_ds = DataSet.read(intermediate_name)
        partial_model = full_and_part[i][1]
        if model in bad_models:
            total_bad += 1
            continue
        for the_model,count in model_ds:
            evidence = DataSet.evidence(the_model)
            probability = copy.probability(evidence)
            if probability == 0:
                print "bad: %s" % str(model)
                bad_models[model] = True
                unique_bad += 1
                total_bad += 1
                continue

            else:
                full_epochs[epoch_num].append(model)
                partial_epochs[epoch_num].append(partial_model)
                epoch_num = (epoch_num+1) % num_epochs

    for i in range(num_epochs):
        counts = [1 for j in range(len(full_epochs[i]))]
        full_epochs[i] = DataSet.to_dict(full_epochs[i],counts)

    return full_epochs,partial_epochs

def main():
    rows = int(sys.argv[1])
    cols = int(sys.argv[2])
    num_epochs = 10
    


    vtree_filename = '../graphs/asdf-%d-%d.vtree' % (rows,cols)
    sdd_filename = '../graphs/asdf-%d-%d.sdd' % (rows,cols)

    psi,scale = 2.0,None # learning hyper-parameters
    N,M = 2**10,2**10 # size of training/testing dataset
    em_max_iters = 10 # maximum # of iterations for EM
    em_threshold = 1e-4 # convergence threshold
    seed = 1 # seed for simulating datasets

    ########################################
    # READ INPUT
    ########################################

    print "== reading vtree/sdd"

    vtree = Vtree.read(vtree_filename)
    manager = SddManager(vtree)
    sdd = SddNode.read(sdd_filename,manager)
    pmanager = PSddManager(vtree)
    copy = pmanager.copy_and_normalize_sdd(sdd,vtree)
    pmanager.make_unique_true_sdds(copy,make_true=False) #AC: set or not set?
    print "1"


    psdd_parameters = copy.theta_count()

    for alpha in [sdd,copy]:
        start = time.time()
        model_count = alpha.model_count()
        #print "      model count: %s (%.3fs)" % \
        #    (locale.format("%d",model_count,grouping=True),time.time()-start)

    full_datasets, partial_lists = epochs_partial(rows,cols,num_epochs,copy)

    for i in range(num_epochs):
        print len(partial_lists)
        total_counts = 0
        for model,count in full_datasets[i]:
            total_counts += count
        print total_counts
        print ""

    ########################################
    # SIMULATE
    ########################################

    print "2"


if __name__ == '__main__':
    main()
