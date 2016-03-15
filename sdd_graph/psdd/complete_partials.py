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
    full_ints = map(lambda x: map(int,x[:-1].split(',')),full_lines)
    print full_ints[0]
    print len(full_ints[0])

    return


    print partials_lines[4]

    full_and_part = zip(map(int,full_lines),map(int,partials_lines))
    
    for i in range(20):
        for j in range(len(full_and_part[i])):
            if full_and_part[i][1] == 1:
                print "1 %d" % full_and_part[i][0]
            elif full_and_part[i][1] == 0:
                print "0 %d" % full_and_part[1][0]

    random.shuffle(full_and_part)




    return

def main():
    rows = int(sys.argv[1])
    cols = int(sys.argv[2])
    num_epochs = 10
    
    epochs_partial(rows,cols,num_epochs,0)

    return

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

    ########################################
    # SIMULATE
    ########################################

    print "2"


if __name__ == '__main__':
    main()
