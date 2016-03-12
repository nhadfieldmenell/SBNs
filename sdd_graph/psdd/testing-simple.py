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

def filter_bad(dataset,copy):
    badCount = 0
    goodCount = 0
    models = []
    counts = []
    copy.uniform_weights()
    for model, count in dataset:
        evidence = DataSet.evidence(model)
        probability = copy.probability(evidence)
        if probability == 0:
            print "%d: %s" % (count,model)
            badCount += count
        else:
            goodCount += count
            models.append(model)
            counts.append(count)

    print "bad count: %d" % badCount
    print "good count: %d" % goodCount

    return DataSet.to_dict(models,counts)

def create_epochs(rows,cols,num_epochs,copy):
    """Creates a list of datasets, each of the same size and with randomized data.

    Randomize the data and create num_epochs datasets, each of the same size.

    Returns:
        a list of datasets, all the same size.
    """
    filename = "../datasets/full_data_%d_%d.txt"%(rows,cols)
    file = open(filename,"r")
    lines = file.readlines()
    random.shuffle(lines)
    file.close()
    file = open(filename,"w")
    for i in lines:
        file.write(str(i))
    file.close()
    dataset = DataSet.read(filename)
    goods = filter_bad(dataset,copy)
    counter = 0
    models = [[] for i in range(num_epochs)]
    counts = [[] for i in range(num_epochs)]
    for model, count in goods:
        for i in range(count):
            index = counter % num_epochs
            models[index].append(model)
            counts[index].append(1)
            counter += 1

    sets = []
    for i in range(num_epochs):
        sets.append(DataSet.to_dict(models[i],counts[i]))

    return sets



def model_str(model,n):
    """pretty print model"""

    keys = model.keys()
    keys.sort()
    st = []
    for i,key in enumerate(keys):
        val = str(model[key])
        if i > 0 and i % n == 0:
            st.append(',')
        st.append(val)
    return "".join(st)

if __name__ == '__main__':
    rows = int(sys.argv[1])
    cols = int(sys.argv[2])

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

    """
    print "         sdd size: %d" % sdd.size()
    print "           sdd nc: %d" % sdd.node_count()
    print "        psdd size: %d" % copy.size()
    print "          psdd nc: %d" % copy.node_count()
    print "  psdd parameters: %d" % copy.theta_count()
    print "       psdd trues: %d" % copy.true_count()
    """

    for alpha in [sdd,copy]:
        start = time.time()
        model_count = alpha.model_count()
        #print "      model count: %s (%.3fs)" % \
        #    (locale.format("%d",model_count,grouping=True),time.time()-start)

    ########################################
    # SIMULATE
    ########################################

    print "2"

    """
    start = time.time()
    copy.random_weights(psi=1.0) # set random weights on PSDD
    complete = DataSet.simulate(copy,N,seed=seed)
    training = complete.hide_values_at_random(0.25,seed=seed)
    if type(seed) is int or type(seed) is long: seed = seed+1 # update seed
    testing  = DataSet.simulate(copy,M,seed=seed)
    print "simulate datasets: %.3fs" % (time.time()-start)
    print "         training: %d unique, %d instances" % (len(training),training.N)
    print "          testing: %d unique, %d instances" % (len(testing),testing.N)
    """
    """ 
    epochs = []
    num_epochs = 10
    for i in range(num_epochs):
        epoch_name = "../datasets/uber-data_%d_%d_%d.txt" % (rows,cols,i)
        epoch = filter_bad(DataSet.read(epoch_name),copy)
        epochs.append(epoch)

    """
    """
    epochs = [epoch0,epoch1,epoch2,epoch3,epoch4,epoch5,epoch6,epoch7,epoch8,epoch9]
    """
    
    epochs = create_epochs(rows,cols,10,copy)

    totalLL = 0

    """
    testing_name = "../uber_testing_%d_%d.txt" % (rows,cols)
    training = DataSet.read(training_name)
    testing = DataSet.read(testing_name)
    training = filter_bad(training,copy)
    testing = filter_bad(testing,copy)
    """
    if type(seed) is int or type(seed) is long: seed = seed+1 # update seed

    print "3"




    print "4"

    ########################################
    # LEARN
    ########################################

    for i in range(10):
        testing  = epochs[i]
        
        models = []
        counts = []
        for j in range(10):
            if j != i:
                for model,count in epochs[j]:
                    models.append(model)
                    counts.append(count)

        training = DataSet.to_dict(models,counts)

        # for complete data, for testing purposes
        start = time.time()
        copy.learn(training,psi=psi,scale=scale,show_progress=True)
        print "    training time: %.3fs" % (time.time()-start)
        ll = copy.log_likelihood_alt(training)
        lprior = copy.log_prior(psi=psi,scale=scale)
        print "       training.N: %d" % training.N
        print "   log likelihood: %.8f" % (ll/training.N)
        print "   log prior: %.8f" % (lprior/training.N)
        print "    log posterior: %.8f" % ((ll+lprior)/training.N)
        print "   log likelihood unnormalized: %.8f" % ll
        print "   log prior unnormalized: %.8f" % lprior
        print "    log posterior unnormalized: %.8f" % (ll+lprior)

        print "  zero parameters: %d (should be zero)" % copy.zero_count()
        copy.marginals()

        

        """
        #for incomplete data
        start = time.time()
        copy.random_weights(psi=1.0) # initial seed for EM
        stats = copy.soft_em(training,psi=psi,scale=scale,
                             threshold=em_threshold,max_iterations=em_max_iters)
        #ll = stats.ll
        #ll = copy.log_likelihood_alt(training)
        ll = copy.log_likelihood(training)
        lprior = copy.log_prior(psi=psi,scale=scale)
        print "    training time: %.3fs (iters: %d)" % (time.time()-start,stats.iterations)
        print "   log likelihood: %.8f" % (ll/training.N)
        print "    log posterior: %.8f" % ((ll+lprior)/training.N)
        print "  zero parameters: %d (should be zero)" % copy.zero_count()
        """

        ########################################
        # TEST
        ########################################

        print "== TESTING =="
        ll = copy.log_likelihood_alt(testing)
        totalLL += ll/testing.N
        print "        testing.N: %d" % testing.N
        print "   log likelihood: %.8f" % (ll/testing.N)
        print "   log prior: %.8f" % (lprior/testing.N)
        print "    log posterior: %.8f" % ((ll+lprior)/testing.N)
        print "   log likelihood unnormalized: %.8f" % ll
        print "   log prior unnormalized: %.8f" % lprior
        print "    log posterior unnormalized: %.8f" % (ll+lprior)

    avgLL = totalLL/float(10)
    print "average log likelihood: %f" % avgLL
