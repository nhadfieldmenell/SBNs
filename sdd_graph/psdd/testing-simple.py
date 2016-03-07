#!/usr/bin/env python

import math
import time
import glob
import sys

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
    epoch0_name = "../uber-data_%d_%d_0.txt" % (rows,cols)
    epoch1_name = "../uber-data_%d_%d_1.txt" % (rows,cols)
    epoch2_name = "../uber-data_%d_%d_2.txt" % (rows,cols)
    epoch3_name = "../uber-data_%d_%d_3.txt" % (rows,cols)
    epoch4_name = "../uber-data_%d_%d_4.txt" % (rows,cols)
    epoch5_name = "../uber-data_%d_%d_5.txt" % (rows,cols)
    epoch6_name = "../uber-data_%d_%d_6.txt" % (rows,cols)
    epoch7_name = "../uber-data_%d_%d_7.txt" % (rows,cols)
    epoch8_name = "../uber-data_%d_%d_8.txt" % (rows,cols)
    epoch9_name = "../uber-data_%d_%d_9.txt" % (rows,cols)
    epoch0 = filter_bad(DataSet.read(epoch0_name),copy)
    epoch1 = filter_bad(DataSet.read(epoch1_name),copy)
    epoch2 = filter_bad(DataSet.read(epoch2_name),copy)
    epoch3 = filter_bad(DataSet.read(epoch3_name),copy)
    epoch4 = filter_bad(DataSet.read(epoch4_name),copy)
    epoch5 = filter_bad(DataSet.read(epoch5_name),copy)
    epoch6 = filter_bad(DataSet.read(epoch6_name),copy)
    epoch7 = filter_bad(DataSet.read(epoch7_name),copy)
    epoch8 = filter_bad(DataSet.read(epoch8_name),copy)
    epoch9 = filter_bad(DataSet.read(epoch9_name),copy)
    """
    
    epochs = []
    num_epochs = 10
    for i in range(num_epochs):
        epoch_name = "../datasets/uber-data_%d_%d_%d.txt" % (rows,cols,i)
        epoch = filter_bad(DataSet.read(epoch_name),copy)
        epochs.append(epoch)

 
    """
    epochs = [epoch0,epoch1,epoch2,epoch3,epoch4,epoch5,epoch6,epoch7,epoch8,epoch9]
    """

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
        print "   log likelihood: %.8f" % (ll/training.N)
        print "    log posterior: %.8f" % ((ll+lprior)/training.N)
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
        print "   log likelihood: %.8f" % (ll/testing.N)
        print "    log posterior: %.8f" % ((ll+lprior)/testing.N)

    avgLL = totalLL/float(10)
    print "average log likelihood: %f" % avgLL
