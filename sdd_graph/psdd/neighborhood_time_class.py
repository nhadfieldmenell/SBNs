#!/usr/bin/env python

import math
import time
import glob
import sys
import os.path
import random
import pickle
import heapq

from collections import defaultdict
from pypsdd import *

# for printing numbers with commas
import locale
locale.setlocale(locale.LC_ALL, "en_US.UTF8")


def draw_grid(model,m,n,edge2index):
    for i in xrange(m):
        for j in xrange(n):
            sys.stdout.write('.')
            if j < n-1:
                """
                edge = ((i,j),(i,j+1))
                index = g.edge_to_index[edge] + 1
                """
                edge = (i*m+j+1,i*m+j+2)
                index = edge2index[edge]
                sys.stdout.write('-' if model[index] else ' ')
        sys.stdout.write('\n')
        if i < m-1:
            for j in xrange(n):
                """
                edge = ((i,j),(i+1,j))
                index = g.edge_to_index[edge] + 1
                """
                edge = (i*m+j+1,i*m+m+j+1)
                index = edge2index[edge]
                sys.stdout.write('|' if model[index] else ' ')
                sys.stdout.write(' ')
        sys.stdout.write('\n')

def filter_bad(copy,in_fn,bad_fn,rows,cols,edge2index):
    """Create a dataset from the file that consists of only models that are consistent with the sdd
    If there is a file that already contains the indices of the bad paths, then don't recompute.
    If there is no such file, find the bad paths and store their indices in the file with name bad_fn.
    '../datasets/first_last-%d-%d-%d-%d-%d' % (rows,cols,start,end,run)
    """



    full_file = open(in_fn, "r")
    full_lines = full_file.readlines()
    full_file.close()

    full_ints = map(lambda x: map(int,x[:-1].split(',')),full_lines)

    bad_lines = None
    bad_paths = {}
    file_exists = os.path.isfile(bad_fn)
    if file_exists:
        bad_file = open(bad_fn,'r')
        bad_lines = bad_file.readlines()
        bad_file.close()
        for i in bad_lines:
            bad_paths[int(i)] = True

    data = []

    copy.uniform_weights()
    bad_models = {}
    unique_bad = 0
    total_bad = 0
    total_good = 0
    unique_good = 0

    bad_indices = {}
    good_models = {}
    bad_printed = 0
    times_printed = 0

    if not file_exists: 
        cur_time = time.time()
        for i in range(len(full_ints)):
            prev_time = cur_time
            cur_time = time.time()
            if times_printed < 100:
                print "time to evaluate model %d: %f" % (i-1,cur_time-prev_time)
            model = full_ints[i]
            if str(model) in good_models:
                total_good += 1
                data.append(model)
                continue
            if str(model) in bad_models:
                total_bad += 1
                bad_indices[i] = True
                continue
            evidence = DataSet.evidence(model)
            probability = copy.probability(evidence)
            if probability == 0:
                if bad_printed < 25:
                    bad_printed += 1
                    print "Bad model:"
                    draw_grid(model,rows,cols,edge2index)
                bad_models[str(model)] = True
                unique_bad += 1
                total_bad += 1
                bad_indices[i] = True
                continue

            else:
                good_models[str(model)] = True
                unique_good += 1
                total_good += 1
                data.append(model)

        print "total bad: %d, unique bad: %d, total good: %d, unique good: %d" % (total_bad,unique_bad, total_good, unique_good)
        full_dataset = []

        bad_file = open(bad_fn,'w')
        for i in bad_indices.keys():
            bad_file.write("%d\n" % i)
        bad_file.close()

        counts = [1 for j in range(len(full_ints))]
        full_dataset.append(DataSet.to_dict(data,counts))

        return full_dataset

    else:
        for i in range(len(full_ints)):
            model = full_ints[i]
            if i in bad_paths:
                bad_models[str(model)] = True
                unique_bad += 1
                total_bad += 1
                continue

            else:
                if str(model) in good_models:
                    total_good += 1
                    data.append(model)
                    continue
                else:
                    good_models[str(model)] = True
                    unique_good += 1
                    total_good += 1
                    data.append(model)

        print "total bad: %d, total good: %d" % (len(bad_paths), total_good)
        full_dataset = []
        counts = [1 for j in range(len(full_ints))]
        full_dataset.append(DataSet.to_dict(data,counts))

        print "num bad paths: %d" % len(bad_paths)
        return full_dataset

            
def enumerate_mpe(copy,num_enumerate,evidence,num_edges,edge2index,rows,cols):
    print "== best-m MPE =="
    count = 0
    mpe = []
    for val,model in copy.enumerate(evidence):
        if count == 0: mpe = model
        count += 1
        val = val/copy.theta_sum
        check_val = copy.probability(evidence=model)
        print "%.6e (%.6e): %s" % (val,check_val,str(model))
        model_array = []
        for k in range(num_edges):
            model_array.append(model[k+1])
        draw_grid(model_array,rows,cols,edge2index)
        if count == num_enumerate: break

def tuple_to_node(point,cols):
    return cols*point[0] + point[1] + 1

def node_to_tuple(node_num,cols):
    row = (node_num-1) / cols
    col = (node_num-1) % cols
    return (row,col)

def main():
    rows = int(sys.argv[1])
    cols = int(sys.argv[2])
    start = int(sys.argv[3])
    end = int(sys.argv[4])
    edge_filename = '../graphs/edge-nums-%d-%d.pickle' % (rows,cols)
    edge2index = pickle.load(open(edge_filename,'rb'))
    edge_tuple_filename = '../graphs/edge-to-tuple-%d-%d.pickle' % (rows,cols)
    edge_index2tuple = pickle.load(open(edge_tuple_filename,'rb'))
    num_edges = (rows-1)*cols + (cols-1)*rows
    empty_data = [-1 for i in range(num_edges)]
    empty_data = tuple(empty_data)
    empty_evidence = DataSet.evidence(empty_data)

    

    fn_prefix = '../graphs/start_end-%d-%d-%d-%d' % (rows,cols,start,end)
    vtree_filename = '%s.vtree' % fn_prefix
    sdd_filename = '%s.sdd' % fn_prefix 

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

    psdd_parameters = copy.theta_count()

    filter_bad(copy,'../datasets/first_last-6-6-4-28-0.txt','bad_paths/taxi-6-6-4-28-0.txt',rows,cols,edge2index)

    return


    psdds = []
    for class_num in range(6):
        pmanager = PSddManager(vtree)
        copy = pmanager.copy_and_normalize_sdd(sdd,vtree)
        pmanager.make_unique_true_sdds(copy,make_true=False) #AC: set or not set?
        psdd_parameters = copy.theta_count()

        fn = '../datasets/start_end-%d-%d-%d-%d-%d.txt' % (rows,cols,start,end,class_num)
        training = DataSet.read(fn)

        start = time.time()
        copy.learn(training,psi=psi,scale=scale,show_progress=True)
        print "== TRAINING =="
        print "    training time: %.3fs" % (time.time()-start)
        ll = copy.log_likelihood_alt(training)
        lprior = copy.log_prior(psi=psi,scale=scale)
        print "   training: %d unique, %d instances" % (len(training),training.N)
        print "   log likelihood: %.8f" % (ll/training.N)
        print "   log prior: %.8f" % (lprior/training.N)
        print "   log posterior: %.8f" % ((ll+lprior)/training.N)
        print "   log likelihood unnormalized: %.8f" % ll
        print "   log prior unnormalized: %.8f" % lprior
        print "   log posterior unnormalized: %.8f" % (ll+lprior)
        print "   log prior over parameters: %.8f" % (lprior/psdd_parameters)

        print "  zero parameters: %d (should be zero)" % copy.zero_count()
        copy.marginals()

        psdds.append(copy)

    for i in range(6):
        for j in range(i+1,6):
            kl_divergence = psdd.kl(psdds[i],psdds[j])
            print "kl (%d,%d): %d" % (i,j,kl_divergence)

if __name__ == '__main__':
    main()
