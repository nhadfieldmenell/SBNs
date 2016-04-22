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
    full_tuple = map(tuple,full_ints)

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
    good_printed = 0

    if not file_exists: 
        cur_time = time.time()
        for i in range(len(full_tuple)):
            prev_time = cur_time
            cur_time = time.time()
            #if times_printed < 100:
            #    print "time to evaluate model %d: %f" % (i-1,cur_time-prev_time)
            model = full_tuple[i]
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
                    #print "Bad model:"
                    #draw_grid(model,rows,cols,edge2index)
                bad_models[str(model)] = True
                unique_bad += 1
                total_bad += 1
                bad_indices[i] = True
                continue

            else:
                if good_printed < 25:
                    good_printed += 1
                    print "Good model:"
                    draw_grid(model,rows,cols,edge2index)
                good_models[str(model)] = True
                unique_good += 1
                total_good += 1
                data.append(model)

        print "total bad: %d, unique bad: %d, total good: %d, unique good: %d" % (total_bad,unique_bad, total_good, unique_good)

        bad_file = open(bad_fn,'w')
        for i in bad_indices.keys():
            bad_file.write("%d\n" % i)
        bad_file.close()

        full_dataset = DataSet.to_dict(data)

        return full_dataset

    else:
        for i in range(len(full_tuple)):
            model = full_tuple[i]
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

        print "total bad: %d, unique bad: %d, total good: %d, unique good: %d" % (total_bad,unique_bad, total_good, unique_good)
        full_dataset = DataSet.to_dict(data)

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

def out_edges(rows,cols,node,edge2index):
    """Find all the edes that pass out of a node.

    Return:
        up: edge index passing out of the top of the node (-1 if in top row)
        down: edge index passing out of bottom (-1 if bottom row)
        left: edge index passing out of left (-1 if left col)
        right: edge index passing out of right (-1 if right col)
    """
    up = -1
    down = -1
    left = -1
    right = -1
    if node > cols:
        up = edge2index[(node-cols,node)]
    if node <= cols*(rows-1):
        down = edge2index[(node,node+cols)]
    if node % cols != 1:
        left = edge2index[(node-1,node)]
    if node % cols != 0:
        right = edge2index[(node,node+1)]

    return up,down,left,right


def neighbor_nodes(rows,cols,node):
    """Find all the nodes that neighbor a node.

    Return:
        A list of all the node indexes that neighbor the given node.
    """

    neighbors = []
    if node > cols:
        neighbors.append(node-cols)
    if node <= cols*(rows-1):
        neighbors.append(node+cols)
    if node % cols != 1:
        neighbors.append(node-1)
    if node % cols != 0:
        neighbors.append(node+1)

    return neighbors

def end_point(rows,cols,node,edge2index):
    """Find all the variables that should be set to represent a path ending at node.

    There are at most 4 sets of variable assignments to do this.
        Each one has 1 edge set to true and the other (1-3) edges set to false.

    Return a list
        Elements of the top-level list are len 2 lists
            First element of second level list is the edge to be set to true
            Second element of second level is list of edges to be set to false
    """
    assignments = []

    up,down,left,right = out_edges(rows,cols,node,edge2index)

    for pos_edge in up,down,left,right:
        if pos_edge != -1:
            asn = [pos_edge,[]]
            for neg_edge in up,down,left,right:
                if neg_edge != pos_edge and neg_edge != -1:
                    asn[1].append(neg_edge)
            assignments.append(asn)

    return assignments

def mid_point(rows,cols,node,edge2index):
    """Find all the variable assignments representing a path passing through node
    
    These are all valid pairs of 2 edges connected to the node.

    Return
        A list of lists of length 2.
        Each sublist is 2 edges taken.

    DO A SANITY CHECK THAT SETTING EXTRA EDGES TO 0 DOES NOTHIG TO PROBABILITY
    """
    assignments = []
    up,down,left,right = out_edges(rows,cols,node,edge2index)

    valid_edges = []
    for i in up,down,left,right:
        if i != -1:
            valid_edges.append(i)

    for i in range(len(valid_edges)):
        for j in range(i+1,len(valid_edges)):
            assignments.append([valid_edges[i],valid_edges[j]])

    return assignments

def prob_start_end_mid(rows,cols,start,end,mid,num_edges,edge2index,copy):
    """Probability that a path starts at start and ends at end and passes through mid.
    
    This value is NOT normalized by the probability that a path starts at star and ends at end.

    Return that probability as a float.
    """

    start_asgnmts = end_point(rows,cols,start,edge2index)
    end_asgnmts = end_point(rows,cols,end,edge2index)
    mid_asgnmts = mid_point(rows,cols,mid,edge2index)

    total_prob = 0.0
    for start_i in range(len(start_asgnmts)):
        for end_i in range(len(end_asgnmts)):
            for mid_i in range(len(mid_asgnmts)):
                start_a = start_asgnmts[start_i]
                end_a = end_asgnmts[end_i]
                mid_a = mid_asgnmts[mid_i]
                if start_a[1].count(mid_a[0]) != 0 or end_a[1].count(mid_a[0]) != 0:
                    continue
                if start_a[1].count(mid_a[1]) != 0 or end_a[1].count(mid_a[1]) != 0:
                    continue
                        
                data = [-1 for i in range(num_edges)]
                for one in mid_a:
                    data[one] = 1
                data[start_a[0]] = 1
                data[end_a[0]] = 1
                for zero in start_a[1]:
                    data[zero] = 0
                for zero in end_a[1]:
                    data[zero] = 0
                ones = []
                zeros = []
                for i in range(len(data)):
                    if data[i] == 0:
                        zeros.append(i)
                    if data[i] == 1:
                        ones.append(i)
                #print "start edge: %d, end edge: %d, mid edges: %d %d" % (start_a[0],end_a[0],mid_a[0],mid_a[1])
                #print "ones: %s" % str(ones)
                #print "zeros: %s" % str(zeros)
                data = tuple(data)
                evidence = DataSet.evidence(data)
                probability = copy.probability(evidence)
                #print "prob: %f" % probability
                total_prob += probability
                
    return total_prob 

def normalized_prob_mid(rows,cols,start,end,mid,num_edges,edge2index,copy):
    """Probability that a path starts at start, ends at end, and passes through mid, normalized.
    Normalizing factor is probability that the path starts at start and ends at end.
    Return normalized probability.
    """
    mid_prob = prob_start_end_mid(rows,cols,start,end,mid,num_edges,edge2index,copy)
    start_end_prob = prob_start_end(rows,cols,start,end,num_edges,edge2index,copy)
    #print "start end prob %d: %f" % (mid,start_end_prob)
    #print "mid prob %d: %f" % (mid,mid_prob)
    return mid_prob/start_end_prob

def prob_start_end(rows,cols,start,end,num_edges,edge2index,copy):
    """Probability that a path starts at start and ends at end.

    Reutrn that probability as a float.
    """

    start_asgnmts = end_point(rows,cols,start,edge2index)
    end_asgnmts = end_point(rows,cols,end,edge2index)

    total_prob = 0.0
    for start_i in range(len(start_asgnmts)):
        for end_i in range(len(end_asgnmts)):
            start_a = start_asgnmts[start_i]
            end_a = end_asgnmts[end_i]
            data = [-1 for i in range(num_edges)]
            data[start_a[0]] = 1
            data[end_a[0]] = 1
            for zero in start_a[1]:
                data[zero] = 0
            for zero in end_a[1]:
                data[zero] = 0
            data = tuple(data)
            evidence = DataSet.evidence(data)
            probability = copy.probability(evidence)
            total_prob += probability

    return total_prob

def visualize_mid_probs(rows,cols,start,end,num_edges,edge2index,copy):
    probs = []
    for i in range(rows):
        for j in range(cols):
            mid = i*cols + j + 1
            if mid == start:
                sys.stdout.write("start   ")
                continue
            if mid == end:
                sys.stdout.write(" end    ")
                continue
            prob_mid = normalized_prob_mid(rows,cols,start,end,mid,num_edges,edge2index,copy)
            sys.stdout.write("%.3f   " % prob_mid)
        sys.stdout.write("\n\n")

def perform_analysis(rows,cols,start,end,fn_prefix,data_fn,bad_fn,edge2index):
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

    training = filter_bad(copy,data_fn,bad_fn,rows,cols,edge2index)

    start_time = time.time()
    copy.learn(training,psi=psi,scale=scale,show_progress=True)
    print "== TRAINING =="
    print "    training time: %.3fs" % (time.time()-start_time)
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

    visualize_mid_probs(rows,cols,start,end,num_edges,edge2index,copy)

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

    fn_prefix_fixed = '../graphs/fixed_ends-%d-%d-%d-%d' % (rows,cols,start,end)
    data_fn_fixed = '../datasets/fixed_ends-%d-%d-%d-%d.txt' % (rows,cols,start,end)
    bad_fn_fixed = 'bad_paths/fixed_bad-%d-%d-%d-%d.txt' % (rows,cols,start,end)

    print "FIXED ENDPOINT PROBABILITIES"
    perform_analysis(rows,cols,start,end,fn_prefix_fixed,data_fn_fixed,bad_fn_fixed,edge2index)


    return
    fn_prefix_fixed = '../graphs/fixed_ends-%d-%d-%d-%d' % (rows,cols,start,end)
    vtree_filename_fixed = '%s.vtree' % fn_prefix_fixed
    sdd_filename_fixed = '%s.sdd' % fn_prefix_fixed

    psi,scale = 2.0,None # learning hyper-parameters
    N,M = 2**10,2**10 # size of training/testing dataset
    em_max_iters = 10 # maximum # of iterations for EM
    em_threshold = 1e-4 # convergence threshold
    seed = 1 # seed for simulating datasets

    ########################################
    # READ INPUT
    ########################################

    print "== reading vtree/sdd"

    vtree_fixed = Vtree.read(vtree_filename_fixed)
    manager_fixed = SddManager(vtree_fixed)
    sdd_fixed = SddNode.read(sdd_filename_fixed,manager_fixed)
    pmanager_fixed = PSddManager(vtree_fixed)
    copy_fixed = pmanager_fixed.copy_and_normalize_sdd(sdd_fixed,vtree_fixed)
    pmanager_fixed.make_unique_true_sdds(copy_fixed,make_true=False) #AC: set or not set?

    psdd_parameters_fixed = copy_fixed.theta_count()

    data_fn_fixed = '../datasets/fixed_ends-%d-%d-%d-%d.txt' % (rows,cols,start,end)
    bad_fn_fixed = 'bad_paths/fixed_bad-%d-%d-%d-%d.txt' % (rows,cols,start,end)
    training_fixed = filter_bad(copy_fixed,data_fn_fixed,bad_fn_fixed,rows,cols,edge2index)

    start_time = time.time()
    copy_fixed.learn(training_fixed,psi=psi,scale=scale,show_progress=True)
    print "== TRAINING =="
    print "    training time: %.3fs" % (time.time()-start_time)
    ll_fixed = copy_fixed.log_likelihood_alt(training_fixed)
    lprior_fixed = copy_fixed.log_prior(psi=psi,scale=scale)
    print "   training: %d unique, %d instances" % (len(training_fixed),training_fixed.N)
    print "   log likelihood: %.8f" % (ll_fixed/training_fixed.N)
    print "   log prior: %.8f" % (lprior_fixed/training_fixed.N)
    print "   log posterior: %.8f" % ((ll_fixed+lprior_fixed)/training_fixed.N)
    print "   log likelihood unnormalized: %.8f" % ll_fixed
    print "   log prior unnormalized: %.8f" % lprior_fixed
    print "   log posterior unnormalized: %.8f" % (ll_fixed+lprior_fixed)
    print "   log prior over parameters: %.8f" % (lprior_fixed/psdd_parameters_fixed)

    print "  zero parameters: %d (should be zero)" % copy_fixed.zero_count()
    copy_fixed.marginals()



    print "FIXED ENDPOINT PROBABILITIES"
    visualize_mid_probs(rows,cols,start,end,num_edges,edge2index,copy_fixed)

    return
    for i in range(1,37):
        probability_i = normalized_prob_mid(rows,cols,start,end,i,num_edges,edge2index,copy)
        print "prob of passing through %d: %f" % (i,probability_i)

    return
    #tot = 0.0 
    mid = 17
    tot +=  normalized_prob_mid(rows,cols,start,end,mid,num_edges,edge2index,copy)

    mid = 24
    tot += normalized_prob_mid(rows,cols,start,end,mid,num_edges,edge2index,copy)
    mid = 22 
    tot +=  normalized_prob_mid(rows,cols,start,end,mid,num_edges,edge2index,copy)
    mid = 29 
    tot +=  normalized_prob_mid(rows,cols,start,end,mid,num_edges,edge2index,copy)

    #print tot

    return
    total_prob = 0.0
    s_neighbors = (start,start-1,start+cols,start+cols-1) 
    e_neighbors = (end,end-1,end+cols)#,end+cols-1) 
    for s_n in s_neighbors:
        for e_n in e_neighbors:
            total_prob += prob_start_end(rows,cols,s_n,e_n,num_edges,edge2index,copy)

    print total_prob
    return


    """This stuff is for seeing the effects of time on paths"""
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
