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

def evaluate_prediction(prediction,full,partial):
    correctly_guessed = 0
    incorrectly_guessed = 0
    not_guessed = 0

    for i in range(len(prediction)):
        if prediction[i] == 1:
            if full[i] == 1:
                if not partial[i] == 1:
                    correctly_guessed += 1
            else:
                incorrectly_guessed += 1
    actual_num_edges = full.count(1)-partial.count(1)
    not_guessed = actual_num_edges - correctly_guessed
    print "Correctly guessed edges: %d" % correctly_guessed
    print "Incorrectly guessed edges: %d" % incorrectly_guessed
    print "Not guessed edges: %d" % not_guessed

    return correctly_guessed,incorrectly_guessed,not_guessed

def print_3(partial,mpe,full,rows,cols,edge2index):
    """Draw grids for all 3 paths, followed by an extra newline"""

    print "Partial"
    draw_grid(partial,rows,cols,edge2index)
    print "Predicted"
    draw_grid(mpe,rows,cols,edge2index)
    print "Actual"
    draw_grid(full,rows,cols,edge2index)

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

def epochs_partial(rows,cols,num_epochs,copy):
    """Randomize the data instances and separate the data instances and partials into equal collections

    Keep relationship between full and partial data.

    Returns:
        A list of num_epochs full datasets.

        A list of num_epochs arrays of full_data tuples.

        A list of num_epochs list of arrays.
            Each array holds multiple incomplete data instances.
            The different epochs have the same data instances as the corresponding epochs in full datasets.
    """

    full_file = open("../datasets/full_data_%d_%d.txt" % (rows,cols), "r")
    partials_file = open("../datasets/partials_%d_%d.txt" % (rows,cols), "r")
    full_lines = full_file.readlines()
    partials_lines = partials_file.readlines()
    full_file.close()
    partials_file.close()

    bad_lines = None
    bad_paths = {}
    bad_filename = "bad_paths/bad_%d_%d.txt" % (rows,cols)
    file_exists = os.path.isfile(bad_filename)
    if file_exists:
        bad_file = open(bad_filename,'r')
        bad_lines = bad_file.readlines()
        bad_file.close()
        for i in bad_lines:
            bad_paths[int(i)] = True

    full_ints = map(lambda x: map(int,x[:-1].split(',')),full_lines)
    part_ints = map(lambda x: map(int,x[:-1].split(',')),partials_lines)

    full_tuple = map(tuple,full_ints)
    part_tuple = map(tuple,part_ints)

    full_and_part = zip(full_tuple,part_tuple)
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
    
    #random.shuffle(full_and_part)

    epoch_num = 0
    full_epochs = [[] for i in range(num_epochs)]
    partial_epochs = [[] for i in range(num_epochs)]

    copy.uniform_weights()
    bad_models = {}
    unique_bad = 0
    total_bad = 0
    total_good = 0

    bad_indices = {}

    if not file_exists: 
        for i in range(len(full_and_part)):
            model = full_and_part[i][0]
            partial_model = full_and_part[i][1]
            if str(model) in bad_models:
                total_bad += 1
                bad_indices[i] = True
                continue
            evidence = DataSet.evidence(model)
            probability = copy.probability(evidence)
            if probability == 0:
                #print "bad: %s" % str(model)
                bad_models[str(model)] = True
                unique_bad += 1
                total_bad += 1
                bad_indices[i] = True
                continue

            else:
                total_good += 1
                full_epochs[epoch_num].append(model)
                partial_epochs[epoch_num].append(partial_model)
                epoch_num = (epoch_num+1) % num_epochs

        print "total bad: %d, unique bad: %d, total good: %d" % (total_bad,unique_bad, total_good)
        full_datasets = []

        bad_file = open(bad_filename,'w')
        for i in bad_indices.keys():
            bad_file.write("%d\n" % i)
        bad_file.close()

        for i in range(num_epochs):
            counts = [1 for j in range(len(full_epochs[i]))]
            full_datasets.append(DataSet.to_dict(full_epochs[i],counts))

        return full_datasets,full_epochs,partial_epochs

    else:
        for i in range(len(full_and_part)):
            model = full_and_part[i][0]
            partial_model = full_and_part[i][1]
            if i in bad_paths:
                bad_models[str(model)] = True
                unique_bad += 1
                total_bad += 1
                continue

            else:
                total_good += 1
                full_epochs[epoch_num].append(model)
                partial_epochs[epoch_num].append(partial_model)
                epoch_num = (epoch_num+1) % num_epochs

        print "total bad: %d, total good: %d" % (len(bad_paths), total_good)
        full_datasets = []

        for i in range(num_epochs):
            counts = [1 for j in range(len(full_epochs[i]))]
            full_datasets.append(DataSet.to_dict(full_epochs[i],counts))

        print "num bad paths: %d" % len(bad_paths)
        return full_datasets,full_epochs,partial_epochs

def most_likely_completions(full_datasets,partial_epochs,num_epochs,rows,cols,edge2index):
    for i in range(3):
        models = []
        counts = []
        for j in range(num_epochs):
            if j != i:
                for model,count in full_datasets[j]:
                    models.append(model)
                    counts.append(count)

        training = DataSet.to_dict(models,counts)
        full_instances = []
        total_count = 0
        for model,count in training:
            full_instances.append([model,count])
            total_count += count
        print "Total count: %d" % total_count
        observed_partials = {}
        for part_inst in partial_epochs[i]:
            if tuple(part_inst) in observed_partials:
                continue
            observed_partials[tuple(part_inst)] = True
            possibles = [k for k in range(len(full_instances))]
            for j in range(len(part_inst)):
                if part_inst[j] == 1:
                    to_pop = []
                    for k in range(len(full_instances)):
                        if full_instances[k][0][j] != 1:
                            to_pop.insert(0,k)
                    for k in to_pop:
                        full_instances.pop(k)
                elif part_inst[j] == 0:
                    to_pop = []
                    for k in range(len(full_instances)):
                        if full_instances[k][0][j] != 1:
                            to_pop.insert(0,k)
                    for k in to_pop:
                        full_instances.pop(k)
            heap = []
            for j in range(len(full_instances)):
                heapq.heappush(heap,(full_instances[j][1],full_instances[j][0]))
            print "Partial"
            draw_grid(part_inst,rows,cols,edge2index)
            for j in range(3):
                if len(heap) == 0:
                    break
                count,model = heapq.heappop(heap)
                print count
                draw_grid(model,rows,cols,edge2index)
                
            



def main():
    rows = int(sys.argv[1])
    cols = int(sys.argv[2])
    num_epochs = 10
    edge_filename = '../graphs/edge-nums-%d-%d.pickle' % (rows,cols)
    edge2index = pickle.load(open(edge_filename,'rb'))
    num_edges = (rows-1)*cols + (cols-1)*rows
    


    vtree_filename = '../graphs/asdf-%d-%d.vtree' % (rows,cols)
    sdd_filename = '../graphs/asdf-%d-%d.sdd' % (rows,cols)


    vtree_filename_no_mp = '../graphs/asdf-no-mp-%d-%d.vtree' % (rows,cols)
    sdd_filename_no_mp = '../graphs/asdf-no-mp-%d-%d.sdd' % (rows,cols)

    psi,scale = 2.0,None # learning hyper-parameters
    N,M = 2**10,2**10 # size of training/testing dataset
    em_max_iters = 10 # maximum # of iterations for EM
    em_threshold = 1e-4 # convergence threshold
    seed = 1 # seed for simulating datasets

    ########################################
    # READ INPUT
    ########################################

    print "== reading vtree/sdd"


    """No Midpoint SDD Stuff"""
    vtree_no_mp = Vtree.read(vtree_filename_no_mp)
    manager_no_mp = SddManager(vtree_no_mp)
    sdd_no_mp = SddNode.read(sdd_filename_no_mp,manager_no_mp)
    pmanager_no_mp = PSddManager(vtree_no_mp)
    copy_no_mp = pmanager_no_mp.copy_and_normalize_sdd(sdd_no_mp,vtree_no_mp)
    pmanager_no_mp.make_unique_true_sdds(copy_no_mp,make_true=False) #AC: set or not set?


    vtree = Vtree.read(vtree_filename)
    manager = SddManager(vtree)
    sdd = SddNode.read(sdd_filename,manager)
    pmanager = PSddManager(vtree)
    copy = pmanager.copy_and_normalize_sdd(sdd,vtree)
    pmanager.make_unique_true_sdds(copy,make_true=False) #AC: set or not set?


    psdd_parameters = copy.theta_count()

    for alpha in [sdd,copy]:
        start = time.time()
        model_count = alpha.model_count()
        #print "      model count: %s (%.3fs)" % \
        #    (locale.format("%d",model_count,grouping=True),time.time()-start)


    full_datasets, full_instances, partial_instances = epochs_partial(rows,cols,num_epochs,copy_no_mp)
    
    
    partials_completed = []
    for i in range(num_epochs):
        partials_completed.append([])
        for j in range(len(partial_instances[i])):
            partials_completed[i].append([])
            for k in range(len(partial_instances[i][j])):
                if partial_instances[i][j][k] == 1:
                    partials_completed[i][j].append(1)
                else:
                    partials_completed[i][j].append(0)

    most_likely_completions(full_datasets,partial_instances,num_epochs,rows,cols,edge2index)
    return
    """
    for i in range(num_epochs):
        print len(partial_instances[i])
        print len(full_instances[i])
        total_counts = 0
        for model,count in full_datasets[i]:
            total_counts += count
        print total_counts
        print ""
    """

    total_mpe_val = 0.0
    num_evaluated = 0.0
    total_correct = 0
    total_incorrect = 0
    total_not_guessed = 0
    total_fully_guessed = 0
    total_only_guessed_one = 0
    for i in range(num_epochs):
        testing  = full_datasets[i]
        
        models = []
        counts = []
        for j in range(num_epochs):
            if j != i:
                for model,count in full_datasets[j]:
                    models.append(model)
                    counts.append(count)

        training = DataSet.to_dict(models,counts)

        # for complete data, for testing purposes
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

        print "== best-m MPE =="
        count = 0
        mpe = []
        for val,model in copy.enumerate():
            if count == 0: mpe = model
            count += 1
            val = val/copy.theta_sum
            check_val = copy.probability(evidence=model)
            print "%.6e (%.6e): %s" % (val,check_val,str(model))
            model_array = []
            for k in range(num_edges):
                model_array.append(model[k+1])
            draw_grid(model_array,rows,cols,edge2index)
            if count == 10: break

        #continue

        for j in range(len(partial_instances[i])):
            evidence = DataSet.evidence(partial_instances[i][j])
            mpe_val, mpe_inst = copy.mpe(evidence)
            #print mpe_val
            total_mpe_val += mpe_val
            #print mpe_inst
            mpe_array = []
            for k in range(len(full_instances[i][j])):
                if mpe_inst[k+1] == 1:
                    mpe_array.append(1)
                else:
                    mpe_array.append(0)
            #print full_instances[i][j]
            print_3(partials_completed[i][j],mpe_array,full_instances[i][j],rows,cols,edge2index)
            correct,incorrect,not_guessed = evaluate_prediction(mpe_array,full_instances[i][j],partial_instances[i][j])
            total_correct += correct
            total_incorrect += incorrect
            total_not_guessed += not_guessed
            num_evaluated += 1
            if not_guessed == 0  and incorrect == 0:
                total_fully_guessed += 1
            if mpe_array.count(1) == partials_completed[i][j].count(1) + 1:
                total_only_guessed_one += 1
                print "Only guessed one!"

    #return
    average_correct = total_correct/num_evaluated
    average_incorrect = total_incorrect/num_evaluated
    average_not_guessed = total_not_guessed/num_evaluated
    average_mpe = total_mpe_val/num_evaluated
    print "Total Evaluated: %.8f" % num_evaluated
    print "average correct: %.8f" % average_correct
    print "average incorrect: %.8f" % average_incorrect
    print "average not guessed: %.8f" % average_not_guessed
    print "average mpe val: %.8f" % average_mpe
    print "total fully guessed: %d" % total_fully_guessed
    print "total only guessed one: %d" % total_only_guessed_one

    ########################################
    # SIMULATE
    ########################################



if __name__ == '__main__':
    main()
