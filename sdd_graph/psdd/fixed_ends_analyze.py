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

class PathManager(object):
    def __init__(self,rows,cols,edge2index,edge_index2tuple,copy=None):
        self.rows = rows
        self.cols = cols
        self.num_edges = (rows-1) * cols + (cols-1) * rows
        self.edge2index = edge2index
        self.edge_index2tuple = edge_index2tuple
        self.paths = []
        self.copy = copy

    def draw_grid(self,model):
        m = self.rows
        n = self.cols
        for i in xrange(m):
            for j in xrange(n):
                sys.stdout.write('.')
                if j < n-1:
                    edge = (i*m+j+1,i*m+j+2)
                    index = self.edge2index[edge]
                    sys.stdout.write('-' if model[index] == 1 else ' ')
            sys.stdout.write('\n')
            if i < m-1:
                for j in xrange(n):
                    edge = (i*m+j+1,i*m+m+j+1)
                    index = self.edge2index[edge]
                    sys.stdout.write('|' if model[index] == 1 else ' ')
                    sys.stdout.write(' ')
            sys.stdout.write('\n')

    def draw_edge_probs(self,model,edge_num2prob,start,end):
        m = self.rows
        n = self.cols
        for i in xrange(m):
            sys.stdout.write("  ")
            for j in xrange(n):
                if i*n + j + 1 == start:
                    sys.stdout.write('s')
                elif i*n + j + 1 == end:
                    sys.stdout.write('e')
                else:
                    sys.stdout.write('.')
                if j < n-1:
                    edge = (i*m+j+1,i*m+j+2)
                    index = self.edge2index[edge]
                    if model[index] == 1:
                        sys.stdout.write(' ----- ')
                    elif index in edge_num2prob.keys():
                        sys.stdout.write(' %s ' % ('%.4f' % edge_num2prob[index])[1:])
                    else:
                        sys.stdout.write('       ')
            sys.stdout.write('\n')
            if i < m-1:
                for j in xrange(n):
                    edge = (i*m+j+1,i*m+m+j+1)
                    index = self.edge2index[edge]
                    if model[index] == 1:
                        sys.stdout.write('  |     ')
                    elif index in edge_num2prob.keys():
                        sys.stdout.write('%s   ' % ('%.4f' % edge_num2prob[index])[1:])
                    else:
                        sys.stdout.write('        ')
            sys.stdout.write('\n')

    def most_likely_start(self,start,end):
        """Find the most likely start edge given the start node and end node.

        Return:
            edge to be set to true
            list of edges to be set to false
        """
        start_asgnmts = self.end_point(start)
        end_asgnmts = self.end_point(end)

        start_end_prob = self.prob_start_end(start,end)

        edge_num2prob = {}
        best_prob = 0.0
        best_i = 0
        for s_i in range(len(start_asgnmts)):
            total_prob = 0.0
            s_a = start_asgnmts[s_i]
            s_edge = s_a[0]
            for e_i in range(len(end_asgnmts)):
                e_a = end_asgnmts[e_i]
                p = Path(self)
                p.add_and_neg_edges([e_a[0]],e_a[1])
                p.add_and_neg_edges([s_a[0]],s_a[1])
                #print p.model_tuple()
                evidence = DataSet.evidence(p.model_tuple())
                path_prob = self.copy.probability(evidence)
                #p.ones_and_zeros()
                #print "path probability normalized: %.6f" % (path_prob/start_end_prob)
                total_prob += path_prob
            normalized_prob = total_prob/start_end_prob
            edge_num2prob[s_edge] = normalized_prob
            #print "total prob: %.6f" % total_prob
            #print "Probability of taking edge %d: %.6f" % (start_asgnmts[s_i][0],total_prob/start_end_prob)
            if total_prob > best_prob:
                best_prob = total_prob
                best_i = s_i


        self.draw_edge_probs([-1 for i in range(self.num_edges)],edge_num2prob,start,end)
        return start_asgnmts[best_i]

    def mpe(self,evidence):
        """Find the most likely total assignment based on the partial evidence.

        Return the probability and the full variable assignment.
        """
        print "== MPE =="
        count = 0
        mpe = []
        for val,model in self.copy.enumerate(evidence):
            if count == 0: mpe = model
            count += 1
            val = val/self.copy.theta_sum
            check_val = self.copy.probability(evidence=model)
            print "%.6e (%.6e): %s" % (val,check_val,str(model))
            model_array = []
            for k in range(self.num_edges):
                model_array.append(model[k+1])
            self.draw_grid(model_array)
            return val,model_array



    def partial_prob(self,partial,end):
        """Find the probability of a path with the given partial path and an end at node end.

        Sum over all possible ends.

        Return the probability.
        """

        total_prob = 0.0
        end_asgnmts = self.end_point(end)
        for end_asgnmt in end_asgnmts:
            path = Path(self,partial[:])
            path.add_and_neg_edges([end_asgnmt[0]],end_asgnmt[1])
            evidence = DataSet.evidence(path.model_tuple())
            path_prob = self.copy.probability(evidence)
            total_prob += path_prob

        return total_prob



    def most_likely_next(self,node,prev_node,start,end,cur_path):
        """Find the most likely next edge to be taken conditioned on end point and current path.

        Attributes:
            node: current node
            prev_node: previous visited node
            end: end node
            cur_path: instantiations of -1,0,1 for the path (equivalent to a Path.model)

        Return the edge index to be set to positive.
        """
        neighbors = self.neighbor_nodes(node)
        end_asgnmts = self.end_point(end)

        possible_edges = []
        for neighbor in neighbors:
           if neighbor != prev_node:
               possible_edges.append(self.edge2index[(min(neighbor,node),max(neighbor,node))])
        
        partial_prob = self.partial_prob(cur_path[:],end)
        edge_num2prob = {}
        best_prob = 0.0
        best_i = 0
        for s_i in range(len(possible_edges)):
            total_prob = 0.0
            edge = possible_edges[s_i]
            for e_i in range(len(end_asgnmts)):
                e_a = end_asgnmts[e_i]
                p = Path(self,cur_path[:])
                if p.add_and_neg_edges([e_a[0]],e_a[1]) == -1:
                    continue
                if p.add_edge(edge) == -1:
                    continue
                
                evidence = DataSet.evidence(p.model_tuple())
                path_prob = self.copy.probability(evidence)
                total_prob += path_prob
            normalized_prob = total_prob/partial_prob
            edge_num2prob[edge] = normalized_prob
            #print "total prob: %.6f" % total_prob
            #print "Probability of taking edge %d: %.6f" % (start_asgnmts[s_i][0],total_prob/start_end_prob)
            if total_prob > best_prob:
                best_prob = total_prob
                best_i = s_i


        print ""
        print ""
        self.draw_edge_probs(cur_path,edge_num2prob,start,end)
        return possible_edges[best_i]



    def best_all_at_once(self,start,end):
        end_asgnmts = self.end_point(end)
        start_asgnmts = self.end_point(start)


        best_model = None
        best_prob = 0.0
        for s_a in start_asgnmts:
            for e_a in end_asgnmts:
                inst = [-1 for i in range(self.num_edges)]
                inst[s_a[0]] = 1
                inst[e_a[0]] = 1
                for neg_edge in s_a[1]:
                    inst[neg_edge] = 0
                for neg_edge in e_a[1]:
                    inst[neg_edge] = 0
                evidence = DataSet.evidence(inst)
                val,model = self.mpe(evidence)
                if val > best_prob:
                    best_prob = val
                    best_model = model

        self.draw_grid(best_model)
        return best_model

        


    def best_step_by_step(self,start,end):
        """Find the most likely path to be taken between start and end"""
        path = Path(self)
        cur_node = start

        pos_edge,neg_edges = self.most_likely_start(start,end)

        if path.add_and_neg_edges([pos_edge],neg_edges) == -1:
            print "INVALID PATH"
            return -1

        prev_node = start
        incident_nodes = self.edge_index2tuple[pos_edge]
        if cur_node == incident_nodes[0]:
            cur_node = incident_nodes[1]
        else:
            cur_node = incident_nodes[0]

        while cur_node != end:
            next_edge = self.most_likely_next(cur_node,prev_node,start,end,path.model[:])
            prev_node = cur_node
            incident_nodes = self.edge_index2tuple[next_edge]
            if cur_node == incident_nodes[0]:
                cur_node = incident_nodes[1]
            else:
                cur_node = incident_nodes[0]
            if path.add_edge(next_edge) == -1:
                print "INVALID ADDING EDGE %d" % next_edge

        self.draw_edge_probs(path.model[:],{},start,end)

        return path.model[:]


                

    def tuple_to_node(self,point):
        """Given a node represented as a tuple return corresponding node index"""
        return self.cols*point[0] + point[1] + 1

    def node_to_tuple(self,node_num):
        """Given a node index, return corresponding node tuple"""
        row = (node_num-1) / self.cols
        col = (node_num-1) % self.cols
        return (row,col)

    def out_edges(self,node):
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
        if node > self.cols:
            up = self.edge2index[(node-self.cols,node)]
        if node <= self.cols*(self.rows-1):
            down = self.edge2index[(node,node+self.cols)]
        if node % self.cols != 1:
            left = self.edge2index[(node-1,node)]
        if node % self.cols != 0:
            right = self.edge2index[(node,node+1)]

        return up,down,left,right


    def draw_all_paths(self):
        for path in self.paths:
            self.draw_grid(path.model)
            print ""

    def neighbor_nodes(self,node):
        """Find all the nodes that neighbor a node.

        Return:
            A list of all the node indexes that neighbor the given node.
        """

        neighbors = []
        if node > self.cols:
            neighbors.append(node-self.cols)
        if node <= self.cols*(self.rows-1):
            neighbors.append(node+self.cols)
        if node % self.cols != 1:
            neighbors.append(node-1)
        if node % self.cols != 0:
            neighbors.append(node+1)

        return neighbors

    def add_path(self,model):
        self.paths.append(Path(self,model))

    def end_point(self,node):
        """Find all the variables that should be set to represent a path ending at node.

        There are at most 4 sets of variable assignments to do this.
            Each one has 1 edge set to true and the other (1-3) edges set to false.

        Return a list
            Elements of the top-level list are len 2 lists
                First element of second level list is the edge to be set to true
                Second element of second level is list of edges to be set to false
        """
        assignments = []

        up,down,left,right = self.out_edges(node)

        for pos_edge in up,down,left,right:
            if pos_edge != -1:
                asn = [pos_edge,[]]
                for neg_edge in up,down,left,right:
                    if neg_edge != pos_edge and neg_edge != -1:
                        asn[1].append(neg_edge)
                assignments.append(asn)

        return assignments

    def mid_point(self,node):
        """Find all the variable assignments representing a path passing through node
        
        These are all valid pairs of 2 edges connected to the node.

        Return
            A list of lists of length 2.
            Each sublist is 2 edges taken.

        DO A SANITY CHECK THAT SETTING EXTRA EDGES TO 0 DOES NOTHIG TO PROBABILITY
        """
        assignments = []
        up,down,left,right = self.out_edges(node)

        valid_edges = []
        for i in up,down,left,right:
            if i != -1:
                valid_edges.append(i)

        for i in range(len(valid_edges)):
            for j in range(i+1,len(valid_edges)):
                assignments.append([valid_edges[i],valid_edges[j]])

        return assignments

    def start_set(self,start,end):
        start_asgnmts = self.end_point(start)
        end_asgnmts = self.end_point(end)

        for i in range(len(start_asgnmts)):
            for j in range(len(end_asgnmts)):
                bad_path = False
                p = Path(self)
                s_a = start_asgnmts[i]
                e_a = end_asgnmts[j]
                p.add_edge(s_a[0])
                p.add_edge(e_a[0])
                for neg_edge in s_a[1]:
                    if p.negate_edge(neg_edge) == -1:
                        bad_path = True
                        break
                for neg_edge in e_a[1]:
                    if p.negate_edge(neg_edge) == -1:
                        bad_path = True
                        break
                if bad_path == False:
                    self.paths.append(p)


    def prob_start_end_mid(self,start,end,mid):
        """Probability that a path starts at start and ends at end and passes through mid.
        
        This value is NOT normalized by the probability that a path starts at star and ends at end.

        Return that probability as a float.
        """

        start_asgnmts = self.end_point(start)
        end_asgnmts = self.end_point(end)
        mid_asgnmts = self.mid_point(mid)

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
                            
                    data = [-1 for i in range(self.num_edges)]
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
                    probability = self.copy.probability(evidence)
                    #print "prob: %f" % probability
                    total_prob += probability
                    
        return total_prob 

    def normalized_prob_mid(self,start,end,mid):
        """Probability that a path starts at start, ends at end, and passes through mid, normalized.
        Normalizing factor is probability that the path starts at start and ends at end.
        Return normalized probability.
        """
        mid_prob = self.prob_start_end_mid(start,end,mid)
        start_end_prob = self.prob_start_end(start,end)
        #print "start end prob %d: %f" % (mid,start_end_prob)
        #print "mid prob %d: %f" % (mid,mid_prob)
        return mid_prob/start_end_prob

    def prob_start_end(self,start,end):
        """Probability that a path starts at start and ends at end.

        Reutrn that probability as a float.
        """

        start_asgnmts = self.end_point(start)
        end_asgnmts = self.end_point(end)

        total_prob = 0.0
        for start_i in range(len(start_asgnmts)):
            for end_i in range(len(end_asgnmts)):
                start_a = start_asgnmts[start_i]
                end_a = end_asgnmts[end_i]
                data = [-1 for i in range(self.num_edges)]
                data[start_a[0]] = 1
                data[end_a[0]] = 1
                for zero in start_a[1]:
                    data[zero] = 0
                for zero in end_a[1]:
                    data[zero] = 0
                data = tuple(data)
                evidence = DataSet.evidence(data)
                probability = self.copy.probability(evidence)
                total_prob += probability

        return total_prob

    def visualize_mid_probs(self,start,end):
        probs = []
        for i in range(self.rows):
            for j in range(self.cols):
                mid = i*self.cols + j + 1
                if mid == start:
                    sys.stdout.write("start   ")
                    continue
                if mid == end:
                    sys.stdout.write(" end    ")
                    continue
                prob_mid = self.normalized_prob_mid(start,end,mid)
                sys.stdout.write("%.3f   " % prob_mid)
            sys.stdout.write("\n\n")
        
class Path(object):
    """An object to hold and manipulate variable instantiations for a psdd.
    """
    def __init__(self,manager,model=None):
        self.manager = manager
        if model == None:
            self.model = [-1 for i in range(self.manager.num_edges)]
        else:
            self.model = model[:]

    def model_tuple(self):
        return tuple(self.model)

    def ones_and_zeros(self):
        ones = []
        zeros = []
        for i in range(self.manager.num_edges):
            if self.model[i] == 0:
                zeros.append(i)
            elif self.model[i] == 1:
                ones.append(i)

        print "zeros: %s" % str(zeros)
        print "ones: %s" % str(ones)

    def add_and_neg_edges(self,to_add,to_neg):
        """add all the edges in to_add and negate all the edges in to_neg.
        If any operation is invalid, return -1.
        Otherwise return 1.
        """

        for i in to_add:
            if self.add_edge(i) == -1:
                return -1
        for j in to_neg:
            if self.negate_edge(j) == -1:
                return -1
        return 1

    def negate_edge(self,edge_num):
        """Change the model to reflect negating an edge.

        Set model[edge_num] to 0 if legal to do so (edge hasn't been set to 1 yet).

        Returns:
            If legal move: 1
            If illegal move: -1
        """
        if self.model[edge_num] == 1:
            return -1
        else:
            self.model[edge_num] = 0
            return 1


    def add_edge(self,edge_num):
        """Change the model to reflect adding an edge.

        Set model[edge_num] to 1 if legal to do so (edge hasn't been set to 0 yet).

        Returns:
            If legal move: 1
            If illegal move: -1
        """
        if self.model[edge_num] == 0:
            return -1
        else:
            self.model[edge_num] = 1
            return 1


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
                    #draw_grid(model,rows,cols,edge2index)
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


def generate_copy(rows,cols,start,end,fn_prefix,data_fn,bad_fn,edge2index,num_edges):
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
    return copy

def perform_analysis(rows,cols,start,end,fn_prefix,data_fn,bad_fn,edge2index,num_edges,trial_name):
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

    path_man = PathManager(rows,cols,edge2index,copy)

    print "%s PROBABILITIES" % trial_name
    path_man.visualize_mid_probs(start,end)

def find_kl(rows,cols,fn_prefix,bad_fn,data_fn):
    """Find the kl divergence for psdds trained on data taken from different time classes.

    Only run after determining the bad instances for the grid size.
    trip_id2class is 1-indexed while data lines and bad paths are 0-indexed
    """
    trip_id2class = pickle.load(open('../pickles/trip_id2class.pickle','rb'))
    with open(bad_fn,'r') as infile:
        bads = map(int,infile.readlines())
    bads.sort()
    bad_i = 0

    full_file = open(data_fn, "r")
    full_lines = full_file.readlines()
    full_file.close()

    full_ints = map(lambda x: map(int,x[:-1].split(',')),full_lines)
    full_tuples = map(tuple,full_ints)

    class_sets = []
    for i in range(6):
        class_sets.append([])

    for i in range(len(full_tuples)):
        if i == bads[bad_i]:
            bad_i += 1
            continue
        class_sets[trip_id2class[i+1]].append(full_tuples[i])

    class_sets = map(lambda x: DataSet.to_dict(x),class_sets)
    
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

    psdds = []
    pmanagers = []
    for class_num in range(6):
        pmanager = PSddManager(vtree)
        copy = pmanager.copy_and_normalize_sdd(sdd,vtree)
        pmanager.make_unique_true_sdds(copy,make_true=False) #AC: set or not set?
        psdd_parameters = copy.theta_count()

        training = class_sets[class_num]

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

        print "\nCLASS %d HAS %d UNIQUE AND %d TOTAL INSTANCES\n" % (class_num,len(training),training.N)

        psdds.append(copy)
        pmanagers.append(pmanager)

    for i in range(6):
        for j in range(i+1,6):
            kl_divergence = PSddNode.kl_psdds(psdds[i],pmanagers[i],psdds[j],pmanagers[j])
            print "kl (%d,%d): %d" % (i,j,kl_divergence)
    



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

    #man = PathManager(rows,cols,edge2index)
    #man.start_set(start,end)
    #man.draw_all_paths()
    #return

    fn_prefix_fixed = '../graphs/fixed_ends-%d-%d-%d-%d' % (rows,cols,start,end)
    data_fn_fixed = '../datasets/fixed_ends-%d-%d-%d-%d.txt' % (rows,cols,start,end)
    bad_fn_fixed = 'bad_paths/fixed_bad-%d-%d-%d-%d.txt' % (rows,cols,start,end)

    fn_prefix_general = '../graphs/general_ends-%d-%d' % (rows,cols)
    data_fn_general = '../datasets/general_ends-%d-%d.txt' % (rows,cols)
    bad_fn_general = 'bad_paths/general_bad-%d-%d.txt' % (rows,cols)
 
    #find_kl(rows,cols,fn_prefix_general,bad_fn_general,data_fn_general)


    copy = generate_copy(rows,cols,start,end,fn_prefix_general,data_fn_general,bad_fn_general,edge2index,num_edges)
    man = PathManager(rows,cols,edge2index,edge_index2tuple,copy)
    man.best_all_at_once(start,end)
    man.best_step_by_step(start,end)

    return
    man.visualize_mid_probs(start,end)

    


    return

    perform_analysis(rows,cols,start,end,fn_prefix_general,data_fn_general,bad_fn_general,edge2index,num_edges,"GENERAL ENDPOINT")
    perform_analysis(rows,cols,start,end,fn_prefix_fixed,data_fn_fixed,bad_fn_fixed,edge2index,num_edges,"FIXED ENDPOINT")







    return
    total_prob = 0.0
    s_neighbors = (start,start-1,start+cols,start+cols-1) 
    e_neighbors = (end,end-1,end+cols)#,end+cols-1) 
    for s_n in s_neighbors:
        for e_n in e_neighbors:
            total_prob += prob_start_end(rows,cols,s_n,e_n,num_edges,edge2index,copy)

    print total_prob
    return



if __name__ == '__main__':
    main()
