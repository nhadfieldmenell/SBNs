#!/usr/bin/env python

import test_graph as g
import sys
import pickle
import sdd



if __name__ == '__main__':
	
    import sys

    if len(sys.argv) != 3:
        print "usage: %s [GRID-M] [GRID-N]" % sys.argv[0]
        exit(1)

    m,n = (int(sys.argv[1]),int(sys.argv[2]))
    fnPrefix = ("asdf-%d-%d" % (m,n))
    gFn = ("asdf-%d-%d.graph.pickle" % (m,n))

    vtree = sdd.sdd_vtree_read('%s.vtree' % fnPrefix)
    manager = sdd.sdd_manager_new(vtree)
    sdd.sdd_vtree_free(vtree)
    alpha = sdd.sdd_read(('%s.sdd' % fnPrefix),manager)
    #graph = pickle.load(open(gFn,'rb'))
    graph = g.Graph.grid_graph(m,n)

    size_file = open('graph_sizes.txt','a')
    size_file.write "%dx%d size: %d\n" % (m,n,sdd.sdd_size(alpha))
    size_file.write "%dx%d model count: %d\n" % (m,n,sdd.model_count(alpha,manager))
    size_file.write "%dx%d global model count: %d\n" % (m,n,g.global_model_count(alpha,manager))
    size_file.close()

    """THIS IS IMPORTANT"""
    #g.print_grids(alpha,m,n,graph,manager)

