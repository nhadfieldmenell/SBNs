[noah@deduction sdd_graph]$ vim print_sdd.py 

#!/usr/bin/env python

import test_graph as g
import sys
import pickle
import sdd



if __name__ == '__main__':

    import sys

    if len(sys.argv) != 5:
        print "usage: %s [GRID-M] [GRID-N] [STARTPOINT] [ENDPOINT]" % sys.argv[0]
        exit(1)

    m,n,start,end = int(sys.argv[1]),int(sys.argv[2]),(int(sys.argv[3]),int(sys.argv[4])
    #fnPrefix = ("graphs/asdf-%d-%d" % (m,n))
    #gFn = ("graphs/asdf-%d-%d.graph.pickle" % (m,n))
    fnPrefix = ("graphs/start_end-%d-%d-%d-%d" % (m,n,start,end))
    gFn = ("graphs/start_end-%d-%d-%d-%d.graph.pickle" % (m,n,start,end))

    vtree = sdd.sdd_vtree_read('%s.vtree' % fnPrefix)
    manager = sdd.sdd_manager_new(vtree)
    sdd.sdd_vtree_free(vtree)
    alpha = sdd.sdd_read(('%s.sdd' % fnPrefix),manager)
    #graph = pickle.load(open(gFn,'rb'))
    graph = g.Graph.grid_graph(m,n)

    """
    size_file = open('graph_sizes.txt','a')
    size_file.write("%dx%d size: %d\n" % (m,n,sdd.sdd_size(alpha)))
    #size_file.write("%dx%d model count: %d\n" % (m,n,sdd.sdd_model_count(alpha,manager)))
    size_file.write("%dx%d global model count: %d\n" % (m,n,g.global_model_count(alpha,manager)))
    size_file.close()
    """
    """THIS IS IMPORTANT"""
    g.print_grids(alpha,m,n,graph,manager)


