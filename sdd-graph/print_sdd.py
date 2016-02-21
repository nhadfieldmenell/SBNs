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
    fnPrefix = ("asdf-%d-%d.zdd" % (m,n))
    gFn = ("asdf-%d-%d.graph.pickle" % (m,n))

    vtree = sdd.sdd_vtree_read('%s.vtree' % fnPrefix)
    manager = sdd.sdd_manager_new(vtree)
    sdd.sdd_vtree_free(vtree)
    alpha = sdd.sdd_read(('%s.sdd' % fnPrefix),manager)
    graph = pickle.load(open(gFn,'rb'))

    print graph[1]

    #g.print_grids(alpha,m,n,graph,manager)

