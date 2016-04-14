#!/usr/bin/env python

from vtrees import vtrees
import sdd
import time
from test_grids_minato import print_grids
import minato2sdd as orig

def convert(filename):
    start = time.time()
    manager,alpha = orig.parse_bdd(filename+".zdd")
    end = time.time()
    print "      sdd node count: %d" % sdd.sdd_count(alpha)
    print "            sdd size: %d" % sdd.sdd_size(alpha)
    print "     sdd model count: %d" % sdd.sdd_model_count(alpha,manager)
    print "  global model count: %d" % orig.global_model_count(alpha,manager)
    print "       read bdd time: %.3fs" % (end-start)

    sdd.sdd_save(filename + ".sdd",alpha)
    #sdd.sdd_save_as_dot(filename +".sdd.dot",alpha)
    vtree = sdd.sdd_manager_vtree(manager)
    sdd.sdd_vtree_save(filename + ".vtree",vtree)
    #sdd.sdd_vtree_save_as_dot(filename +".vtree.dot",vtree)

    print "===================="
    print "before garbage collecting..." 
    print "live size:", sdd.sdd_manager_live_count(manager)
    print "dead size:", sdd.sdd_manager_dead_count(manager)
    print "garbage collecting..."
    sdd.sdd_manager_garbage_collect(manager)
    print "live size:", sdd.sdd_manager_live_count(manager)
    print "dead size:", sdd.sdd_manager_dead_count(manager)

def main():
    import sys

    if len(sys.argv) != 5:
        print "usage: %s [GRID-M] [GRID-N] [STARTPOINT] [ENDPOINT]" % sys.argv[0]
        exit(1)

    m,n,start,end = int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
    #filename = fnPrefix = ("graphs/start_end-%d-%d-%d-%d" % (m,n,start,end))
    filename = ("graphs/all_paths-%d-%d" % (m,n))

    convert(filename)


if __name__ == '__main__':
    main()
