#!/usr/bin/env python

import sys
import pickle
import test_grids_minato as grid

def main():
    if len(sys.argv) != 3:
        print "usage: %s [GRID-M] [GRID-N]" % sys.argv[0]
        exit(1)
    dim = (int(sys.argv[1]),int(sys.argv[2]))
    dimension = (dim[0]-1,dim[1]-1)

    from graphillion import GraphSet
    import graphillion.tutorial as tl
    universe = tl.grid(*dimension)
    print universe
    grid.print_edge_numbering(universe,dim[0],dim[1])

if __name__ == '__main__':
    main()
