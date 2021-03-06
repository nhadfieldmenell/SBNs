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
    GraphSet.set_universe(universe)
    paths = GraphSet()
    grid.print_edge_numbering(paths.universe(),dim[0],dim[1])
    grid.create_edge_to_index(paths,dim)

if __name__ == '__main__':
    main()
