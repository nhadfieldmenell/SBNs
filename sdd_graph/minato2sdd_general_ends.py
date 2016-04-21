#!/usr/bin/env python

import sys
import minato2sdd_fixed_ends as fixed

def main():
    if len(sys.argv) != 3:
        print "usage: %s [GRID-M] [GRID-N]" % sys.argv[0]
        exit(1)

    m,n = int(sys.argv[1]),int(sys.argv[2])
    filename = fnPrefix = ("graphs/general_ends-%d-%d" % (m,n))

    fixed.convert(filename)


if __name__ == '__main__':
    main()

