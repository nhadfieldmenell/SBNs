#!/usr/bin/env python
import random
import pickle


def main():
    fl2ts = pickle.load(open('../pickles/first_last2trip_ids-10-10.pickle','rb'))
    all_fls = []
    added_fls = []
    orderedfl2in = {}
    for fl in tl2ts:
        orderedfl2in[(min(fl[0],fl[1]),max(fl[0],fl[1]))] = True
    print len(orderedfl2in)


    return

if __name__ == '__main__':
    main()
