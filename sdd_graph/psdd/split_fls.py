#!/usr/bin/env python
import random
import pickle


def main():
    fl2ts = pickle.load(open('../pickles/first_last2trip_ids-10-10.pickle','rb'))
    fls = []
    ordered_fl2in = {}
    for fl in fl2ts:
        ordered_fl2in[(min(fl[0],fl[1]),max(fl[0],fl[1]))] = True
    print len(fl2ts)
    print len(ordered_fl2in)
    for fl in ordered_fl2in:
        fls.append(fl)

    print len(fls)



    return

if __name__ == '__main__':
    main()
