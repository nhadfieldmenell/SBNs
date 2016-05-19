#!/usr/bin/env python
import random
import pickle


def main():
    fl2ts = pickle.load(open('../pickles/first_last2trip_ids-10-10.pickle','rb'))
    fls = []
    ordered_fl2in = {}
    for fl in fl2ts:
        ordered_fl2in[(min(fl[0],fl[1]),max(fl[0],fl[1]))] = True
    for fl in ordered_fl2in:
        fls.append(fl)

    num_splits = 6
    block_size = len(fls)/num_splits
    count = 0
    for i in range(num_splits):
        fn = 'better_pickles/fl_split%d.pickle' % i
        fl2epoch = {}
        if i == num_splits-1:
            for j in range(i*block_size,len(fls)):
                fl2epoch[fls[j]] = True
        else:
            for j in range(i*block_size,(i+1)*block_size):
                fl2epoch[fls[j]] = True

        count += len(fl2epoch)
        with open(fn,'wb') as output:
            pickle.dump(fl2epoch,output)
    print count





    return

if __name__ == '__main__':
    main()
