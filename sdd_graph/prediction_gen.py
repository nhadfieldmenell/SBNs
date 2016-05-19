#!/usr/bin/env python
import pickle
import time
import sys
import fixed_end_analyze as an

def main():
    run = sys.argv[1]
    rows=cols=10

    edge_filename = '../graphs/edge-nums-%d-%d.pickle' % (rows,cols)
    edge2index = pickle.load(open(edge_filename,'rb'))
    edge_tuple_filename = '../graphs/edge-to-tuple-%d-%d.pickle' % (rows,cols)
    edge_index2tuple = pickle.load(open(edge_tuple_filename,'rb'))
    num_edges = (rows-1)*cols + (cols-1)*rows
    fl_fn = 'better_pickles/fl_split%d.pickle' % run
    fl2in = pickle.load(open(fl_fn,'rb'))

    fn_prefix = '../graphs/general_ends-%d-%d' % (rows,cols)
    copy = an.generate_copy_new(fn_prefix)

    out_fn = 'better_pickles/predictions_%d.pickle' % run
    man = an.PathManager(rows,cols,edge2index,edge_index2tuple,copy)

    fl2prediction = {}
    for fl in fl2in:
        fl2prediction[fl] = man.best_all_at_once(fl[0],fl[1])


    with open(out_fn,'wb') as output:
        pickle.dump(all_at_once,output)




   


if __name__ == '__main__':
    main()
