#!/usr/bin/python
import sys
import random
import pickle


def main():
    t2bad = pickle.load(open('../pickles/trip_id2bad_better.pickle','rb'))
    #t2model = pickle.load(open('../pickles/trip_id2model_better.pickle','rb'))
    all_ids = []
    for i in range(1,464046):
        if i not in t2bad:
            all_ids.append(i)
    print "%d good instances" % len(all_ids)
    random.shuffle(all_ids)
    for i in range(10):
        print all_ids[i]
    tenth = len(all_ids)/10
    t_id2training = {}
    t_id2testing = {}
    for i in range(2*tenth):
        t_id2testing[all_ids[i]] = True
    for i in range(2*tenth,len(all_ids)):
        t_id2training[all_ids[i]] = True
    print "Testing instances"
    count = 0
    for t in t_id2testing:
        print t
        if count > 10:
            break
        count += 1
    print "Training instances"
    count = 0
    for t in t_id2training:
        print t
        if count > 10:
            break
        count += 1

    for t in t_id2testing:
        if t in t_id2training:
            print "%d in both!"
    for t in t_id2training:
        if t in t_id2testing:
            print "%d in both!"

    
    return


if __name__ == '__main__':
    main()
