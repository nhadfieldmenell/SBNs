#!/usr/bin/python
import sys
import random
import pickle


def main():
    t2bad = pickle.load(open('../pickles/trip_id2bad_better.pickle','rb'))
    t2model = pickle.load(open('../pickles/trip_id2model_better.pickle','rb'))
    for i in range(1,464046):
        if i not in t2model:
            print i
    print "all values have models"
    return


if __name__ == '__main__':
    main()
