#!/usr/bin/env python
import pickle

def main():
    t2model = pickle.load(open('better_pickles/trip_id2model.pickle','rb'))
    for t in t2model:
        t2model[t] = tuple(t2model[t])
    with open('better_pickles/trip_id2model.pickle','wb') as output:
        pickle.dump(t2model,output)


if __name__ == '__main__':
    main()

