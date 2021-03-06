#!/usr/bin/python
import random
import pickle


def main():
    t2bad = pickle.load(open('better_pickles/trip_id2bad.pickle','rb'))
    #t2model = pickle.load(open('../pickles/trip_id2model_better.pickle','rb'))
    all_ids = []
    trip_id2good = {}
    for i in range(1,464046):
        if i not in t2bad:
            all_ids.append(i)
            trip_id2good[i] = True
    with open('better_pickles/trip_id2good.pickle','wb') as output:
        pickle.dump(trip_id2good,output)
    return
    print "%d good instances" % len(all_ids)
    random.shuffle(all_ids)
    tenth = len(all_ids)/10
    t_id2training = {}
    t_id2testing = {}
    for i in range(2*tenth):
        t_id2testing[all_ids[i]] = True
    for i in range(2*tenth,len(all_ids)):
        t_id2training[all_ids[i]] = True

    with open('better_pickles/t2testing.pickle','wb') as output:
        pickle.dump(t_id2testing,output)
    with open('better_pickles/t2training.pickle','wb') as output:
        pickle.dump(t_id2training,output)

    
    return


if __name__ == '__main__':
    main()
