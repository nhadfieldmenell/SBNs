#!/usr/bin/python
import pickle


def main():
    num_files = 4
    fl2prediction = {}
    for i in range(num_files):
        fn = 'better_pickles/predictions_%d.pickle' % i
        fl2some = pickle.load(open(fn,'rb'))
        for fl in fl2some:
            fl2prediction[fl] = fl2some[fl]

    out_fn = 'better_pickles/fl2prediction.pickle'
    with open(out_fn,'wb') as output:
        pickle.dump(fl2prediction,output)


if __name__ == '__main__':
    main()

