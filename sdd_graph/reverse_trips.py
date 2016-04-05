#!/usr/bin/python

import buildGraphs as bg

def main():
    infile = open('cab_trips.txt','r')
    lines = infile.readlines()
    infile.close()
    outfile = open('cab_chronological.txt','w')
    i = 0
    cur_id = bg.normalize(lines[i])[0]
    while i < len(lines.len())
        this_trip = []
        prev_id = cur_id
        while i < len(lines) and cur_id == prev_id:
            this_trip.append(lines[i])
            i += 1
            if i >= len(lines):
                break
            cur_id = bg.normalize(lines[i])[0]
        for j in range(len(this_trip)-1,-1,-1):
            outfile.write(this_trip[j])
            


if __name__ == '__main__':
    main()
