#!/usr/bin/python
import sys

def normalize(line):
    counter = 0
    while line[counter] != ',':
        counter += 1
    first_comma = counter
    lat = float(line[:first_comma])
    counter += 1
    while line[counter] != ',':
        counter += 1
    second_comma = counter
    lon = float(line[first_comma+1:second_comma])
    counter += 2
    occupied = int(line[counter-1])
    time = int(line[counter+1:])
    return lat,lon,occupied,time

def main():
    fn = open('new_udveoyx_csv.txt','r')
    out_file = open('converted_udveoyx.txt','w')
    lines = fn.readlines()
    fn.close()
    prev_occ = 0
    trip_num = 0
    for line in lines:
        lat,lon,occ,time = normalize(line)
        if occ == 1 
            if prev_occ == 0:
                trip_num += 1
            out_file.write("%d,%f,%f,%d" % (trip_num,lat,lon,time))
    out_file.close()
    print trip_num


if __name__ == '__main__':
    main()
