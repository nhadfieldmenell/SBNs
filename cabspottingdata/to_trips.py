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

def parse_id(line):
    line_num = 9
    while line[line_num] != '"':
        line_num += 1
    return line[9:line_num]

def replace_spaces(id):
    with open('new_%s.txt' % id,'r') as data:
        plaintext = data.read()

    with open('csv_%s.txt' % id, 'w') as data:
        data.write(plaintext.replace(' ',','))

def find_ids():
    id_file = open('_cabs.txt','r')
    id_lines = id_file.readlines()
    id_file.close()
    ids = []
    for line in id_lines:
        ids.append(parse_id(line))
    ids.sort()
    return ids

def convert_all_to_csv(ids):
    for driver_id in ids:
        replace_spaces(driver_id)

def create_trip_list(ids):
    with open('cab_trips.txt','w') as out_file:
        trip_num = 0
        for trip_id in ids:
            prev_occ = 0
            fn = open('csv_%s.txt' % trip_id,'r')
            lines = fn.readlines()
            fn.close()
            for line in lines:
                lat,lon,occ,time = normalize(line)
                if occ == 1: 
                    if prev_occ == 0:
                        trip_num += 1
                    out_file.write("%d,%f,%f,%d\n" % (trip_num,lat,lon,time))
                prev_occ = occ


def main():
    ids = find_ids()
    print ids
    #convert_all_to_csv(ids)
    create_trip_list(ids)
    return

    fn = open('new_udveoyx_csv.txt','r')
    out_file = open('converted_udveoyx.txt','w')
    lines = fn.readlines()
    fn.close()
    prev_occ = 0
    trip_num = 0
    for line in lines:
        lat,lon,occ,time = normalize(line)
        if occ == 1: 
            if prev_occ == 0:
                trip_num += 1
            out_file.write("%d,%f,%f,%d\n" % (trip_num,lat,lon,time))
        prev_occ = occ
    out_file.close()
    print trip_num


if __name__ == '__main__':
    main()
