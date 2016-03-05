import sys

def writeTrip(tripNum,orig,out):
    found = 0
    for line in orig:
        num = int(line[:5])
        if tripNum == num:
            found = 1
            out.write(line)
            #print line
        elif found == 1:
            break
	
orig = open('csvGPS.txt','r')
out = open('getTripsOut.txt','w')
numArgs = len(sys.argv)
for i in range(1,numArgs):
    writeTrip(int(sys.argv[i][:-1]),orig,out)

