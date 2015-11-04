orig = open('csvGPS.txt','r')
firstLast = open('firstLast.txt','w')
tripNum = 0
prevLine = ""
for line in orig:
	newTripNum = int(line[:5])
	if newTripNum != tripNum:
		firstLast.write(prevLine)
		firstLast.write(line)
		tripNum = newTripNum
	prevLine = line