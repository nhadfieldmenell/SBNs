import math
import numpy
#bin numbers coorespond to south-western corner of region


#parse trip id, latitude, and longitude from a raw line of data
def normalize(line):
	tripNum = int(line[:5])
	latitude = float(line[32:41])
	Lonitude = float(line[42:-1])
	time = convertTimestamp(line[6:25])
	return tripNum,latitude,Lonitude,time


#pass in a timestamp in form 2007-01-06T08:42:58
#return a quadruple: day of week (0:sunday,1:monday,2:tuesday,...,6:saturday),hour,minute,second
def convertTimestamp(stamp):
	day = int(stamp[9])
	if day == 7:
		day = 0
	hour = int(stamp[11:13])
	minute = int(stamp[14:16])
	second = int(stamp[17:19])
	return day,hour,minute,second


#read in trips from file that contains trip start on one line, trip end on next line
#trips[][7]: 0-start lat, 1-start Lon, 2-end lat, 3-end Lon, 4-start[day,hour,minute,second], 5-end[day,hour,minute,second], 6-dist (in miles)
#There is no trip 0 or 15707--it skips from 15706 to 15708
def createTrips(orig):
	#1-indexed array
	#first entry (trips[0]) is all 0's
	#done to stay consistent with tripIds
	trips = [[0 for x in range(7)] for x in range(25001)]
	counter = 0
	index = 0
	for line in orig:
		parsed = normalize(line)
		if counter%2 == 0:
			index += 1
			
			#only happens to skip 15707
			if parsed[0] != index:
				index += 1
			trips[index][0] = parsed[1]
			trips[index][1] = parsed[2]
			trips[index][4] = parsed[3]
		else:
			trips[index][2] = parsed[1]
			trips[index][3] = parsed[2]
			trips[index][5] = parsed[3]
			trips[index][6] = gpsDist(trips[index][0],trips[index][1],trips[index][2],trips[index][3])
		counter += 1
	return trips


#create a list that contains full info about trip
#full[][3]: full[][0] is trip start [day,hour,minute,second], full[][1] is trip duration(minutes),
#full[][2] is a list of all coordinate positions in the order they appear (full[][2][0] is latitude, full[][2][1] is longitude)
def createFull(fn,trips,latStep,lonStep,minLat,minLon):
	fullTrips = [[[] for x in range(3)] for x in range(25001)]
	
	#find the day, hour, minute, duration
	for i in range(len(trips)):
		if trips[i][0] == 0:
			continue
		fullTrips[i][0] = trips[i][4]
		tripStartMin = trips[i][4][1]*60+trips[i][4][2]
		tripEndMin = trips[i][5][1]*60+trips[i][5][2]
		#if trip spans 2 days
		if trips[i][4][0] != trips[i][5][0]:
			tripEndMin += 60*24
		fullTrips[i][1] = tripEndMin-tripStartMin
		
	previousTripNum = 0
	previousGridSpot = [0,0]
	#create a path on the grid
	for line in fn:
		normalized = normalize(line)
		tripId = normalized[0]
		latitude = normalized[1]
		longitude = normalized[2]
		gridSpot = convertGPS(latitude,longitude,latStep,lonStep,minLat,minLon)
		
		#new trip
		if tripId != previousTripNum:
			previousTripNum = tripId
			previousGridSpot[0] = gridSpot[0]
			previousGridSpot[1] = gridSpot[1]
			theSpot = []
			theSpot.append(gridSpot[0])
			theSpot.append(gridSpot[1])
			fullTrips[tripId][2].append(theSpot)
		else:
			if gridSpot[0] != previousGridSpot[0] or gridSpot[1] != previousGridSpot[1]:
				previousGridSpot[0] = gridSpot[0]
				previousGridSpot[1] = gridSpot[1]
				theSpot = []
				theSpot.append(gridSpot[0])
				theSpot.append(gridSpot[1])
				fullTrips[tripId][2].append(theSpot)
	
	return fullTrips


#calc the min and max latitute and Lonitude traversed in system
def minMax(trips):
	maxLat = float(0)
	minLat = float(500)
	maxLon = float(-500)
	minLon = float(0)
	
	for trip in trips:
		if trip[0] == 0 or trip[1] == 0 or trip[2] == 0 or trip[3] == 0:
			continue
			
		if trip[0] < minLat:
			minLat = trip[0]
		elif trip[0] > maxLat:
			maxLat = trip[0]
		if trip[2] < minLat:
			minLat = trip[2]
		elif trip[2] > maxLat:
			maxLat = trip[2]
			
		if trip[1] < minLon:
			minLon = trip[1]
		elif trip[1] > maxLon:
			maxLon = trip[1]
		if trip[3] < minLon:
			minLon = trip[3]
		elif trip[3] > maxLon:
			maxLon = trip[3]
	return maxLat, minLat, maxLon, minLon

	
#create arrays of buckets, where each bucket holds a lat/Lon 
#and all trips within latStep/lonStep of that point
def createStartEnd(latStep,lonStep,theTrips):
	#each of these arrays hold an array whose first value is the end
	#of a latitude or Lonitude line (with length lat/LonStep)
	#the subsuquent values are tripIds that fall on that line
	startLat = []
	startLon = []
	endLat = []
	endLon = []
	lat = minLat
	while lat <= maxLat:
		current = []
		current2 = []
		current.append(lat)
		startLat.append(current)
		endLat.append(current2)
		lat += latStep
		
	lon = minLon
	while lon <= maxLon:
		current = []
		current2 = []
		current.append(lon)
		startLon.append(current)
		endLon.append(current2)
		lon += lonStep
		
	index = 0
	addedTo = 0
	for trip in theTrips:
		if index == 0:
			index += 1
			continue
			
		if trip[0] == 0 or trip[1] == 0 or trip[2] == 0 or trip[3] == 0:
			index += 1
			continue
			
		adjStartLat = trip[0] - minLat
		adjIndex = int(adjStartLat/latStep)
		if startLat[adjIndex].count(index) == 0:
			startLat[adjIndex].append(index)
			addedTo = adjIndex
			
		adjEndLat = trip[2] - minLat
		adjIndex = int(adjEndLat/latStep)
		if endLat[adjIndex].count(index) == 0:
			endLat[adjIndex].append(index)
			
		adjStartLon = trip[1] - minLon
		adjIndex = int(adjStartLon/lonStep)
		if startLon[adjIndex].count(index) == 0:
			startLon[adjIndex].append(index)
			
		adjEndLon = trip[3] - minLon
		adjIndex = int(adjEndLon/lonStep)
		if endLon[adjIndex].count(index) == 0:
			endLon[adjIndex].append(index)
	
		index += 1
		
	return startLat, startLon, endLat, endLon


#finds points that have the most overlapping trips (either starts or ends)
#pass in arrays of start lat/lon or end lat/lon and the number of good points to find
def findGoodPoints(latArray,LonArray,numPoints):
	bestStartLat = 0
	bestStartLon = 0
	numMatches = 0
	latIndex = 0
	fewestMatches = 0
	fewestIndex = 0
	#array of 2-tuples
	#first tuple is a 3-tuple: matches, startlat, startLon
	#second tuple is all of the tripId's that start in that position
	goodPoints = []
	for i in range(numPoints):
		point = []
		triple = [0,0,0]
		point.append(triple)
		blank = []
		point.append(blank)
		goodPoints.append(point)	
		
	#finds the lat-Lon square of radius latLonStep that has the most uber pickups
	for lats in latArray:
		lonIndex = 0
		for lons in LonArray:
			matches = 0
			matchArray = []
			for latI in lats:
				if lons.count(latI) == 1:
					matches += 1
					matchArray.append(latI)
					
			if matches > fewestMatches:
				point = []
				triple = [matches,latIndex,lonIndex]
				point.append(triple)
				point.append(matchArray)
				goodPoints[fewestIndex] = point
				
				fewestMatches = goodPoints[0][0][0]
				fewestIndex = 0
				
				for i in range(numPoints):
					if goodPoints[i][0][0] < fewestMatches:
						fewestMatches = goodPoints[i][0][0]
						fewestIndex = i
						
			lonIndex += 1
			
		latIndex += 1
	return goodPoints


#always rounds down
#converts gps to bin number in lat/lon arrays
def convertGPS(lat,lon,latStep,lonStep,minLat,minLon):
	latInt = int((lat - minLat)/latStep)
	lonInt = int((lon - minLon)/lonStep)
	return latInt,lonInt


#pass in bin numbers for lat/lon and return the corresponding GPS
def binToGPS(latBin,lonBin,latStep,lonStep,minLat,minLon):
	latGPS = minLat+latBin*latStep
	lonGPS = minLon+lonBin*lonStep
	return latGPS,lonGPS

#pass in bucket numbers for GPS points rather than actual points
def findTrips(startLatNum,startLonNum,endLatNum,endLonNum,startLats,startLons,endLats,endLons,trips):
	validTrips = []
	for tripId in startLats[startLatNum]:
		if float(tripId).is_integer():
			if endLons[endLonNum].count(tripId) == 1:
				if endLats[endLatNum].count(tripId) == 1:
					if startLons[startLonNum].count(tripId) == 1:
						validTrips.append(tripId)
	return validTrips
	
	


#pass in a list of id's that all have the same start region
#find the end region(s) that hold the most end points from those trips
#startLatPos is index in the startLat array of the starting latitude
#minDist will allow you to input a minimum distance (in miles) for start-end point pairs
def findBestEnd(trips,startIds,startLatPos,startLonPos,latStep,lonStep,minLat,minLon,numLats,numLons,minDist,gridSize):
	diffTrips = []
	for trip in trips:
		tripInfo = []
		startInfo = convertGPS(trip[0],trip[1],latStep,lonStep,minLat,minLon)
		endInfo = convertGPS(trip[2],trip[3],latStep,lonStep,minLat,minLon)
		tripInfo.append(startInfo[0])
		tripInfo.append(startInfo[1])
		tripInfo.append(endInfo[0])
		tripInfo.append(endInfo[1])
		diffTrips.append(tripInfo)
		
	#dests is a 2D array representing the grid of the city each grid space holds the 
	#indices of the trips that started at the start point and end at that grid space
	dests = [[0 for x in range(numLons)] for x in range(numLats)]
	destArray = [[[] for x in range(numLons)] for x in range(numLats)]
	for tripNum in startIds:
		theTrip = diffTrips[tripNum]
		dests[theTrip[2]][theTrip[3]] += 1
		destArray[theTrip[2]][theTrip[3]].append(tripNum)
		
	#print dests
	best = 0
	index = [0,0]
	secondBest = 0
	secondIndex = [0,0]
	minDistBest = 0
	minDistIndex = [0,0]
	
	#alrightIndicies holds all end points that have at 
	#least alrightMatchNum trips originating at the start point
	alrightIndicies = []
	alrightMatchNum = 10;
	for i in range(numLats):
		for j in range(numLons):
			if dests[i][j] > alrightMatchNum:
				alrightIndicies.append(index)
			if dests[i][j] > secondBest:
				if dests[i][j] > best:
					secondBest = best
					secondIndex[0] = index[0]
					secondIndex[1] = index[1]
					best = dests[i][j]
					index = [i,j]
				else:
					secondBest = dests[i][j]
					secondIndex = [i,j]
			if dests[i][j] > minDistBest:
				if math.sqrt(math.pow(i-startLatPos,2)+math.pow(j-startLonPos,2)) >= minDist:
					minDistBest = dests[i][j]
					minDistIndex[0] = i
					minDistIndex[1] = j
					
		
	print "best"
	print best
	print index
	print destArray[index[0]][index[1]]
	print "\nsecond best"	
	print secondBest
	print secondIndex
	print destArray[secondIndex[0]][secondIndex[1]]
	print "\nminDist best"
	print minDistBest
	print minDistIndex
	print destArray[minDistIndex[0]][minDistIndex[1]]
	return alrightIndicies
		
	


#finds the number of ride starts per time of day
#select granularity with hourInc (how many hours each block is)
#returns list of 3 tuples: start time, end time, # of rides
#24 should be divisible by hourInc
def ridesByHour(trips,hourInc):
	numPeriods = 24/hourInc
	tripsPerTime = [[0 for x in range(3)] for x in range(numPeriods)]
	for trip in trips:
		# because 1-indexed
		if trip[0] == 0:
			continue
		tripsPerTime[trip[4][1]/hourInc][2]+=1
	for i in range(numPeriods):
		startHour = i*hourInc
		endHour = (i+1)*hourInc
		tripsPerTime[i][0] = startHour
		tripsPerTime[i][1] = endHour
	
	best = 0
	bestPeriod = [0,0]
	tot = 0			
	for i in range(numPeriods):
		tot += tripsPerTime[i][2]
		if tripsPerTime[i][2] > best:
			best = tripsPerTime[i][2]
			bestPeriod = [tripsPerTime[i][0],tripsPerTime[i][1]]
	
	print "tot"
	print tot
	print "best period"		
	print best
	print bestPeriod
	for i in range(numPeriods):
		print str(i) + ": " + str(tripsPerTime[i][2])
	return tripsPerTime


#finds the # of rides per time period per day
#returns list of list of 3 tuples: tripsPerPeriod[i][j][k]:
#	i selects day, j selects hour range, k selects from 3 tuple: start time, end time, # of rides
def ridesByHourAndDay(trips,hourInc):
	numPeriods = 24/hourInc
	tripsPerPeriod = [[[0 for x in range(3)] for x in range(numPeriods)] for x in range(7)]
	for trip in trips:
		# because 1-indexed
		if trip[0] == 0:
			continue
		tripsPerPeriod[trip[4][0]][trip[4][1]/hourInc][2]+=1
	for day in range(7):	
		for i in range(numPeriods):
			startHour = i*hourInc
			endHour = (i+1)*hourInc
			tripsPerPeriod[day][i][0] = startHour
			tripsPerPeriod[day][i][1] = endHour
	
	tot = 0		
	for day in range(7):
		for i in range(numPeriods):
			#print str(day) + "," + str(i) + ": " + str(tripsPerPeriod[day][i][2])
			tot += tripsPerPeriod[day][i][2]
		
	numTops = 10
	#4-tuples: day, period start hour, period end hour, # of trips
	#holds these tuples for the numTops-th best periods
	topPeriods = [[0 for x in range(4)] for x in range(numTops)]
	minTrips = 0
	minIndex = 0
	for day in range(7):
		for period in range(numPeriods):
			if tripsPerPeriod[day][period][2] > minTrips:
				topPeriods[minIndex] = [day,tripsPerPeriod[day][period][0],tripsPerPeriod[day][period][1],tripsPerPeriod[day][period][2]]
				
				#find new lowest
				minTrips = tripsPerPeriod[day][period][2]
				for topIndex in range(numTops):
					if topPeriods[topIndex][3] <= minTrips:
						minTrips = topPeriods[topIndex][3]
						minIndex = topIndex
	print "tot: " + str(tot)
	print "top periods"
	for i in topPeriods:
		print i


#pass in coordinates for 2 points
#return distance (in miles) between the two
def gpsDist(lat1,lon1,lat2,lon2):
	latDif = lat2-lat1
	lonDif = lon2-lon1
	latDist = latDif/0.0145
	lonDist = lonDif/0.01825
	return math.sqrt(math.pow(latDist,2)+math.pow(lonDist,2))	


#pass in grid spots for 2 points
#return distance (in miles) between the two
def gridDist(lat1,lon1,lat2,lon2,latStep,lonStep):
	latDif = (lat2-lat1)*latStep
	lonDif = (lon2-lon1)*lonStep
	latDist = latDif/0.0145
	lonDist = lonDif/0.01825
	return math.sqrt(math.pow(latDist,2)+math.pow(lonDist,2))

#pass in trips
#return an array holding the length of each trip
#array indices do not match up to trip numbers because of no trip 0 or trip 15707
def tripLengths(trips):
	tripLengths = []
	for i in range(len(trips)):
		if trips[i] == 0:
			continue
		tripLengths.append(trips[i][6])
		
	print "average distance (miles): " + str(numpy.mean(tripLengths))
	print "25th percentile: " + str(numpy.percentile(tripLengths,25))	
	print "50th percentile: " + str(numpy.percentile(tripLengths,50))
	print "75th percentile: " + str(numpy.percentile(tripLengths,75))
	print "90th percentile: " + str(numpy.percentile(tripLengths,90))
	return tripLengths


#pass in start and end grid points
#return string of the form "startLatGrid,startLonGrid,endLatGrid,endLonGrid"
def ptsToString(lat1,lon1,lat2,lon2):
	return str(lat1)+","+str(lon1)+","+str(lat2)+","+str(lon2)


#pass in full trips, width of latitude in grid, width of longitude in grid
def pointAtoB(fullTrips,latGridSpots,lonGridSpots,minDist,latStep,lonStep,pointsOut,gridSize):
	pointsOut.write("grid size " + str(gridSize) +"\n")
	pointsOut.write("minDist " + str(minDist) + "\n")
	numTrips = len(fullTrips)
	
	#aToB is a dict whose keys are strings of the form "startLatGrid,startLonGrid,endLatGrid,endLonGrid"
	#values are # of trips that traverse that path
	aToB = {}
	
	#tripTraversals[i] holds a list strings "startLat,startLon,endLat,endLon" for each trip
	#it only holds the traversals that are longer than minDist
	tripTraversals = [[] for x in range(numTrips)]
	
	for tripIndex in range(numTrips):
		pointsOut.write("trip " + str(tripIndex) + "\n")
		print tripIndex
		if fullTrips[tripIndex][0] == []:
			continue
		coordList = fullTrips[tripIndex][2]
		coordListLen = len(coordList)
		for startIndex in range(coordListLen):
			#iterate backwards through list to save time
			#once you see one that is closer than minDist, stop looking at the ones that came earlier in the trip
			#assumes that trip gets continually farther from start point
			for endIndex in range(coordListLen-1,startIndex,-1):
				endPoint = coordList[endIndex]
				startPoint = coordList[startIndex]
				tripString = ptsToString(startPoint[0],startPoint[1],endPoint[0],endPoint[1])
				if gridDist(startPoint[0],startPoint[1],endPoint[0],endPoint[1],latStep,lonStep) < minDist:
					continue
				elif tripTraversals[tripIndex].count(tripString) == 0:
					tripTraversals[tripIndex].append(tripString)
					pointsOut.write(tripString + "\n")
					if tripString in aToB:
						aToB[tripString] += 1
					else:
						aToB[tripString] = 1
					
	best = 0
	bestKey = ""
	for key in aToB.keys():
		if aToB[key] > best:
			best = aToB[key]
			bestKey = key
	
	tripsWithPoints = []
	for i in range(numTrips):
		if tripTraversals[i].count(bestKey) == 1:
			tripsWithPoints.append(i)
			
	print tripsWithPoints


#return a list of all the ID's of trips that traverse tripString
def indicesFromTraversals(tripTraversals,tripString):
	traversing = []
	for i in range(len(tripTraversals)):
		if tripTraversals[i].count(tripString) == 1:
			traversing.append(i)
	return traversing


#populate dict aToB with trip strings stored in aToB.txt
def readPtsFromFile(pointsIn):
	numTrips = 25001
	#aToB is a dict whose keys are strings of the form "startLatGrid,startLonGrid,endLatGrid,endLonGrid"
	#values are # of trips that traverse that path
	aToB = {}
	
	#tripTraversals[i] holds a list strings "startLat,startLon,endLat,endLon" for each trip
	#it only holds the traversals that are longer than minDist
	tripTraversals = [[] for x in range(numTrips)]
	tripId = 0
	for line in pointsIn:
		if line[0] == "g" or line[0] == "m":
			continue
		if line[0] == "t":
			tripId = int(line[5:-1])
			continue
		theLine = line[:-1]
		tripTraversals[tripId].append(theLine)
		if theLine in aToB:
			aToB[theLine] += 1
		else:
			aToB[theLine] = 1
	
	
	numBests = 10
	#bestArray[][0]: count, bestArray[][1]: key
	bestArray = [[0 for x in range(2)] for x in range(numBests)]
	minBest = 0
	minIndex = 0
	for key in aToB.keys():
		score = aToB[key]
		if score > minBest:
			bestArray[minIndex][0] = score
			bestArray[minIndex][1] = key
			minBest = score
			for i in range(len(bestArray)):
				if bestArray[i][0] <= minBest:
					minBest = bestArray[i][0]
					minIndex = i
					
	print bestArray
	print indicesFromTraversals(tripTraversals,bestArray[0][1])
	print indicesFromTraversals(tripTraversals,bestArray[3][1])
	print indicesFromTraversals(tripTraversals,bestArray[7][1])
			
	"""
	best = 0
	bestKey = ""
	for key in aToB.keys():
		if aToB[key] > best:
			best = aToB[key]
			bestKey = key
	
	tripsWithPoints = []
	for i in range(numTrips):
		if tripTraversals[i].count(bestKey) == 1:
			tripsWithPoints.append(i)
			
	print len(tripsWithPoints)
	"""


#store in tripsInPeriod the id's of all trips within a numHrs period (hours) starting on day at startHr
def getTripsByPeriod(trips,day,startHr,numHrs):
	tripsInPeriod = []
	for tripNum in range(len(trips)):
		if trips[tripNum][0] == 0:
			continue
		
		#the or takes care of the case where a period spans midnight 
		if (trips[tripNum][4][0] == day and trips[tripNum][4][1] >= startHr and trips[tripNum][4][1] < startHr + numHrs) or (trips[tripNum][4][0] == day+1 and trips[tripNum][4][1] + 24 < startHr + numHrs):
			tripsInPeriod.append(tripNum)
			
	return tripsInPeriod


#pass in trips, day, start hour, period length (hours), latStep, lonStep, number of good points to find
#yields the most popular start points and end points for that period
def bestStAndEnPrd(trips,day,startHr,numHrs,latStep,lonStep,numPoints,minLat,minLon):
	tripsInPeriod = getTripsByPeriod(trips,day,startHr,numHrs)
	theseTrips = []
	#indexes[i] holds the tripID of the ith entry in theseTrips
	indexes = []
	for tripNum in tripsInPeriod:
		indexes.append(tripNum)
		theseTrips.append(trips[tripNum])
		"""
		startLats.append(trips[tripNum][0])
		startLons.append(trips[tripNum][1])
		endLats.append(trips[tripNum][2])
		endLons.append(trips[tripNum][3])
		"""
	startEnd = createStartEnd(latStep,lonStep,theseTrips)	
	startLats = startEnd[0]
	startLons = startEnd[1]
	endLats = startEnd[2]
	endLons = startEnd[3]
	
	goodStarts = findGoodPoints(startLats,startLons,numPoints)
	goodEnds = findGoodPoints(endLats,endLons,numPoints)
	print "starts"
	for start in goodStarts:
		gps = binToGPS(start[0][1],start[0][2],latStep,lonStep,minLat,minLon)
		print str(start[0]) + "," + str(gps[0]) + "," + str(gps[1])
	print "ends"
	for end in goodEnds:
		gps = binToGPS(end[0][1],end[0][2],latStep,lonStep,minLat,minLon)
		print str(end[0]) + "," + str(gps[0]) + "," + str(gps[1])
	#print goodStarts
	#print goodEnds
	


#yields all trips longer than minDist that occured during the specified hour
def getDistTripsByPeriod(trips,day,startHr,numHrs,minDist):
	tripsInPeriod = getTripsByPeriod(trips,day,startHr,numHrs)
	longTrips = []
	for tripId in tripsInPeriod:
		thisTrip = trips[tripId]
		if gpsDist(thisTrip[0],thisTrip[1],thisTrip[2],thisTrip[3]) >= minDist:
			longTrips.append(tripId)
			
	print longTrips


orig = open('firstLast.txt','r')
fullFn = open('csvGps.txt','r')
pointsOut = open('aToB.txt','w')
#pointsIn = open('aToB.txt','r')

#enter the square edge length to specify grid regions (in miles)
#1 mile is approximately 0.0145 in latitude in SF area
#1 mile is approximately 0.01825 in longitude in SF area
gridSize = 0.4
latStep = 0.0145*gridSize
lonStep = 0.01825*gridSize

#trips[][7]: start lat, start Lon, end lat, end Lon, start[day,hour,minute,second], end[day,hour,minute,second], dist (in miles)
trips = createTrips(orig)


minMaxRet = minMax(trips)
maxLat = minMaxRet[0]
minLat = minMaxRet[1]
maxLon = minMaxRet[2]
minLon = minMaxRet[3]
latGridSpots = int((maxLat-minLat)/latStep) + 1
lonGridSpots = int((maxLon-minLon)/lonStep) + 1

minDist = 2

startEndRet = createStartEnd(latStep,lonStep,trips)
startLat = startEndRet[0]
startLon = startEndRet[1]
endLat = startEndRet[2]
endLon = startEndRet[3]

numPoints = 1
goodStarts = findGoodPoints(startLat,startLon,numPoints)

bestEnds = findBestEnd(trips,goodStarts[0][1],goodStarts[0][0][1],goodStarts[0][0][2],latStep,lonStep,minLat,minLon,len(startLat),len(startLon),2,gridSize)


#getDistTripsByPeriod(trips,0,0,2,1.5)

#bestStAndEnPrd(trips,0,4,2,latStep,lonStep,5,minLat,minLon)
#getTripsByPeriod(trips,0,0,2)
#readPtsFromFile(pointsIn)
#fullTrips = createFull(fullFn,trips,latStep,lonStep,minLat,minLon)
#pointAtoB(fullTrips,latGridSpots,lonGridSpots,minDist,latStep,lonStep,pointsOut,gridSize)


"""
tripLengths(trips)

ridesByHour(trips,1)
ridesByHourAndDay(trips,2)



print minLat
print maxLat
print minLon
print maxLon


startEndRet = createStartEnd(latStep,lonStep)
startLat = startEndRet[0]
startLon = startEndRet[1]
endLat = startEndRet[2]
endLon = startEndRet[3]

specStartLat = float(37.788929)
specStartLon = float(-122.399598)
specEndLat = float(37.754739)
specEndLon = float(  -122.415317)
startPoint = convertGPS(specStartLat,specStartLon,latStep,lonStep,minLat,minLon)
endPoint = convertGPS(specEndLat,specEndLon,latStep,lonStep,minLat,minLon)

#theTrips = findTrips(startPoint[0],startPoint[1],endPoint[0],endPoint[1],startLat,startLon,endLat,endLon,trips)
#print theTrips
"""
"""
convGPS = convertGPS(37.774, -122.4152,latLonStep,latLonStep,minLat,minLon)
print convGPS[0]
print convGPS[1]

print startLat[80][0]
print startLon[41][0]
"""
""""
numPoints = 1
goodStarts = findGoodPoints(startLat,startLon,numPoints)
#goodEnds = findGoodPoints(endLat,endLon,numPoints)

bestEnds = findBestEnd(trips,goodStarts[0][1],goodStarts[0][0][1],goodStarts[0][0][2],latStep,lonStep,minLat,minLon,len(startLat),len(startLon),4,gridSize)



print "starts"
for point in goodStarts:
	print point[0]
	print startLat[point[0][1]][0]
	print startLon[point[0][2]][0]
	
"""
"""	
print "ends"
for point in goodEnds:
	print point[0]
	

#for track in goodStarts[6][1]:
#	print str(track)+','+str(trips[track][0])
		

	
#for i in range(20):
#	print startLat[i]
	
#for i in range(20):
#	print endLat[i]
#	
#for i in range(20):
#	print startLon[i]
	
#for i in range(20):
#	print startLon[i]

minMaxRet = minMax(trips)
maxLat = minMaxRet[0]
minLat = minMaxRet[1]
maxLon = minMaxRet[2]
minLon = minMaxRet[3]
latGridSpots = int((maxLat-minLat)/latStep) + 1
lonGridSpots = int((maxLon-minLon)/lonStep) + 1

minDist = 2

startEndRet = createStartEnd(latStep,lonStep,trips)
startLat = startEndRet[0]
startLon = startEndRet[1]
endLat = startEndRet[2]
endLon = startEndRet[3]

numPoints = 1
goodStarts = findGoodPoints(startLat,startLon,numPoints)

bestEnds = findBestEnd(trips,goodStarts[0][1],goodStarts[0][0][1],goodStarts[0][0][2],latStep,lonStep,minLat,minLon,len(startLat),len(startLon),1.5,gridSize)
"""
	
#print str(minLat) + ',' + str(maxLat) + ',' + str(minLon) + ',' + str(maxLon)