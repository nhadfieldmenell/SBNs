import numpy as np
import sys

#     0   1   2   3
#   +---+---+---+---+
# 0 |   |   |   |   |
#   +---+---+---+---+
# 1 |   |   |   |   |
#   +---+---+---+---+
# 2 |   |   |   |   |
#   +---+---+---+---+
# 3 |   |   |   |   |
#   +---+---+---+---+

#find all paths going from x to y in the grid
#create a grid for each path that has 1 for spot in the path and 0 for spot not in the path
#remGrid has dimensions numX x numY. Holds a 1 in (x,y) if Axy has been assigned, 0 if Axy not assigned
#return tuple (x,y,a) a is 0 if Axy is negative, 1 if Axy is positive 
#returnVal[a][b][3], a is number of possible paths, b is which point in the path, 3 tuple above
def paths(numX,numY,startX,startY,endX,endY,remGrid):
	returnVal = []
	if startX == endX and startY == endY:
		returnVal.append([startX,startY,1])
		for x in range(numX):
			for y in range(numY):
				if remGrid[x,y] == 0 and not (x == startX and y == startY):
					returnVal.append([x,y,0])
		theReturnVal = []
		theReturnVal.append(returnVal)	
		#print returnVal
		#print theReturnVal		
		return theReturnVal
	
	newRem = np.copy(remGrid)
	newRem[startX,startY] = 1
		
	if startX > 0 and remGrid[startX-1,startY] == 0:
		newX = startX-1
		
		#all ways to get from (newX,startY) to (endX,endY)
		possibleContinuations = paths(numX,numY,newX,startY,endX,endY,newRem)
		for futurePath in possibleContinuations:
			returnVal.append(futurePath)
		
	if startY > 0 and remGrid[startX,startY-1] == 0:
		newY = startY-1
		
		possibleContinuations = paths(numX,numY,startX,newY,endX,endY,newRem)
		for futurePath in possibleContinuations:
			returnVal.append(futurePath)
			
	if startX < numX-1 and remGrid[startX+1,startY] == 0:
		newX = startX+1
		
		#all ways to get from (newX,startY) to (endX,endY)
		possibleContinuations = paths(numX,numY,newX,startY,endX,endY,newRem)
		for futurePath in possibleContinuations:
			returnVal.append(futurePath)
	
	if startY < numY-1 and remGrid[startX,startY+1] == 0:
		newY = startY+1
		
		possibleContinuations = paths(numX,numY,startX,newY,endX,endY,newRem)
		for futurePath in possibleContinuations:
			returnVal.append(futurePath)
			
	for possPath in returnVal:
		possPath.append([startX,startY,1])
		
	return returnVal
		

	
numX = 4
numY = 4
remGrid = [[0 for x in range(numX)] for y in range(numY)]
remGrid = np.array(remGrid)
paths = paths(numX,numY,3,3,1,1,remGrid)

prevPaths = {}
uniquePaths = []

for path in paths:
	pathString = [0 for x in range(numX*numY)]
	for point in path:
		remGrid[point[0],point[1]] = point[2]
		pathString[numY*point[1] + point[0]] = str(point[2])
	pathString = str(pathString)
	"""
	print pathString
	for y in range(numY):
		for x in range(numX):
			sys.stdout.write(str(remGrid[x,y]))
		sys.stdout.write("\n")
	sys.stdout.write("\n")
	#print path
	#print pathString
	"""
	if not (pathString in prevPaths):
		prevPaths[pathString] = 1
		thisPath = np.copy(remGrid)
		uniquePaths.append(thisPath)
		for y in range(numY):
			for x in range(numX):
				sys.stdout.write(str(remGrid[x,y]))
			sys.stdout.write("\n")
		sys.stdout.write("\n")
		
	#"""
	
print len(paths)
print len(uniquePaths)