import numpy as np

#assumes that each edge is listed only once
#returns a list of possible paths
#if a path has a list in it, means those are all possible next steps
def allGraphs(edges,thisEdge):

	retList = [thisEdge[1]]

	#create a copy of the graph without the edge that was just taken
	#newEdges = np.copy(edges)
	#newEdges = np.delete(newEdges,np.where(newEdges==thisEdge))

	newEdges = edges[:]
	newEdges.pop(newEdges.index(thisEdge))

	nextSteps = []

	for edge in newEdges:
		if edge[0] == thisEdge[1]:
			nextSteps.append(allGraphs(newEdges,edge))

	retList.append(nextSteps)

	return retList

def everything(edges):
	retList = []

	for edge in edges:
		thatOne = allGraphs(edges,edge)
		translated = translateOutput(thatOne)
		for item in translated:
			newItem = str(edge[0])
			newItem += item
			retList.append(newItem)

	return retList

def allGraphsDirections(edges,thisEdge):

	retList = [thisEdge[1]]



	#create a copy of the graph without the edge that was just taken
	#newEdges = np.copy(edges)
	#newEdges = np.delete(newEdges,np.where(newEdges==thisEdge))

	newEdges = edges[:]
	newEdges.pop(newEdges.index(thisEdge))



	oppEdge = [thisEdge[1],thisEdge[0]]

	print thisEdge
	print oppEdge

	print newEdges
	newEdges.pop(newEdges.index(oppEdge))

	print "here"

	nextSteps = []

	for edge in newEdges:
		if edge[0] == thisEdge[1]:
			nextSteps.append(allGraphsDirections(newEdges,edge))

	retList.append(nextSteps)

	return retList

def everythingDirections(edges):
	retList = []

	for edge in edges:
		thatOne = allGraphsDirections(edges,edge)
		translated = translateOutput(thatOne)
		for item in translated:
			newItem = str(edge[0])
			newItem += item
			retList.append(newItem)

	return retList

def translateOutput(output):
	if len(output) == 0:
		return [""]
	string = str(output[0])
	possibleEndings = []
	possibleEndings.append(string)
	for i in range(len(output[1])):
		for ending in translateOutput(output[1][i]):
			newString = ""
			newString+=string
			newString+=ending
			possibleEndings.append(newString)

	return possibleEndings

edges = [[0,1],[1,3],[1,4],[3,2],[5,0],[0,6]]

#edges = [[0,1],[5,0],[0,2]]

#output = allGraphs(edges,[5,0])

#print translateOutput(output)

#print everything(edges)
#print len(everything(edges))

doubEdges = edges[:]
for edge in edges:
	doubEdges.append([edge[1],edge[0]])

print doubEdges

bothDir = everythingDirections(doubEdges)

print bothDir
print len(bothDir)

pathDict = {}
for path in bothDir:
	if path in pathDict:
		print ("already here: ") + path
	else:
		pathDict[path] = 1

print "done"





