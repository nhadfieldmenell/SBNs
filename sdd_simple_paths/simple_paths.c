#include <stdio.h>
#include <stdlib.h>
#include "sddapi.h"

#define gridX 6
#define gridY 6

// build with: sudo gcc -O2 -std=c99 simple_paths.c -Iinclude -Llib -lsdd -lm -o simple_paths

// The SDD variable that corresponds to grid spot (x,y) is y*gridX + x + 1

/*
int** copyGrid(int remGrid[gridX][gridY]){
	int i,j;
	int** retGrid[gridX][gridY];
	for (i = 0; i < gridX; i++){
		for(j = 0; j < gridY; j++){
			retGrid[i][j] = remGrid[i][j];
		}
	}
	return retGrid;
}
*/

//     0   1   2   3
//   +---+---+---+---+
// 0 |   |   |   |   |
//   +---+---+---+---+
// 1 |   |   |   |   |
//   +---+---+---+---+
// 2 |   |   |   |   |
//   +---+---+---+---+
// 3 |   |   |   |   |
//   +---+---+---+---+

//find all paths going from x to y in the grid
//remGrid has dimensions numX x numY. Holds a 1 in (x,y) if Axy has been assigned, 0 if Axy not assigned
//do not change remGrid; rather, duplicate it and change the new one
SddNode* paths(SddManager* m, int startX, int startY, int endX, int endY, int remGrid[gridX][gridY]){
	SddNode* retNode = sdd_manager_true(m);
	int i, j;
	int thisLit = startY*gridX + startX + 1;
	retNode = sdd_conjoin(sdd_manager_literal(thisLit,m),retNode,m);

	//base case, start spot is end spot
	if (startX == endX && startY == endY){
		for (i = 0; i < gridX; i++) {
			for (j = 0; j < gridY; j++) {
				if (remGrid[i][j] == 0 && (i != endX || j != endY)) {
					retNode = sdd_conjoin(sdd_manager_literal(0-(j*gridX + i + 1),m),retNode,m);
				}
			}
		}
		return retNode;	
	}

	int newGrid[gridX][gridY];

	for (i = 0; i < gridX; i++){
		for(j = 0; j < gridY; j++){
			newGrid[i][j] = remGrid[i][j];
		}
	}

	newGrid[startX][startY] = 1;

	SddNode* orNode = sdd_manager_false(m);

	int newX = startX-1;
	if (newX >= 0 && remGrid[newX][startY] == 0) {
		orNode = sdd_disjoin(orNode,paths(m,newX,startY,endX,endY,newGrid),m);
	}

	newX = startX+1;
	if (newX < gridX && remGrid[newX][startY] == 0) {
		orNode = sdd_disjoin(orNode,paths(m,newX,startY,endX,endY,newGrid),m);
	}

	int newY = startY-1;
	if (newY >= 0 && remGrid[startX][newY] == 0) {
		orNode = sdd_disjoin(orNode,paths(m,startX,newY,endX,endY,newGrid),m);
	}

	newY = startY+1;
	if (newY < gridY && remGrid[startX][newY] == 0) {
		orNode = sdd_disjoin(orNode,paths(m,startX,newY,endX,endY,newGrid),m);
	}
	
	retNode = sdd_conjoin(orNode,retNode,m);

	return retNode;
}

int main(int argc, char** argv) {
	int i,j;

	// initialize manager
	SddLiteral var_count = gridX*gridY; // initial number of variables
	printf("var_count: %ld\n",var_count);
	int auto_gc_and_minimize = 0; // disable (0) or enable (1) auto-gc & auto-min
	SddManager* m = sdd_manager_create(var_count,auto_gc_and_minimize);
	//Vtree* vtree = sdd_vtree_new(var_count, "right");
	//SddManager* m = sdd_manager_new(vtree);

	int remGrid[gridX][gridY];
	for (i = 0; i < gridX; i++){
		for (j = 0; j < gridY; j++){
			remGrid[i][j] = 0;
		}
	}
	
	SddNode* node = paths(m,0,0,2,2,remGrid);
	//node = sdd_condition(1,node,m);
	//node = sdd_condition(4,node,m);
	//node = sdd_condition(-3,node,m);
	//node = sdd_condition(-2,node,m);
	int model_count = sdd_model_count(node,m);
	printf("model_count: %d\n",model_count);
	
	// free manager
	sdd_manager_free(m);
	return 0;
}