#include <stdio.h>
#include <stdlib.h>
#include "sddapi.h"

SddNode* oneInTen(SddManager* m){
	SddNode* delta = sdd_manager_false(m);
	SddNode* alpha;
	int i,j;
	for(i = 1; i < 11; i++) {
		alpha = sdd_manager_true(m);
		for(j = 1; j < 11; j++){
			if (j == i){
				alpha = sdd_conjoin(sdd_manager_literal(j,m),alpha,m);
			}
			else{
				alpha = sdd_conjoin(sdd_manager_literal(-j,m),alpha,m);
			}
		}
		delta = sdd_disjoin(alpha,delta,m);
	}
	return delta;
}

SddNode* lots(SddManager* m){
	SddNode* delta = sdd_manager_false(m);
	SddNode* alpha = sdd_manager_true(m);
	int i,j;
	for (i = 1; i < 10; i++){
		alpha = sdd_conjoin(sdd_manager_literal(i,m),alpha,m);
	}
	delta = sdd_disjoin(alpha,delta,m);
	alpha = sdd_manager_true(m);
	for (i = 1; i < 10; i++){
		alpha = sdd_conjoin(sdd_manager_literal(-i,m),alpha,m);
	}
	delta = sdd_disjoin(alpha,delta,m);
	return delta;
}

// output: SDD that is valid when exactly r out of n inputs are true
SddNode* combinations(int start, int n, int r, int index, int *indices, SddManager* m){  //call with start == 1, index == 0
	SddNode* alpha;
	int i,j;
	if (index == r){
		alpha = sdd_manager_true(m);
		int theIndex = 0;
		int makeTrue = indices[0];
		for (i = 1; i <= n; i++){
			if (i == makeTrue){
				alpha = sdd_conjoin(sdd_manager_literal(i,m),alpha,m);
				theIndex++;
				makeTrue = indices[theIndex];
			}
			else{
				alpha = sdd_conjoin(sdd_manager_literal(-i,m),alpha,m);
			}
		}
		
		/*
		for (j = 0; j < r; j++){
			printf("%d ",indices[j]);
		}
		printf("\n");
		*/
		
		return alpha;
	}
	
	alpha = sdd_manager_false(m);
	
	for(i = start; i <= n && n - i + 1 >= r - index; i++){
		indices[index] = i;
		alpha = sdd_disjoin(alpha,combinations(i+1,n,r,index+1,indices,m),m);
	}
	return alpha;
}


// inputs: manager m, ints r & n
// output: an SDD that is true when exactly r out of n variables are true
SddNode* rInN(SddManager* m, int n, int r){
	SddNode* delta;
	int *indices = malloc(sizeof(int) * r);
	
	delta = combinations(1,n,r,0,indices,m);
	free(indices);
	return delta;
}

int main(int argc, char** argv) {
	// initialize manager
	SddLiteral var_count = 10; // initial number of variables
	int auto_gc_and_minimize = 0; // disable (0) or enable (1) auto-gc & auto-min
	SddManager* m = sdd_manager_create(var_count,auto_gc_and_minimize);
	// CONSTRUCT, MANIPULATE AND QUERY SDDS
	
	SddLiteral A=1,B=2,C=3,D=4,E=5,F=6,G=7,H=8,I=9,J=10;
	
	int r = 4;
	int n = 10;
	//SddNode* delta = rInN(m,n,r);
	
	//SddNode* delta = oneInTen(m);
	SddNode* delta = lots(m);
	
	/*
	delta = sdd_condition(-A,delta,m);
	delta = sdd_condition(-B,delta,m);
	delta = sdd_condition(-C,delta,m);
	delta = sdd_condition(-D,delta,m);
	delta = sdd_condition(-E,delta,m);
	delta = sdd_condition(F,delta,m);
	delta = sdd_condition(-G,delta,m);
	delta = sdd_condition(-H,delta,m);
	delta = sdd_condition(-I,delta,m);
	delta = sdd_condition(-J,delta,m);
	
	
	int is_abnormal = sdd_node_is_false(delta);
	
	if (is_abnormal) {
		printf("abnormal");
		printf("\n");
	}
	else {
		printf("normal");
		printf("\n");
	}
	*/
	
	SddModelCount count = sdd_model_count(delta,m);
	int i = count;
	printf("model count: %d\n",i);
	
	
	// free manager
	sdd_manager_free(m);
	return 0;
}