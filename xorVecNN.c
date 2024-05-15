#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.c"

#define n 2
#define n1 2
#define n2 1
#define m 4

int main(void) {
    const double lr = 0.1f;
    
    matrix* A0 = zero_mat(n, m);
    matrix* A1 = zero_mat(n1, m);

    vector* b1 = rand_vec(n1);
    vector* b2 = rand_vec(n2);

    matrix* W1 = rand_mat(n1, n);
    matrix* W2 = rand_mat(n2, n1);

    double X_arr[] = {0, 1, 0, 1, 0, 0, 1, 1};
    matrix* X = mat_from_array(X_arr, n, m);
    double Y_arr[] = {0, 1, 1, 0};
    matrix* Y = mat_from_array(Y_arr, 1, m);

    int num_epochs = 10000;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        
    }
    

}