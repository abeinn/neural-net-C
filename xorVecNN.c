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
    
    matrix* A1 = zero_mat(n1, m);
    matrix* Z1 = zero_mat(n1, m);
    matrix* A2 = zero_mat(n2, m);
    matrix* Z2 = zero_mat(n2, m);

    matrix* b1 = rand_mat(n1, 1);
    matrix* b2 = rand_mat(n2, 1);

    matrix* W1 = rand_mat(n1, n);
    matrix* W2 = rand_mat(n2, n1);

    double X_arr[] = {0, 1, 0, 1, 0, 0, 1, 1};
    matrix* X = mat_from_array(X_arr, n, m);
    double Y_arr[] = {0, 1, 1, 0};
    matrix* Y = mat_from_array(Y_arr, 1, m);

    int num_epochs = 10000;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        mat_mul(Z1, W1, X);
        add_vec_to_mat(Z1, Z1, b1);
        sigmoid(A1, Z1);
        
        mat_mul(Z2, W2, A1);
        add_vec_to_mat(Z2, Z2, b2);
        sigmoid(A2, Z2);

        
    }
    
    free_mat(A2);
    free_mat(A1);
    free_mat(Z2);
    free_mat(Z1);
    free_mat(b1);
    free_mat(b2);
    free_mat(W1);
    free_mat(W2);
    free_mat(X);
    free_mat(Y);

}