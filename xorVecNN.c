#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.c"

#define n 2
#define n1 2
#define n2 1
#define m 4

int main(void) {

    srand(12);

    const double lr = 0.1f;
    
    matrix *A1 = zero_mat(n1, m);
    matrix *Z1 = zero_mat(n1, m);
    matrix *A2 = zero_mat(n2, m);
    matrix *Z2 = zero_mat(n2, m);

    matrix *dZ2 = zero_mat(n2, m);
    matrix *dW2 = zero_mat(n2, n1);
    matrix *db2 = zero_mat(n2, 1);
    matrix *dA1 = zero_mat(n1, m);
    matrix *dZ1 = zero_mat(n1, m);
    matrix *dW1 = zero_mat(n1, n);
    matrix *db1 = zero_mat(n1, 1);

    matrix *b1 = rand_mat(n1, 1);
    matrix *b2 = rand_mat(n2, 1);

    matrix *W1 = rand_mat(n1, n);
    matrix *W2 = rand_mat(n2, n1);

    double X_arr[] = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f};
    matrix *X = mat_from_array(X_arr, n, m);
    double Y_arr[] = {0.0f, 1.0f, 1.0f, 0.0f};
    matrix *Y = mat_from_array(Y_arr, 1, m);

    int num_epochs = 10000;
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // forward prop
        mat_mul(Z1, W1, X);
        mat_vec_add(Z1, Z1, b1);
        sigmoid(A1, Z1);
        
        mat_mul(Z2, W2, A1);
        mat_vec_add(Z2, Z2, b2);
        sigmoid(A2, Z2);

        printf("Expected:\n");
        print_mat(Y);
        printf("Predicted:\n");
        print_mat(A2);

        // back prop 
        mat_sub(dZ2, A2, Y);

        mat_mul_trans(dW2, dZ2, A1, false, true);
        mat_scalar_mul(dW2, dW2, 1.0 / m);

        mat_sum_rows(db2, dZ2);
        mat_scalar_mul(db2, db2, 1.0 / m);

        mat_mul_trans(dA1, W2, dZ2, true, false);

        dsigmoid(dZ1, A1);
        mat_elem_mul(dZ1, dA1, dZ1);

        mat_mul_trans(dW1, dZ1, X, false, true);
        mat_scalar_mul(dW1, dW1, 1.0 / m);

        mat_sum_rows(db1, dZ1);
        mat_scalar_mul(db1, db1, 1.0 / m);

        // gradient descent 

        mat_lin_combo(W2, W2, dW2, 1.0, -lr);
        mat_lin_combo(b2, b2, db2, 1.0, -lr);
        mat_lin_combo(W1, W1, dW1, 1.0, -lr);
        mat_lin_combo(b1, b1, db1, 1.0, -lr); 
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

    free_mat(dZ2); 
    free_mat(dW2);  
    free_mat(db2); 
    free_mat(dA1);  
    free_mat(dZ1); 
    free_mat(dW1); 
    free_mat(db1); 

}