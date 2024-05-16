#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "neural_network.c"

#define n 2
#define n1 2
#define n2 1
#define m 4

int main(void) {

    srand(12);

    const double lr = 0.1f;
    const double epochs = 10000;

    double X_arr[] = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f};
    matrix *X = mat_from_array(X_arr, n, m);
    double Y_arr[] = {0.0f, 1.0f, 1.0f, 0.0f};
    matrix *Y = mat_from_array(Y_arr, 1, m);

    size_t layer_sizes[] = {2, 2, 1};
    nn_model *model = create_model(3, layer_sizes);
    train_model(model, X, Y, 4, epochs, lr);

    free_model(model);

    free_mat(X);
    free_mat(Y);

}