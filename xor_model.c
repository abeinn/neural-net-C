#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "neural_network.c"

#define n 2
#define n1 2
#define n2 2
#define m 4

int main(void) {

    srand(12);

    const double lr = 0.1f;
    const double epochs = 10000;

    double X_arr[] = {0.0f, 1.0f, 0.0f, 1.0f, 
                      0.0f, 0.0f, 1.0f, 1.0f};
    matrix *X = mat_from_array(X_arr, n, m);

    double Y_arr[] = {1.0f, 0.0f, 0.0f, 1.0f,
                      0.0f, 1.0f, 1.0f, 0.0f};
    matrix *Y = mat_from_array(Y_arr, n2, m);

    size_t layer_sizes[] = {n, 10, 25, n2};
    enum func layer_activations[] = {INPUT, RELU, RELU, SOFTMAX};
    nn_model *model = create_model(4, layer_sizes, layer_activations);
    train_model(model, X, Y, m, epochs, lr);

    double test_arr[] = {1.0f, 0.0f};
    matrix *test = mat_from_array(test_arr, 2, 1);
    matrix *pred = zero_mat(2, 1);

    model_predict(model, pred, test, 1);
    print_mat(pred);

    free_model(model);

    free_mat(X);
    free_mat(Y);
    free_mat(test);
    free_mat(pred);

}