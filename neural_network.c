#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.c"

typedef struct {
    size_t num_nodes;
    matrix *W;
    matrix *b;
    matrix *A;
    matrix *Z;
    matrix *dW;
    matrix *db;
    matrix *dA;
    matrix *dZ;
} nn_layer; 

typedef struct {
    size_t num_layers;
    nn_layer *layers; 
} nn_model;

nn_model* create_model(size_t num_layers, size_t *layer_sizes) {
    nn_model *model = malloc(sizeof(nn_model));

    check_alloc(model);
    model->num_layers = num_layers;

    nn_layer *layers = calloc(num_layers, sizeof(nn_layer));
    check_alloc(layers);
    model->layers = layers;

    layers[0].num_nodes = layer_sizes[0];
    size_t n_curr, n_prev;
    for (int i = 1; i < num_layers; i++) {
        n_curr = layer_sizes[i];
        n_prev = layer_sizes[i - 1];
        layers[i].num_nodes = n_curr;
        layers[i].W = rand_mat(n_curr, n_prev);
        layers[i].b = rand_mat(n_curr, 1);
        layers[i].dW = zero_mat(n_curr, n_prev);
        layers[i].db = zero_mat(n_curr, 1);
    }
    
    return model;
}

void train_model(nn_model *model, matrix *X, matrix *Y, size_t m, int epochs, double lr) {
    nn_layer *layers = model->layers;
    size_t num_layers = model->num_layers;
    int last_i = num_layers - 1;

    layers[0].A = X; 
    size_t n_curr; 
    for (int i = 1; i < num_layers; i++) {
        n_curr = layers[i].num_nodes;
        layers[i].A = zero_mat(n_curr, m);
        layers[i].Z = zero_mat(n_curr, m);
        layers[i].dA = zero_mat(n_curr, m);
        layers[i].dZ = zero_mat(n_curr, m);
    }

    for (int epoch = 0; epoch < epochs; epoch++) {

        // Forward propagation
        for (int i = 1; i < num_layers; i++) {
            mat_mul(layers[i].Z, layers[i].W, layers[i - 1].A);
            mat_vec_add(layers[i].Z, layers[i].Z, layers[i].b);
            sigmoid(layers[i].A, layers[i].Z);
        }

        printf("Expected:\n");
        print_mat(Y);
        printf("Predicted: \n");
        print_mat(layers[last_i].A);

        // Back propagation
        mat_sub(layers[last_i].dZ, layers[last_i].A, Y);
        for (int i = last_i; i > 0; i--) {
            
            if (i != last_i) {
                mat_mul_trans(layers[i].dA, layers[i + 1].W, layers[i + 1].dZ, true, false);

                dsigmoid(layers[i].dZ, layers[i].A);
                mat_elem_mul(layers[i].dZ, layers[i].dA, layers[i].dZ);
            }
            
            mat_mul_trans(layers[i].dW, layers[i].dZ, layers[i - 1].A, false, true);
            mat_scalar_mul(layers[i].dW, layers[i].dW, 1.0 / m);

            mat_sum_rows(layers[i].db, layers[i].dZ);
            mat_scalar_mul(layers[i].db, layers[i].db, 1.0 / m);

        }

        // Gradient descent
        for (int i = last_i; i > 0; i--) {
            mat_lin_combo(layers[i].W, layers[i].W, layers[i].dW, 1.0, -lr);
            mat_lin_combo(layers[i].b, layers[i].b, layers[i].db, 1.0, -lr);
        }
    }
}

void free_if_valid(matrix *mat) {
    if (mat != NULL) {
        free_mat(mat);
    }
}

void free_model(nn_model *model) {
    size_t num_layers = model->num_layers;
    nn_layer *layers = model->layers;
    for (int i = 0; i < num_layers; i++) {
        free_if_valid(layers[i].W);
        free_if_valid(layers[i].b);
        free_if_valid(layers[i].A);
        free_if_valid(layers[i].Z);
        free_if_valid(layers[i].dW);
        free_if_valid(layers[i].db);
        free_if_valid(layers[i].dA);
        free_if_valid(layers[i].dZ);
    }
    free(layers);
    free(model);
}