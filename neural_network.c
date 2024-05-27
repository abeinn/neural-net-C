#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "math_utils.c"

enum func {
    INPUT,
    RELU,
    SIGMOID,
    SOFTMAX
};

typedef struct {
    size_t num_nodes;
    enum func activation;
    matrix *W;
    matrix *b;
    matrix *A;
    matrix *Z;
    matrix *dW;
    matrix *db;
    matrix *dA;
    matrix *dZ;
    matrix *V_dW;
    matrix *V_db;
    matrix *S_dW;
    matrix *S_db;
} nn_layer; 

typedef struct {
    size_t num_layers;
    nn_layer *layers; 
} nn_model;

nn_model* create_model(size_t num_layers, size_t *layer_sizes, enum func *layer_activations) {
    nn_model *model = malloc(sizeof(nn_model));

    check_alloc(model);
    model->num_layers = num_layers;

    nn_layer *layers = calloc(num_layers, sizeof(nn_layer));
    check_alloc(layers);
    model->layers = layers;
    unsigned int i;

    // Initalize weights and layer sizes
    layers[0].num_nodes = layer_sizes[0];
    size_t n_curr, n_prev;
    for (i = 1; i < num_layers; i++) {
        n_curr = layer_sizes[i];
        n_prev = layer_sizes[i - 1];
        layers[i].num_nodes = n_curr;

        layers[i].W = rand_mat(n_curr, n_prev);
        layers[i].dW = zero_mat(n_curr, n_prev);
        layers[i].S_dW = zero_mat(n_curr, n_prev);
        layers[i].V_dW = zero_mat(n_curr, n_prev);
        
        layers[i].b = rand_mat(n_curr, 1);
        layers[i].db = zero_mat(n_curr, 1);
        layers[i].S_db = zero_mat(n_curr, 1);
        layers[i].V_db = zero_mat(n_curr, 1);

        layers[i].activation = layer_activations[i];
    }
    
    return model;
}

void mini_batch(matrix *mini_X, matrix *mini_Y, matrix *X, matrix *Y, int *indices) {
    // Put random subset of X and Y into mini_X and mini_Y respectively 

    size_t rows_X = X->rows;
    size_t rows_Y = Y->rows;
    size_t cols = mini_X->cols;
    size_t n = X->cols; 
    shuffle_array(indices, n);
    unsigned int index, i_X, i_Y, j; 

    for (j = 0; j < cols; j++) {
        index = indices[j];
        for (i_X = 0; i_X < rows_X; i_X++) {
            mat_set(mini_X, i_X, j, mat_get(X, i_X, index));
        }
        for (i_Y = 0; i_Y < rows_Y; i_Y++) {
            mat_set(mini_Y, i_Y, j, mat_get(Y, i_Y, index));
        }
    }
}

void forward_prop(nn_model *model) {
    size_t num_layers = model->num_layers;
    nn_layer *layers = model->layers;
    unsigned int i;

    for (i = 1; i < num_layers; i++) {
        mat_mul(layers[i].Z, layers[i].W, layers[i - 1].A);
        mat_vec_add(layers[i].Z, layers[i].Z, layers[i].b);

        if (layers[i].activation == SIGMOID) {
            sigmoid(layers[i].A, layers[i].Z);
        } else if (layers[i].activation == SOFTMAX) {
            softmax(layers[i].A, layers[i].Z);
        } else if (layers[i].activation == RELU) {
            relu(layers[i].A, layers[i].Z);
        }
    }
}

void model_predict(nn_model *model, matrix *result, matrix *input, size_t num_inputs) {
    // Evaluate a trained model on input

    size_t num_layers = model->num_layers;
    nn_layer *layers = model->layers;
    size_t n_out = layers[num_layers - 1].num_nodes;
    size_t n_in = layers[0].num_nodes;
    if ((result->cols != num_inputs) || (result->rows != n_out)) {
        printf("Error: Invalid result vector for model_predict\n\n");
        exit(0);
    } else if ((input->cols != num_inputs) || (input->rows != n_in)) {
        printf("Error: Invalid input vector for model_predict\n\n");
        exit(0);
    }

    layers[0].A = input;
    for (int i = 1; i < num_layers; i++) {
        // Initialize intermediate vectors 
        layers[i].A = zero_mat(layers[i].num_nodes, num_inputs);
        layers[i].Z = zero_mat(layers[i].num_nodes, num_inputs);
    }
    
    // Evaluate model on input using trained weights
    forward_prop(model);

    mat_copy(result, layers[num_layers - 1].A);

    layers[0].A = NULL;
    for (int i = 1; i < num_layers; i++) {
        free_mat(layers[i].A);
        free_mat(layers[i].Z);
        layers[i].A = NULL;
        layers[i].Z = NULL;
    }
    
}

void grad_descent_adam(nn_layer *layers, int i, int epoch, double lr, double beta_1, double beta_2, double epsilon) {

    size_t length_w = layers[i].dW->rows * layers[i].dW->cols;
    size_t length_b = layers[i].db->rows;
    unsigned int j;
    epoch++;

    double *data_vw = layers[i].V_dW->data;
    double *data_vb = layers[i].V_db->data;
    double *data_sw = layers[i].S_dW->data;
    double *data_sb = layers[i].S_db->data;
    double *data_dw = layers[i].dW->data;
    double *data_db = layers[i].db->data;
    double *data_w = layers[i].W->data;
    double *data_b = layers[i].b->data;

    mat_lin_combo(layers[i].V_dW, layers[i].V_dW, layers[i].dW, beta_1, 1 - beta_1);
    mat_lin_combo(layers[i].V_db, layers[i].V_db, layers[i].db, beta_1, 1 - beta_1);

    double corr_1 = (1 - pow(beta_1, (double) epoch));
    double corr_2 = (1 - pow(beta_2, (double) epoch));

    #pragma omp parallel for 
    for (j = 0; j < length_w; j++) {
        data_sw[j] = beta_2 * data_sw[j] + (1 - beta_2) * pow(data_dw[j], 2.0);
        double vw_corr = data_vw[j] / corr_1;
        double sw_corr = data_sw[j] / corr_2;
        data_w[j] -= lr * vw_corr / (sqrt(sw_corr) + epsilon);
    }   


    #pragma omp parallel for 
    for (j = 0; j < length_b; j++) {
        data_sb[j] = beta_2 * data_sb[j] + (1 - beta_2) * pow(data_db[j], 2.0);
        double vb_corr = data_vb[j] / corr_1;
        double sb_corr = data_sb[j] / corr_2;
        data_b[j] -= lr * vb_corr / (sqrt(sb_corr) + epsilon);
    }
}

void train_model(nn_model *model, matrix *X, matrix *Y, size_t mini_batch_size, int epochs, double lr, double beta_1, double beta_2, double epsilon) {
    nn_layer *layers = model->layers;
    size_t num_layers = model->num_layers;
    int last_i = num_layers - 1;
    size_t m = mini_batch_size;
    size_t training_set_size = X->cols;
    unsigned int i, j, epoch;
    double loss;
    double small_val = pow(10, -16.0); 

    matrix *mini_X = zero_mat(X->rows, m);
    matrix *mini_Y = zero_mat(Y->rows, m);

    int *indices = malloc(training_set_size * sizeof(int));
    check_alloc(indices);
    for (i = 0; i < training_set_size; i++) {
        indices[i] = i; 
    }

    // Create matrices used in forward and back prop
    layers[0].A = mini_X; 
    size_t n_curr; 
    for (i = 1; i < num_layers; i++) {
        n_curr = layers[i].num_nodes;
        layers[i].A = zero_mat(n_curr, m);
        layers[i].Z = zero_mat(n_curr, m);
        layers[i].dA = zero_mat(n_curr, m);
        layers[i].dZ = zero_mat(n_curr, m);
    }

    printf("Training neural network model\n");
    clock_t start = clock();

    for (epoch = 0; epoch < epochs; epoch++) {

        mini_batch(mini_X, mini_Y, X, Y, indices);

        // Forward propagation
        forward_prop(model);

        // Back propagation

        // Compute dZ for last layer
        mat_sub(layers[last_i].dZ, layers[last_i].A, mini_Y);

        for (i = last_i; i > 0; i--) {
            if (i != last_i) {
                mat_mul_trans(layers[i].dA, layers[i + 1].W, layers[i + 1].dZ, true, false);

                if (layers[i].activation == SIGMOID) {
                    dsigmoid(layers[i].dZ, layers[i].A);
                } else if (layers[i].activation == RELU) {
                    drelu(layers[i].dZ, layers[i].Z);
                }

                mat_elem_mul(layers[i].dZ, layers[i].dA, layers[i].dZ);
            }
            
            mat_mul_trans(layers[i].dW, layers[i].dZ, layers[i - 1].A, false, true);
            mat_scalar_mul(layers[i].dW, layers[i].dW, 1.0 / m);

            mat_sum_rows(layers[i].db, layers[i].dZ);
            mat_scalar_mul(layers[i].db, layers[i].db, 1.0 / m);

        }

        // Gradient descent
        for (i = last_i; i > 0; i--) {

            // Adam optimizer 
            grad_descent_adam(layers, i, epoch, lr, beta_1, beta_2, epsilon);

            // mat_lin_combo(layers[i].W, layers[i].W, layers[i].dW, 1.0, -lr);
            // mat_lin_combo(layers[i].b, layers[i].b, layers[i].db, 1.0, -lr);
        }

        // Calculate loss
        loss = 0.0;
        for (i = 0; i < Y->rows; i++) {
            for (j = 0; j < m; j++) {
                loss -= mat_get(mini_Y, i, j) * log(mat_get(layers[last_i].A, i, j) + small_val);
            }
        }    
        loss /= (double) m;

        if ((epoch + 1) % 1 == 0) {
            printf("Epoch %d/%d     Loss: %g\n", epoch + 1, epochs, loss);
        }
    }

    clock_t end = clock();
    printf("Finished training\n");
    printf("Time taken: %Lf s\n\n", (long double)(end - start) / CLOCKS_PER_SEC);

    layers[0].A = NULL;
    for (i = 1; i < num_layers; i++) {
        free_mat(layers[i].A);
        free_mat(layers[i].Z);
        free_mat(layers[i].dA);
        free_mat(layers[i].dZ);
        layers[i].A = NULL;
        layers[i].Z = NULL;
        layers[i].dA = NULL;
        layers[i].dZ = NULL;
    }

    double corrects = 0.0;
    matrix *Y_hat = zero_mat(Y->rows, Y->cols);
    matrix *y = zero_mat(Y->rows, 1);
    matrix *y_hat = zero_mat(Y->rows, 1);

    printf("Evaluating model on training data\n");
    model_predict(model, Y_hat, X, X->cols);

    printf("Calculating training accuracy\n");
    for (i = 0; i < Y->cols; i++) {
        mat_get_col(y, Y, i);
        mat_get_col(y_hat, Y_hat, i);

        if (max_index(y) == max_index(y_hat)) {
            corrects += 1.0;
        }
    }

    printf("Training accuracy: %g%%\n\n", 100.0 * corrects / (double) Y->cols);

    free_mat(mini_X);
    free_mat(mini_Y);
    free_mat(Y_hat);
    free_mat(y);
    free_mat(y_hat);
    free(indices);
}

void free_model(nn_model *model) {
    if (model == NULL) {
        return;
    }

    size_t num_layers = model->num_layers;
    nn_layer *layers = model->layers;
    unsigned int i;

    for (i = 0; i < num_layers; i++) {
        free_mat(layers[i].W);
        free_mat(layers[i].b);
        free_mat(layers[i].A);
        free_mat(layers[i].Z);
        free_mat(layers[i].dW);
        free_mat(layers[i].db);
        free_mat(layers[i].dA);
        free_mat(layers[i].dZ);
        free_mat(layers[i].V_dW);
        free_mat(layers[i].V_db);
        free_mat(layers[i].S_dW);
        free_mat(layers[i].S_db);
    }

    free(layers);
    free(model);
}