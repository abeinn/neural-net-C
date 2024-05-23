#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "neural_network.c"

#define MAX_LINE_LENGTH 8192
#define INPUT_SIZE 784
#define OUTPUT_CLASSES 10
#define TRAINING_SET_SIZE 10000
#define TEST_SET_SIZE 28000

void load_training_data(matrix *X, matrix *Y, char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file %s\n", filename);
        exit(0);
    }

    printf("Loading training data from file %s\n", filename);

    char line[MAX_LINE_LENGTH];
    char *field; 
    double val;
    unsigned int i, j;

    fgets(line, sizeof(line), file);
    for (i = 0; i < TRAINING_SET_SIZE; i++) {
        fgets(line, sizeof(line), file);
        if ((i + 1) % 2000 == 0) {
            printf("Data loading progress: %d/%d\n", i + 1, TRAINING_SET_SIZE);
        }

        field = strtok(line, ",");
        val = strtod(field, NULL);
        mat_set(Y, val, i, 1.0f);
        field = strtok(NULL, ",");
        for (j = 0; j < INPUT_SIZE; j++) {
            val = strtod(field, NULL) / 255.0;
            mat_set(X, j, i, val);
            field = strtok(NULL, ",");
        }
    }

    fclose(file);
    printf("Finished loading data\n\n");
}

void load_test_data(matrix *X, char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file %s\n", filename);
        exit(0);
    }

    printf("Loading test data from %s\n", filename);

    char line[MAX_LINE_LENGTH];
    char *field;
    double val;
    unsigned int i, j;

    fgets(line, sizeof(line), file);
    for (i = 0; i < TEST_SET_SIZE; i++) {
        fgets(line, sizeof(line), file);
        if ((i + 1) % 2000 == 0) {
            printf("Data loading progress: %d/%d\n", i + 1, TEST_SET_SIZE);
        }

        field = strtok(line, ",");
        for (j = 0; j < INPUT_SIZE; j++) {
            val = strtod(field, NULL) / 255.0;
            mat_set(X, j, i, val);
            field = strtok(NULL, ",");
        }
    }

    printf("Finished loading data\n\n");
    fclose(file);
}

void write_prediction(matrix *Y, char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file %s\n", filename);
        exit(0);
    }

    printf("Writing predictions to file %s\n", filename);

    size_t rows = Y->rows;
    size_t cols = Y->cols;
    unsigned int i, pred;

    matrix *y = zero_mat(rows, 1);

    fprintf(file, "ImageId,Label\n");

    for (i = 0; i < cols; i++) {
        mat_get_col(y, Y, i);
        pred = max_index(y);
        fprintf(file, "%d,%d\n", i + 1, pred);
    }

    printf("Finished writing data\n\n");
    free_mat(y);
    fclose(file);
}

int main(void) {
    srand(123);
    const double lr = 0.001f;
    const double epochs = 1000;
    const size_t mini_batch_size = 64;

    matrix *train_X = zero_mat(INPUT_SIZE, TRAINING_SET_SIZE);
    matrix *Y = zero_mat(OUTPUT_CLASSES, TRAINING_SET_SIZE);
    load_training_data(train_X, Y, "train.csv");
    
    size_t num_layers = 3;
    size_t layer_sizes[] = {INPUT_SIZE, 128, 10};
    enum func layer_activations[] = {INPUT, RELU, SOFTMAX};
    nn_model *model = create_model(num_layers, layer_sizes, layer_activations);

    train_model(model, train_X, Y, mini_batch_size, epochs, lr);

    matrix *test_X = zero_mat(INPUT_SIZE, TEST_SET_SIZE);
    matrix *Y_hat = zero_mat(OUTPUT_CLASSES, TEST_SET_SIZE);
    load_test_data(test_X, "test.csv");

    model_predict(model, Y_hat, test_X, TEST_SET_SIZE);
    write_prediction(Y_hat, "output.csv");

    free_model(model);
    free_mat(train_X);
    free_mat(Y);
    free_mat(test_X);
    free_mat(Y_hat);
}