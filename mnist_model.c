#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "neural_network.c"

#define MAX_LINE_LENGTH 8192
#define INPUT_SIZE 784
#define OUTPUT_CLASSES 10
#define TRAINING_SET_SIZE 42000

void load_training_data(matrix *X, matrix *Y, char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file");
        exit(0);
    }

    printf("Loading data from file\n");

    char line[MAX_LINE_LENGTH];
    char *field; 
    double val;
    fgets(line, sizeof(line), file);
    for (int i = 0; i < TRAINING_SET_SIZE; i++) {
        fgets(line, sizeof(line), file);
        if ((i + 1) % 2000 == 0) {
            printf("Data loading progress: %d/%d\n", i + 1, TRAINING_SET_SIZE);
        }

        field = strtok(line, ",");
        val = strtod(field, NULL);
        mat_set(Y, val, i, 1.0f);
        field = strtok(NULL, ",");
        for (int j = 0; j < INPUT_SIZE; j++) {
            val = strtod(field, NULL) / 255.0;
            mat_set(X, j, i, val);
            field = strtok(NULL, ",");
        }
    }

    fclose(file);
    printf("Finished loading data\n\n");
    
}

int main(void) {
    srand(123);
    const double lr = 0.5f;
    const double epochs = 10;
    const size_t mini_batch_size = 64;

    matrix *X = zero_mat(INPUT_SIZE, TRAINING_SET_SIZE);
    matrix *Y = zero_mat(OUTPUT_CLASSES, TRAINING_SET_SIZE);
    load_training_data(X, Y, "train.csv");
    
    size_t num_layers = 2;
    size_t layer_sizes[] = {INPUT_SIZE, 10};
    enum func layer_activations[] = {INPUT, SOFTMAX};
    nn_model *model = create_model(num_layers, layer_sizes, layer_activations);

    train_model(model, X, Y, mini_batch_size, epochs, lr);

    free_model(model);
    free_mat(X);
    free_mat(Y);
}