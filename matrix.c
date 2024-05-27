#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

typedef struct {
    size_t rows;
    size_t cols;
    double *data; 
} matrix;

double rand_weight() { return ((double) rand()) / ((double) RAND_MAX); }

void check_alloc(void *ptr) {
    if (ptr == NULL) {
        printf("Error: Memory not allocated\n");
        exit(0);
    }
}

matrix* zero_mat(size_t rows, size_t cols) {
    // Create a matrix of all zeroes

    matrix *mat = malloc(sizeof(matrix));
    check_alloc(mat);
    mat->rows = rows;
    mat->cols = cols; 
    mat->data = calloc(rows * cols, sizeof(double));
    check_alloc(mat->data);
    return mat;
}

matrix* rand_mat(size_t rows, size_t cols) {
    // Create a matrix with random values

    matrix *mat = zero_mat(rows, cols);
    double *data = mat->data;
    size_t length = rows * cols;
    unsigned int i;

    for (i = 0; i < length; i++) {
        data[i] = rand_weight();
    }
    return mat; 
}

matrix* mat_from_array(double *arr, size_t rows, size_t cols) {
    matrix *mat = zero_mat(rows, cols);
    double *data = mat->data;
    size_t length = rows * cols;
    unsigned int i;
    for (i = 0; i < length; i++) {
        data[i] = arr[i];
    }
    return mat; 
}

void free_mat(matrix *mat) {
    if (mat == NULL) {
        return;
    }
    if (mat->data != NULL) {
        free(mat->data);
    }
    free(mat);
}

double mat_get(matrix *mat, int i, int j) { 
    if ((i >= mat->rows) || (i < 0) || (j >= mat->cols) || (j < 0)){
        printf("Error: Index out of bounds for mat_get\n\n");
        exit(0);
    }
    return mat->data[i * mat->cols + j]; 
}

void mat_set(matrix *mat, int i, int j, double val) { 
    if ((i >= mat->rows) || (i < 0) || (j >= mat->cols) || (j < 0)){
        printf("Error: Index out of bounds for mat_set\n\n");
        exit(0);
    }
    mat->data[i * mat->cols + j] = val; 
}

void print_mat(matrix *mat) {
    size_t rows = mat->rows; 
    size_t cols = mat->cols;
    printf("[");
    for (int i = 0; i < rows; ++i) {
        printf("[");
        for (int j = 0; j < cols; ++j) {
            printf("%g", mat_get(mat, i, j));
            if (j < cols - 1) {
                printf(", ");
            }
        }
        printf("]");
        if (i < rows - 1) {
            printf(", \n ");
        }
    }
    printf("]\n\n");
}

void check_same_dims(matrix *mat1, matrix *mat2, char *func_name) {
    // Checks if matrices mat1, mat2 have the same dimensions

    if ((mat1->rows != mat2->rows) || (mat1->cols != mat2->cols)) {
        printf("Error: Matrices dimensions are not equal for function %s\n\n", func_name);
        exit(0);
    }
}

bool mat_is_equal(matrix *mat1, matrix *mat2) {
    check_same_dims(mat1, mat2, "mat_is_equal");
    size_t length = mat1->cols * mat1->rows;
    double *data1 = mat1->data;
    double *data2 = mat2->data;
    unsigned int i;

    for (i = 0; i < length; i++) {
        if (data1[i] != data2[i]) {
            return false;
        }
    }
    return true; 
}