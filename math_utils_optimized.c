#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "matrix.c"

matrix* zero_mat_opt(size_t rows, size_t cols) { 
    // Create a matrix of all zeroes
    // TODO: NOT OPTIMIZED

    matrix *mat = malloc(sizeof(matrix));
    check_alloc(mat);
    mat->rows = rows;
    mat->cols = cols; 
    mat->data = calloc(rows * cols, sizeof(double));
    check_alloc(mat->data);
    return mat;
}

matrix* rand_mat_opt(size_t rows, size_t cols) {
    // Create a matrix with random values
    // TODO: NOT OPTIMIZED

    matrix *mat = zero_mat(rows, cols);
    double *data = mat->data;
    size_t length = rows * cols;
    unsigned int i;

    for (i = 0; i < length; i++) {
        data[i] = rand_weight();
    }
    return mat; 
}

matrix* mat_from_array_opt(double *arr, size_t rows, size_t cols) {
    // TODO: NOT OPTIMIZED

    matrix *mat = zero_mat(rows, cols);
    double *data = mat->data;
    size_t length = rows * cols;
    unsigned int i;
    for (i = 0; i < length; i++) {
        data[i] = arr[i];
    }
    return mat; 
}

void mat_lin_combo_opt(matrix *result, matrix *mat1, matrix *mat2, double c1, double c2) {
    // Computes c1 * mat1 + c2 * mat2 for matrices mat1, mat2 and scalars c1, c2

    check_same_dims(mat1, mat2, "mat_lin_combo");
    check_same_dims(mat2, result, "mat_lin_combo");
    size_t length = result->rows * result->cols;
    double *data1 = mat1->data;
    double *data2 = mat2->data;
    double *data = result->data;
    unsigned int i;

    for (i = 0; i < length; i++) {
        data[i] = c1 * data1[i] + c2 * data2[i];
    }
}