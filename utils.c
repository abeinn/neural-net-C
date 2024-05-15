#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct {
    size_t rows;
    size_t cols;
    double *data; 
} matrix;

double rand_weight() { return ((double) rand()) / ((double) RAND_MAX); }

void check_alloc(void* ptr) {
    if (ptr == NULL) {
        printf("Error: Memory not allocated\n");
        exit(0);
    }
}

matrix* zero_mat(size_t rows, size_t cols) {
    matrix* mat = malloc(sizeof(matrix));
    check_alloc(mat);
    mat->rows = rows;
    mat->cols = cols; 
    mat->data = calloc(rows * cols, sizeof(double));
    check_alloc(mat->data);
    return mat;
}

double mat_get(matrix* mat, int i, int j) { return mat->data[i * mat->cols + j]; }

void mat_set(matrix* mat, int i, int j, double val) { mat->data[i * mat->cols + j] = val; }

matrix* rand_mat(size_t rows, size_t cols) {
    
    matrix* mat = zero_mat(rows, cols);
    double* data = mat->data;
    size_t length = rows * cols;
    for (int i = 0; i < length; i++) {
        data[i] = rand_weight();
    }
    return mat; 
}

matrix* mat_from_array(double* arr, size_t rows, size_t cols) {
    matrix* mat = zero_mat(rows, cols);
    double* data = mat->data;
    size_t length = rows * cols;
    for (int i = 0; i < length; i++) {
        data[i] = arr[i];
    }
    return mat; 
}

void free_mat(matrix* mat) {
    free(mat->data);
    free(mat);
}

void print_mat(matrix* mat) {
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

void check_same_dims(matrix* mat1, matrix* mat2) {
    size_t rows1 = mat1->rows;
    size_t cols1 = mat1->cols;
    size_t rows2 = mat2->rows;
    size_t cols2 = mat2->cols; 

    if ((rows1 != rows2) || (cols1 != cols2)) {
        printf("Error: Matrices dimensions are not equal\n\n");
        exit(0);
    }
}

void check_mul_dims(matrix* mat1, matrix* mat2) {
    size_t cols1 = mat1->cols;
    size_t rows2 = mat2->rows;
    if (cols1 != rows2) {
        printf("Error: Matrices dimensions invalid for multiplication\n\n");
        exit(0);
    }
}

void mat_add(matrix* result, matrix* mat1, matrix* mat2) {
    check_same_dims(mat1, mat2);
    check_same_dims(mat2, result);
    size_t length = result->rows * result->cols;
    double* data1 = mat1->data;
    double* data2 = mat2->data;
    double* data = result->data;
    for (int i = 0; i < length; i++) {
        data[i] = data1[i] + data2[i];
    }
}

void mat_vec_add(matrix* result, matrix* mat1, matrix* vec) {
    check_same_dims(result, mat1);
    size_t rows = result->rows;
    size_t cols = result->cols;
    if (vec->rows != rows || vec->cols != 1) {
        printf("Error: Invalid vector dimensions for add_vec_to_mat\n\n");
        exit(0);
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat_set(result, i, j, mat_get(mat1, i, j) + mat_get(vec, i, 1));
        }
    }
}

void mat_elem_mul(matrix* result, matrix* mat1, matrix* mat2) {
    check_same_dims(mat1, mat2);
    check_same_dims(mat2, result);
    size_t length = result->rows * result->cols; 
    double* data1 = mat1->data;
    double* data2 = mat2->data;
    double* data = result->data;
    for (int i = 0; i < length; i++) {
        data[i] = data1[i] * data2[i];
    }
}

void mat_mul(matrix* result, matrix* mat1, matrix* mat2) {
    check_mul_dims(mat1, mat2);
    size_t rows1 = mat1->rows;
    size_t cols1 = mat1->cols;
    size_t cols2 = mat2->cols;
    if ((rows1 != result->rows) || (cols2 != result->cols)) {
        printf("Error: Result dimensions are invalid in mat_mul\n\n");
        exit(0);
    }

    int dot_prod; 
    for (int i = 0; i < rows1; i++){ 
        for (int j = 0; j < cols2; j++) {
            dot_prod = 0;
            for (int k = 0; k < cols1; k++) {
                dot_prod += mat_get(mat1, i, k) * mat_get(mat2, k, j);
            }
            mat_set(result, i, j, dot_prod);
        }
    }
}

void mat_scalar_mul(matrix* result, matrix* mat, double c) {
    check_same_dims(result, mat);
    size_t length = result->rows * result->cols;
    double* result_data = result->data;
    double* data = mat->data;
    for (int i = 0; i < length; i++) {
        result_data[i] = c * data[i];
    }
}

void sigmoid(matrix* result, matrix* mat) {
    size_t length = mat->rows * mat->cols;
    double* data = mat->data;
    double* result_data = result->data;
    for (int i = 0; i < length; i++) {
        result_data[i] = 1 / (1 + exp(-1 * data[i]));
    }
}

