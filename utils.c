#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct {
    size_t length;
    double *data;
} vector;

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

vector* zero_vec(size_t length) {
    vector* vec = malloc(sizeof(vector));
    check_alloc(vec);
    vec->length = length; 
    vec->data = calloc(length, sizeof(double));
    check_alloc(vec->data);
    return vec;
}

vector* rand_vec(size_t length) {
    vector* vec = zero_vec(length);
    for (int i = 0; i < length; i++) {
        vec->data[i] = rand_weight();
    }
    return vec;
}

vector* vec_from_array(double* arr, size_t length) {
    vector* vec = zero_vec(length);
    for (int i = 0; i < length; i++) {
        vec->data[i] = arr[i];
    }
    return vec; 
}

void free_vec(vector* vec) {
    free(vec->data);
    free(vec);
}

void print_vec(vector* vec) {
    size_t length = vec->length;
    double* data = vec->data;
    printf("[");
    for (int i = 0; i < length - 1; i++) {
        printf("%g, ", data[i]);
    }
    printf("%g]", data[length - 1]);
}

size_t check_lengths(vector* vec1, vector* vec2) {
    size_t length1 = vec1->length;
    size_t length2 = vec2->length;
    if (length1 != length2) {
        printf("Error: Vector lengths are not equal\n\n");
        exit(0);
    } 
    return length1;
}

void add_vecs(vector* result, vector* vec1, vector* vec2) {
    check_lengths(vec1, vec2);
    size_t length = check_lengths(vec2, result);
    double* data1 = vec1->data;
    double* data2 = vec2->data;
    double* data = result->data;
    for (int i = 0; i < length; i++) {
        data[i] = data1[i] + data2[i];
    }
}

void copy_vec(vector* vec1, vector* vec2) {
    size_t length = check_lengths(vec1, vec2);
    double* data1 = vec1->data;
    double* data2 = vec2->data;
    for (int i = 0; i < length; i++) {
        data1[i] = data2[i];
    }
}

double dot_product(vector* vec1, vector* vec2) {
    size_t length = check_lengths(vec1, vec2);
    double* data1 = vec1->data;
    double* data2 = vec2->data;
    double s = 0.0f;
    for (int i = 0; i < length; i++) {
        s += data1[i] * data2[i];
    }
    return s;
}

void elem_product(vector* result, vector* vec1, vector* vec2) {
    check_lengths(vec1, vec2);
    size_t length = check_lengths(vec2, result);
    double* data1 = vec1->data;
    double* data2 = vec2->data;
    double* data = result->data;
    for (int i = 0; i < length; i++) {
        data[i] = data1[i] * data2[i];
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

void mat_get_row(vector* vec, matrix* mat, int i) {
    size_t cols = mat->cols;
    if (cols != vec->length) {
        printf("Error: Invalid vector length for mat_get_row");
        exit(0);
    }
    double* data = vec->data;
    for (int j = 0; j < cols; j++) {
        data[j] = mat_get(mat, i, j);
    }
}

void mat_get_col(vector* vec, matrix* mat, int j) {
    size_t rows = mat->rows;
    if (rows != vec->length) {
        printf("Error: Invalid vector length for mat_get_col");
        exit(0);
    }
    double* data = vec->data;
    for (int i = 0; i < rows; i++) {
        data[i] = mat_get(mat, i, j);
    }
}

void print_mat(matrix* mat) {
    size_t rows = mat->rows; 
    size_t cols = mat->cols;
    vector* vec = zero_vec(cols);
    printf("[");
    for (int i = 0; i < rows - 1; i++) {
        mat_get_row(vec, mat, i);
        print_vec(vec);
        printf(",\n");
    }
    mat_get_row(vec, mat, rows - 1);
    print_vec(vec);
    printf("]");
    free_vec(vec);
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

void add_mats(matrix* result, matrix* mat1, matrix* mat2) {
    check_same_dims(mat1, mat2);
    check_same_dims(mat2, result);
    size_t rows = result->rows;
    size_t cols = result->cols;
    size_t length = rows * cols; 
    double* data1 = mat1->data;
    double* data2 = mat2->data;
    double* data = result->data;
    for (int i = 0; i < length; i++) {
        data[i] = data1[i] + data2[i];
    }
}

void mat_mul(matrix* result, matrix* mat1, matrix* mat2) {
    check_mul_dims(mat1, mat2);
    size_t rows = mat1->rows;
    size_t cols = mat2->cols;
    if ((rows != result->rows) || (cols != result->cols)) {
        printf("Error: Result dimensions are invalid in mat_mul\n\n");
        exit(0);
    }

    vector* row = zero_vec(mat1->cols);
    vector* col = zero_vec(mat2->rows);
    for (int i = 0; i < rows; i++){ 
        for (int j = 0; j < cols; j++) {
            mat_get_row(row, mat1, i);
            mat_get_col(col, mat2, j);
            mat_set(result, i, j, dot_product(row, col));
        }
    }

    free(row);
    free(col);
}

int main(void) {
    double arr1[] = {1, 2, 3, 4, 5, 6};
    double arr2[] = {-1, 2, 0, 1, 4, 5};
    matrix* mat1 = mat_from_array(arr1, 2, 3);
    matrix* mat2 = mat_from_array(arr2, 3, 2);
    print_mat(mat1);
    printf("\n\n");
    print_mat(mat2);
    printf("\n\n");

    matrix* mat3 = zero_mat(2, 2);
    mat_mul(mat3, mat1, mat2);
    print_mat(mat3);
    printf("\n\n");

    free_mat(mat1);
    free_mat(mat2);
    free_mat(mat3);
}