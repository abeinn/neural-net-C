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
    double **data; 
} matrix;

double rand_weight() { return ((double) rand()) / ((double) RAND_MAX); }

void check_alloc(void* ptr) {
    if (ptr == NULL) {
        printf("Error: Memory not allocated\n");
        exit(0);
    }
}

vector* zero_vector(size_t length) {
    vector* vec = malloc(sizeof(vector));
    check_alloc(vec);
    vec->length = length; 
    vec->data = calloc(length, sizeof(double));
    check_alloc(vec->data);
    return vec;
}

vector* rand_vector(size_t length) {
    vector* vec = zero_vector(length);
    for (int i = 0; i < length; i++) {
        vec->data[i] = rand_weight();
    }
    return vec;
}

vector* vector_from_array(double* arr, size_t length) {
    vector* vec = zero_vector(length);
    for (int i = 0; i < length; i++) {
        vec->data[i] = arr[i];
    }
    return vec; 
}

void free_vector(vector* vec) {
    free(vec->data);
    free(vec);
}

void print_vector(vector* vec) {
    size_t length = vec->length;
    double* data = vec->data;
    printf("[");
    for (int i = 0; i < length - 1; i++) {
        printf("%g, ", data[i]);
    }
    printf("%g]\n", data[length - 1]);
}

size_t check_lengths(vector* vec1, vector* vec2) {
    size_t length1 = vec1->length;
    size_t length2 = vec2->length;
    if (length1 != length2) {
        printf("Error: Vector lengths are not equal");
        exit(0);
    }
    return length1;
}

void add_to_vector(vector* vec1, vector* vec2) {
    size_t length = check_lengths(vec1, vec2);
    double* data1 = vec1->data;
    double* data2 = vec2->data;
    for (int i = 0; i < length; i++) {
        data1[i] += data2[i];
    }
}

vector* add_vectors(vector* vec1, vector* vec2) {
    size_t length = check_lengths(vec1, vec2);
    double* data1 = vec1->data;
    double* data2 = vec2->data;
    vector* vec = zero_vector(length);
    for (int i = 0; i < length; i++) {
        vec->data[i] = data1[i] + data2[i];
    }
}

void copy_vector(vector* vec1, vector* vec2) {
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

vector* elem_product(vector* vec1, vector* vec2) {
    size_t length = check_lengths(vec1, vec2);
    vector* vec = zero_vector(length);
    double* data1 = vec1->data;
    double* data2 = vec2->data;
    double* data = vec->data;
    for (int i = 0; i < length; i++) {
        data[i] = data1[i] * data2[i];
    }
    return vec;
}

matrix* zero_matrix(size_t rows, size_t cols) {
    matrix* mat = malloc(sizeof(matrix));
    check_alloc(mat);
    mat->rows = rows;
    mat->cols = cols; 
    double** data = malloc(rows, sizeof(double*));
    check_alloc(data);
    mat->data = data;
    for (int i = 0; i < rows; i++) {
        data[i] = calloc(cols, sizeof(double));
        check_alloc(data[i]);
    }
    return mat;
}

matrix* rand_matrix(size_t rows, size_t cols) {
    matrix* mat = zero_matrix(rows, cols);
    double** data = mat->data;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = rand_weight();
        }
    }
}

int main(void) {
    double arr1[] = {1, 2, 3};
    double arr2[] = {-1, 4, 2};
    vector* vec1 = vector_from_array(arr1, 3);
    vector* vec2 = vector_from_array(arr2, 3);
    printf("%g\n", dot_product(vec1, vec2));
    vector* vec3 = elem_product(vec1, vec2);
    print_vector(vec3);
}