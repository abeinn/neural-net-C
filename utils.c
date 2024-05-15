#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct {
    size_t length;
    double *data;
} vector;

typedef struct {
    size_t m;
    size_t n;
    double **data; 
} matrix;

double rand_weight() { return ((double) rand()) / ((double) RAND_MAX); }

void checkAlloc(void* ptr) {
    if (ptr == NULL) {
        printf("Error: Memory not allocated\n");
        exit(0);
    }
}

vector* zeroVector(size_t length) {
    vector* vec = malloc(sizeof(vector));
    checkAlloc(vec);
    vec->length = length; 
    vec->data = calloc(length, sizeof(double));
    checkAlloc(vec->data);
    return vec;
}

vector* randVector(size_t length) {
    vector* vec = zeroVector(length);
    for (int i = 0; i < length; i++) {
        vec->data[i] = rand_weight();
    }
    return vec;
}

vector* vectorFromArray(double* arr, size_t length) {
    vector* vec = zeroVector(length);
    for (int i = 0; i < length; i++) {
        vec->data[i] = arr[i];
    }
    return vec; 
}

void freeVector(vector* vec) {
    free(vec->data);
    free(vec);
}

void printVector(vector* vec) {
    size_t length = vec->length;
    double* data = vec->data;
    printf("[");
    for (int i = 0; i < length - 1; i++) {
        printf("%g, ", data[i]);
    }
    printf("%g]\n", data[length - 1]);
}

size_t checkLengths(vector* vec1, vector* vec2) {
    size_t length1 = vec1->length;
    size_t length2 = vec2->length;
    if (length1 != length2) {
        printf("Error: Vector lengths are not equal");
        exit(0);
    }
    return length1;
}

void addToVector(vector* vec1, vector* vec2) {
    size_t length = checkLengths(vec1, vec2);
    double* data1 = vec1->data;
    double* data2 = vec2->data;
    for (int i = 0; i < length; i++) {
        data1[i] += data2[i];
    }
}

vector* addVectors(vector* vec1, vector* vec2) {
    size_t length = checkLengths(vec1, vec2);
    double* data1 = vec1->data;
    double* data2 = vec2->data;
    vector* vec = zeroVector(length);
    for (int i = 0; i < length; i++) {
        vec->data[i] = data1[i] + data2[i];
    }
}

void copyVector(vector* vec1, vector* vec2) {
    size_t length = checkLengths(vec1, vec2);
    double* data1 = vec1->data;
    double* data2 = vec2->data;
    for (int i = 0; i < length; i++) {
        data1[i] = data2[i];
    }
}

double dotProduct(vector* vec1, vector* vec2) {
    size_t length = checkLengths(vec1, vec2);
    double* data1 = vec1->data;
    double* data2 = vec2->data;
    double s = 0.0f;
    for (int i = 0; i < length; i++) {
        s += data1[i] * data2[i];
    }
    return s;
}

vector* elemProduct(vector* vec1, vector* vec2) {
    size_t length = checkLengths(vec1, vec2);
    vector* vec = zeroVector(length);
    double* data1 = vec1->data;
    double* data2 = vec2->data;
    double* data = vec->data;
    for (int i = 0; i < length; i++) {
        data[i] = data1[i] * data2[i];
    }
    return vec;
}

matrix* zeroMatrix(size_t length) {
    vector* vec = malloc(sizeof(vector));
    checkAlloc(vec);
    vec->length = length; 
    vec->data = calloc(length, sizeof(double));
    checkAlloc(vec->data);
    return vec;
}

int main(void) {
    double arr1[] = {1, 2, 3};
    double arr2[] = {-1, 4, 2};
    vector* vec1 = vectorFromArray(arr1, 3);
    vector* vec2 = vectorFromArray(arr2, 3);
    printf("%g\n", dotProduct(vec1, vec2));
    vector* vec3 = elemProduct(vec1, vec2);
    printVector(vec3);
}