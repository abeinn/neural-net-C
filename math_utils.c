#include "matrix.c"
#include <omp.h>
#include <immintrin.h>

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

void mat_lin_combo(matrix *result, matrix *mat1, matrix *mat2, double c1, double c2) {
    // Computes c1 * mat1 + c2 * mat2 for matrices mat1, mat2 and scalars c1, c2

    check_same_dims(mat1, mat2, "mat_lin_combo");
    check_same_dims(mat2, result, "mat_lin_combo");
    size_t length = result->rows * result->cols;
    double *data1 = mat1->data;
    double *data2 = mat2->data;
    double *data = result->data;
    unsigned int i;

    size_t length_for_vec = length / 4 * 4;
    __m256d c1_vec = _mm256_set1_pd(c1);
    __m256d c2_vec = _mm256_set1_pd(c2);

    #pragma omp parallel for
    for (i = 0; i < length_for_vec; i += 4) {
        _mm256_storeu_pd(data + i, 
            _mm256_add_pd(
                _mm256_mul_pd(_mm256_loadu_pd(data1 + i), c1_vec), 
                _mm256_mul_pd(_mm256_loadu_pd(data2 + i), c2_vec)
            )
        );
    }

    for (i = length_for_vec; i < length; i++) {
        data[i] = c1 * data1[i] + c2 * data2[i];
    }
}

void mat_add(matrix *result, matrix *mat1, matrix *mat2) {
    mat_lin_combo(result, mat1, mat2, 1, 1);
}

void mat_sub(matrix *result, matrix *mat1, matrix *mat2) {
    mat_lin_combo(result, mat1, mat2, 1, -1);
}

void mat_vec_add(matrix *result, matrix *mat, matrix *vec) {
    // Add column vector vec to each column of mat

    check_same_dims(result, mat, "mat_vec_add");
    size_t rows = result->rows;
    size_t cols = result->cols;
    unsigned int i, j;

    if (vec->rows != rows || vec->cols != 1) {
        printf("Error: Invalid vector dimensions for add_vec_to_mat\n\n");
        exit(0);
    }
    
    #pragma omp parallel for collapse(2) 
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            mat_set(result, i, j, mat_get(mat, i, j) + mat_get(vec, i, 0));
        }
    }
    
}

void mat_mul_trans(matrix *result, matrix *mat1, matrix *mat2, bool t1, bool t2) {
    // Multiply matrices mat1 and mat2
    // Can transpose mat1 or mat2 before multiplying by setting t1 = true or t2 = true respectively 
    
    size_t rows1 = t1 ? mat1->cols : mat1->rows;
    size_t cols1 = t1 ? mat1->rows : mat1->cols;
    size_t rows2 = t2 ? mat2->cols : mat2->rows;
    size_t cols2 = t2 ? mat2->rows : mat2->cols; 

    double val1, val2, dot_prod;
    unsigned int i, j, k;

    if ((cols1 != rows2) || (rows1 != result->rows) || (cols2 != result->cols)) {
        printf("Error: Dimensions are invalid for mat_mul_trans\n\n");
        exit(0);
    }

    for (i = 0; i < rows1; i++) {
        for (j = 0; j < cols2; j++) {
            dot_prod = 0; 
            for (k = 0; k < cols1; k++) {
                val1 = t1 ? mat_get(mat1, k, i) : mat_get(mat1, i, k);
                val2 = t2 ? mat_get(mat2, j, k) : mat_get(mat2, k, j);
                dot_prod += val1 * val2;
            }
            mat_set(result, i, j, dot_prod);
        }
    }

    // for (i = 0; i < rows1; i++) {

    // }
}

void mat_mul(matrix *result, matrix *mat1, matrix *mat2) {
    mat_mul_trans(result, mat1, mat2, false, false);
}

void mat_scalar_mul(matrix *result, matrix *mat, double c) {
    // Multiply each element in matrix mat by a scalar c

    check_same_dims(result, mat, "mat_scalar_mul");
    size_t length = result->rows * result->cols;
    double *result_data = result->data;
    double *data = mat->data;
    unsigned int i;

    for (i = 0; i < length; i++) {
        result_data[i] = c * data[i];
    }
}

void mat_copy(matrix *result, matrix *mat) {
    mat_scalar_mul(result, mat, 1.0);
}

void mat_elem_mul(matrix *result, matrix *mat1, matrix *mat2) {
    // Element wise mulitplication of two equal sized matrices

    check_same_dims(mat1, mat2, "mat_elem_mul");
    check_same_dims(mat2, result, "mat_elem_mul");
    size_t length = result->rows * result->cols; 

    double *data1 = mat1->data;
    double *data2 = mat2->data;
    double *data = result->data;
    unsigned int i;

    for (i = 0; i < length; i++) {
        data[i] = data1[i] * data2[i];
    }
}

void mat_sum_rows(matrix *result, matrix *mat) {
    // Sets result[i] = sum of values in row i of mat
    // result should be a vector such that result->rows = mat->rows

    size_t rows = mat->rows;
    size_t cols = mat->cols;
    double sum;
    unsigned int i, j;

    if ((result->cols != 1) || (result->rows != rows)) {
        printf("Error: Invalid result dimensions for sum_rows\n\n");
        exit(0);
    }

    for (i = 0; i < rows; i++) {
        sum = 0;
        for (j = 0; j < cols; j++) {
            sum += mat_get(mat, i, j);
        }
        mat_set(result, i, 0, sum);
    }
}

void sigmoid(matrix *result, matrix *mat) {
    // Sigmoid function applied on each element of mat

    check_same_dims(result, mat, "sigmoid");
    size_t length = mat->rows * mat->cols;
    double *data = mat->data;
    double *result_data = result->data;
    unsigned int i;

    for (i = 0; i < length; i++) {
        result_data[i] = 1 / (1 + exp(-1 * data[i]));
    }
}

void dsigmoid(matrix *result, matrix *mat) {
    // Derivative of sigmoid function
    // Expects mat to be result of sigmoid(mat, X) for some X

    size_t length = mat->rows * mat->cols;
    check_same_dims(result, mat, "dsigmoid");
    double *data = mat->data;
    double *result_data = result->data;
    unsigned int i;

    for (i = 0; i < length; i++) {
        result_data[i] = data[i] * (1 - data[i]);
    }
}

void softmax(matrix *result, matrix *mat) {
    // Softmax function Applied to each element/column of mat

    check_same_dims(result, mat, "softmax");
    size_t rows = mat->rows; 
    size_t cols = mat->cols;
    double sum, val, max;
    unsigned int i, j;

    for (j = 0; j < cols; j++) {
        sum = 0;
        max = mat_get(mat, 0, j);
        for (i = 0; i < rows; i++) {
            if (mat_get(mat, i, j) > max) {
                max = mat_get(mat, i, j);
            }
        }
        for (i = 0; i < rows; i++) {
            sum += exp(mat_get(mat, i, j) - max);
        }
        for (i = 0; i < rows; i++) {
            val = exp(mat_get(mat, i, j) - max) / sum;
            mat_set(result, i, j, val);
        }
    }
}

void relu(matrix *result, matrix *mat) {
    // ReLu function applied to each element of mat 

    check_same_dims(result, mat, "relu");
    size_t length = mat->rows * mat->cols;
    double *data = mat->data;
    double *result_data = result->data;
    unsigned int i;

    for (i = 0; i < length; i++) {
        result_data[i] = fmax(0.0, data[i]);
    }
}

void drelu(matrix *result, matrix *mat) {
    // Derivative of ReLu function applied to each element of mat

    check_same_dims(result, mat, "drelu");
    size_t length = mat->rows * mat->cols;
    double *data = mat->data;
    double *result_data = result->data;
    unsigned int i;

    for (i = 0; i < length; i++) {
        result_data[i] = (data[i] > 0.0) ? 1.0 : 0.0;
    }
}

void shuffle_array(int *array, int n) {
    // Randomly shuffle the values in array

    unsigned int i, j, t;

    if (n > 1) {
        for (i = 0; i < n - 1; i++) {
            j = i + rand() / (RAND_MAX / (n - i) + 1);
            t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
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

void mat_get_col(matrix* result, matrix* mat, int idx) {
    // Get column idx of matrix mat 

    size_t rows = mat->rows;
    size_t cols = mat->cols;
    if ((result->rows != rows) || (result->cols != 1)) {
        printf("Error: Result dimensions invalid for mat_get_col");
    }
    for (int i = 0; i < rows; i++) {
        mat_set(result, i, 0, mat_get(mat, i, idx));
    }
}

int max_index(matrix *vec) {
    // Return the index of the maximum value in vec

    size_t rows = vec->rows;
    if (vec->cols != 1 || rows == 0) {
        printf("Error: max_index expects a column vector input\n\n");
        exit(0);
    }

    double *data = vec->data;
    double max_val = data[0];
    unsigned int max_idx, i = 0;

    for (i = 0; i < rows; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return max_idx;
}
