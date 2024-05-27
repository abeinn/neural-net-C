#include "matrix.c"
#include <immintrin.h>

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
    
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            mat_set(result, i, j, mat_get(mat, i, j) + mat_get(vec, i, 0));
        }
    }
    
}

void transpose(matrix *result, matrix *mat) {
    size_t rows = mat->rows;
    size_t cols = mat->cols;
    double *data = mat->data;
    double *result_data = result->data;
    unsigned int i, j;

    if ((rows != result-> cols) || (cols != result->rows)) {
        printf("Invalid result dimensions for tranpose\n\n");
        exit(0);
    }

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            result_data[j * rows + i] = data[i * cols + j];
        }
    }
}

void mat_mul(matrix *result, matrix *mat1, matrix *mat2) {

    // TODO: Make result all zeroes 

    size_t rows1 = mat1->rows;
    size_t cols1 = mat1->cols;
    size_t rows2 = mat2->rows;
    size_t cols2 = mat2->cols; 

    if ((cols1 != rows2) || (rows1 != result->rows) || (cols2 != result->cols)) {
        printf("Error: Dimensions are invalid for mat_mul\n\n");
        exit(0);
    }

    double *data1 = mat1->data;
    double *data2 = mat2->data;
    double *data = result->data;
    size_t cols2_for_vec = cols2 / 4 * 4;
    unsigned int i;

    for (i = 0; i < rows1; i++) {
        
        unsigned int j, k;
        double val1, val2, dot_prod;
        __m256d sum_vec;
        for (j = 0; j < cols2_for_vec; j += 4) {

            sum_vec = _mm256_setzero_pd();
            for (k = 0; k < cols1; k++) {
                sum_vec = _mm256_add_pd(
                    sum_vec,
                    _mm256_mul_pd(
                        _mm256_set1_pd(mat_get(mat1, i, k)),
                        _mm256_loadu_pd(data2 + (k * cols2 + j))
                    )
                );
            }
            _mm256_storeu_pd(data + (i * cols2 + j), sum_vec);
        }
        for (j = cols2_for_vec; j < cols2; j++) {
            dot_prod = 0; 
            for (k = 0; k < cols1; k++) {
                val1 = mat_get(mat1, i, k);
                val2 = mat_get(mat2, k, j);
                dot_prod += val1 * val2;
            }
            mat_set(result, i, j, dot_prod);
        }
    }
}

void mat_mul_trans(matrix *result, matrix *mat1, matrix *mat2, bool t1, bool t2) {
    // Multiply matrices mat1 and mat2
    // Can transpose mat1 or mat2 before multiplying by setting t1 = true or t2 = true respectively 

    matrix *matA = mat1; 
    matrix *matB = mat2; 

    if (t1) {
        matA = zero_mat(mat1->cols, mat1->rows);
        transpose(matA, mat1);
    } 
    if (t2) {
        matB = zero_mat(mat2->cols, mat2->rows);
        transpose(matB, mat2);
    }

    mat_mul(result, matA, matB);

    if (t1) {
        free_mat(matA);
    }
    if (t2) {
        free_mat(matB);
    }
    
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

void mat_get_col(matrix *result, matrix *mat, int idx) {
    // Get column idx of matrix mat 

    size_t rows = mat->rows;
    size_t cols = mat->cols;
    unsigned int i;

    if ((result->rows != rows) || (result->cols != 1)) {
        printf("Error: Result dimensions invalid for mat_get_col");
    }

    for (i = 0; i < rows; i++) {
        mat_set(result, i, 0, mat_get(mat, i, idx));
    }
}

void mat_get_row(matrix *result, matrix *mat, int idx) {
    // Get row idx of matrix mat 

    size_t rows = mat->rows;
    size_t cols = mat->cols;
    unsigned int j;

    if ((result->cols != cols) || (result->rows != 1)) {
        printf("Error: Result dimensions invalid for mat_get_col");
    }

    for ( j = 0; j < cols; j++) {
        mat_set(result, 0, j, mat_get(mat, idx, j));
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
