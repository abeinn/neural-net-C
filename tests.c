#include "math_utils.c"
#include <time.h>

#define MIN_DIM 1
#define MAX_DIM 100
#define NUM_TESTS 100

size_t rand_dim() {
    return rand() % (MAX_DIM - MIN_DIM + 1) + MIN_DIM;
}

bool test_mat_lin_combo() {
    size_t rows = rand_dim();
    size_t cols = rand_dim();

    matrix *mat1 = rand_mat(rows, cols);
    matrix *mat2 = rand_mat(rows, cols);
    matrix *result = zero_mat(rows, cols);
    matrix *true_result = zero_mat(rows, cols);
    double c1 = rand_weight();
    double c2 = rand_weight();

    mat_lin_combo(result, mat1, mat2, c1, c2);

    size_t length = rows * cols;
    double *data1 = mat1->data;
    double *data2 = mat2->data;
    double *data = true_result->data;
    unsigned int i;

    for (i = 0; i < length; i++) {
        data[i] = c1 * data1[i] + c2 * data2[i];
    }

    bool output = mat_is_equal(result, true_result);

    free_mat(mat1);
    free_mat(mat2);
    free_mat(result);
    free_mat(true_result);

    return output; 
}

bool test_mat_vec_add() {
    size_t rows = rand_dim();
    size_t cols = rand_dim();

    matrix *mat = rand_mat(rows, cols);
    matrix *vec = rand_mat(rows, 1);
    matrix *result = zero_mat(rows, cols);
    matrix *true_result = zero_mat(rows, cols);

    mat_vec_add(result, mat, vec);

    unsigned int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            mat_set(true_result, i, j, mat_get(mat, i, j) + mat_get(vec, i, 0));
        }
    }

    bool output = mat_is_equal(result, true_result);

    free_mat(mat);
    free_mat(vec);
    free_mat(result);
    free_mat(true_result);

    return output;
}

bool test_mat_mul_trans() {
    
    size_t rows1 = rand_dim();
    size_t cols1 = rand_dim();
    size_t rows2 = cols1;
    size_t cols2 = rand_dim();
    bool t1 = false;
    bool t2 = false;

    matrix *mat1 = rand_mat(rows1, cols1);
    matrix *mat2 = rand_mat(rows2, cols2);
    matrix *result = zero_mat(rows1, cols2);
    matrix *true_result = zero_mat(rows1, cols2);

    mat_mul_trans(result, mat1, mat2, t1, t2);

    unsigned int i, j, k;
    double val1, val2, dot_prod;
    for (i = 0; i < rows1; i++) {
        for (j = 0; j < cols2; j++) {
            dot_prod = 0; 
            for (k = 0; k < cols1; k++) {
                val1 = t1 ? mat_get(mat1, k, i) : mat_get(mat1, i, k);
                val2 = t2 ? mat_get(mat2, j, k) : mat_get(mat2, k, j);
                dot_prod += val1 * val2;
            }
            mat_set(true_result, i, j, dot_prod);
        }
    }

    bool output = mat_is_equal(result, true_result);

    free_mat(mat1);
    free_mat(mat2);
    free_mat(result);
    free_mat(true_result);

    return output;
}

void run_tests(bool (*test_func)(), char *func_name) {

    printf("Testing %s\n", func_name);
    bool result = true; 
    unsigned int i;
    clock_t start = clock();

    for (i = 0; i < NUM_TESTS; i++) {
        if (test_func() == false) {
            result = false; 
            break; 
        }
    }

    if (result == true) {
        printf("All tests passed!\n");
    } else {
        printf("TEST FAILED\n");
    }
    
    clock_t end = clock();
    printf("Time taken: %Lf s\n\n", (long double)(end - start) / CLOCKS_PER_SEC);
}

int main(void) {
    
    // run_tests(test_mat_lin_combo, "mat_lin_combo");
    // run_tests(test_mat_vec_add, "mat_vec_add");
    run_tests(test_mat_mul_trans, "mat_mul_trans");
    

}