#include "math_utils.c"

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
    mat_lin_combo(true_result, mat1, mat2, c1, c2);

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

    for (i = 0; i < NUM_TESTS; i++) {
        if (test_func() == false) {
            result = false; 
            break; 
        }
    }

    if (result == true) {
        printf("All tests passed!\n\n");
    } else {
        printf("TEST FAILED\n\n");
    }
}

int main(void) {
    
    run_tests(test_mat_lin_combo, "mat_lin_combo");

}