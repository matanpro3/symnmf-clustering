#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define BETA 0.5
#define EPSILON 1e-4
#define MAX_ITER 300

typedef struct vector {
    double* cord;   
    int dimension;
} vector;

/* Function declarations */
vector* parse_point(char* line, int dimension);
int count_dimensions(char* line);
char* read_line_dynamic(FILE* file);
double** allocate_matrix(int rows, int cols);
void free_matrix(double** mat, int rows);
void matmul(double** A, double** B, double** result, int n, int m, int p);
void transpose(double** A, double** AT, int rows, int cols);
void scale_matrix_elementwise(double** A, double* row_scale, double* col_scale, double** result, int n);
double frobenius_diff(double** H1, double** H2, int n, int k);
double** compute_similarity_matrix(vector** points, int n, int dimension);
double* compute_degree_matrix(double** A, int n);
double** compute_normalized_matrix(double** A, double* D, int n);
double** symnmf(double** W, double** H_init, int n, int k);
void print_matrix(double** matrix, int rows, int cols);
void print_vector_as_diagonal(double* diag, int n);
void cleanup_points(vector** points, int num_points);

char* read_line_dynamic(FILE* file) {
    int line_capacity = 10;
    int current_pos = 0;
    char* line_text = (char*)malloc(line_capacity * sizeof(char));
    if (line_text == NULL) {
        return NULL;
    }

    {
        int character; /* Declare here for C90 */
        while ((character = fgetc(file)) != EOF && character != '\n') {
            if (current_pos >= line_capacity - 1) {
                line_capacity *= 2;
                {
                    char* temp = (char*)realloc(line_text, line_capacity * sizeof(char));
                    if (temp == NULL) {
                        free(line_text);
                        return NULL;
                    }
                    line_text = temp;
                }
            }
            line_text[current_pos++] = (char)character;
        }

        if (character == EOF && current_pos == 0) {
            free(line_text);
            return NULL;
        }
    }

    line_text[current_pos] = '\0';
    return line_text;
}

int count_dimensions(char* line) {
    int count = 0;
    int i = 0;

    while (line[i] != '\0') {
        if (line[i] == ',') {
            count++;
        }
        i++;
    }

    return count + 1;
}

vector* parse_point(char* line, int dimension) {
    vector* point = (vector*)malloc(sizeof(vector));
    if (point == NULL) {
        return NULL;
    }

    point->cord = (double*)malloc(dimension * sizeof(double));
    if (point->cord == NULL) {
        free(point);
        return NULL;
    }

    point->dimension = dimension;

    {
        char* token = strtok(line, ",");
        int i = 0;
        while (token != NULL && i < dimension) {
            point->cord[i++] = atof(token);
            token = strtok(NULL, ",");
        }
    }

    return point;
}

double** allocate_matrix(int rows, int cols) {
    int i, j;
    double** mat = (double**)malloc(rows * sizeof(double*));
    if (!mat) return NULL;
    for (i = 0; i < rows; i++) {
        mat[i] = (double*)calloc(cols, sizeof(double));
        if (!mat[i]) {
            for (j = 0; j < i; j++) free(mat[j]);
            free(mat);
            return NULL;
        }
    }
    return mat;
}

void free_matrix(double** mat, int rows) {
    int i;
    if (!mat) return;
    for (i = 0; i < rows; i++)
        free(mat[i]);
    free(mat);
}

void matmul(double** A, double** B, double** result, int n, int m, int p) {
    int i, j, z;
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
            result[i][j] = 0;
            for (z = 0; z < m; z++) {
                result[i][j] += A[i][z] * B[z][j];
            }
        }
    }
}

void transpose(double** A, double** AT, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++)
            AT[j][i] = A[i][j];
}

void scale_matrix_elementwise(double** A, double* row_scale, double* col_scale, double** result, int n) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (row_scale[i] == 0.0 || col_scale[j] == 0.0) {
                result[i][j] = 0.0;
            } else {
                result[i][j] = A[i][j] * row_scale[i] * col_scale[j];
            }
        }
    }
}

double frobenius_diff(double** H1, double** H2, int n, int k) {
    int i, j;
    double sum = 0.0;
    for (i = 0; i < n; i++)
        for (j = 0; j < k; j++) {
            double diff = H1[i][j] - H2[i][j];
            sum += diff * diff;
        }
    return sum;
}

double** compute_similarity_matrix(vector** points, int n, int dimension) {
    int i, j, d;
    double** A = allocate_matrix(n, n);
    if (!A) return NULL;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                A[i][j] = 0.0;
            } else {
                double dist_sq = 0;
                for (d = 0; d < dimension; d++) {
                    double diff = points[i]->cord[d] - points[j]->cord[d];
                    dist_sq += diff * diff;
                }
                A[i][j] = exp(-dist_sq / 2.0);
            }
        }
    }
    return A;
}

double* compute_degree_matrix(double** A, int n) {
    int i, j;
    double* D = (double*)malloc(n * sizeof(double));
    if (D == NULL) {
        return NULL;
    }

    for (i = 0; i < n; i++) {
        double sum = 0.0;
        for (j = 0; j < n; j++) {
            sum += A[i][j];
        }
        D[i] = sum;
    }
    return D;
}

double** compute_normalized_matrix(double** A, double* D, int n) {
    int i;
    double* D_inv_sqrt;
    double** W = allocate_matrix(n, n);
    if (!W) return NULL;

    /* Create D^(-1/2) vector */
    D_inv_sqrt = (double*)malloc(n * sizeof(double));
    if (!D_inv_sqrt) {
        free_matrix(W, n);
        return NULL;
    }

    for (i = 0; i < n; i++) {
        D_inv_sqrt[i] = (D[i] == 0.0) ? 0.0 : 1.0 / sqrt(D[i]);
    }

    /* W = D^(-1/2) * A * D^(-1/2) */
    scale_matrix_elementwise(A, D_inv_sqrt, D_inv_sqrt, W, n);

    free(D_inv_sqrt);
    return W;
}

double** symnmf(double** W, double** H_init, int n, int k) {
    int iter = 0;
    double diff;
    int i, j;
    double** H = NULL;
    double** H_new = NULL;
    double** WH = NULL;
    double** HT = NULL;
    double** HTH = NULL;
    double** denom = NULL;
    double** temp = NULL;

    H = allocate_matrix(n, k);
    H_new = allocate_matrix(n, k);
    WH = allocate_matrix(n, k);
    HT = allocate_matrix(k, n);
    HTH = allocate_matrix(k, k);
    denom = allocate_matrix(n, k);

    if (!H || !H_new || !WH || !HT || !HTH || !denom) {
        free_matrix(H, n);
        free_matrix(H_new, n);
        free_matrix(WH, n);
        free_matrix(HT, k);
        free_matrix(HTH, k);
        free_matrix(denom, n);
        return NULL;
    }

    /* Initialize H with H_init */
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            H[i][j] = H_init[i][j];
        }
    }

    do {
        /* WH = W * H */
        matmul(W, H, WH, n, n, k);

        /* HT = H^T */
        transpose(H, HT, n, k);

        /* HTH = HT * H */
        matmul(HT, H, HTH, k, n, k);

        /* denom = H * HTH */
        matmul(H, HTH, denom, n, k, k);

        /* Update H_new */
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                if (denom[i][j] > 1e-10) {
                    H_new[i][j] = H[i][j] * ((1 - BETA) + BETA * (WH[i][j] / denom[i][j]));
                } else {
                    H_new[i][j] = H[i][j] * (1 - BETA);
                }

                /* Non-negativity constraint */
                if (H_new[i][j] < 0) {
                    H_new[i][j] = 1e-10;
                }
            }
        }

        /* Compute Frobenius norm diff */
        diff = frobenius_diff(H, H_new, n, k);

        /* Swap H and H_new */
        temp = H;
        H = H_new;
        H_new = temp;

        iter++;
    } while (diff > EPSILON && iter < MAX_ITER);

    /* Free all except final result H */
    free_matrix(H_new, n);
    free_matrix(WH, n);
    free_matrix(HT, k);
    free_matrix(HTH, k);
    free_matrix(denom, n);

    return H;
}

void print_matrix(double** matrix, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%.4f", matrix[i][j]);
            if (j < cols - 1) printf(",");
        }
        printf("\n");
    }
}

void print_vector_as_diagonal(double* diag, int n) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                printf("%.4f", diag[i]);
            } else {
                printf("0.0000");
            }
            if (j < n - 1) printf(",");
        }
        printf("\n");
    }
}

void cleanup_points(vector** points, int num_points) {
    int i;
    if (!points) return;
    for (i = 0; i < num_points; i++) {
        if (points[i]) {
            free(points[i]->cord);
            free(points[i]);
        }
    }
    free(points);
}

int main(int argc, char* argv[]) {
    char* goal;
    char* filename;
    vector** points;
    int num_points;
    int max_points;
    int dimension;
    char* line;
    double** W;

    if (argc == 3) {
        goal = argv[1];
        filename = argv[2];
    } else {
        printf("An Error Has Occurred\n");
        return 1;
    }

    if (strcmp(goal, "sym") != 0 && strcmp(goal, "ddg") != 0 &&
        strcmp(goal, "norm") != 0) {
        printf("An Error Has Occurred\n");
        return 1;
    }

    {
        FILE* input_file = fopen(filename, "r");
        if (!input_file) {
            printf("An Error Has Occurred\n");
            return 1;
        }

        points = NULL;
        num_points = 0;
        max_points = 10;
        dimension = 0;

        points = (vector**)malloc(max_points * sizeof(vector*));
        if (!points) {
            printf("An Error Has Occurred\n");
            fclose(input_file);
            return 1;
        }

        while ((line = read_line_dynamic(input_file)) != NULL) {
            if (strlen(line) == 0) {
                free(line);
                continue;
            }

            if (num_points == 0) {
                dimension = count_dimensions(line);
                if (dimension <= 0) {
                    printf("An Error Has Occurred\n");
                    free(line);
                    cleanup_points(points, num_points);
                    fclose(input_file);
                    return 1;
                }
            }

            if (num_points >= max_points) {
                vector** tmp;
                max_points *= 2;
                tmp = (vector**)realloc(points, max_points * sizeof(vector*));
                if (!tmp) {
                    printf("An Error Has Occurred\n");
                    free(line);
                    cleanup_points(points, num_points);
                    fclose(input_file);
                    return 1;
                }
                points = tmp;
            }

            points[num_points] = parse_point(line, dimension);
            free(line);
            if (!points[num_points]) {
                printf("An Error Has Occurred\n");
                cleanup_points(points, num_points);
                fclose(input_file);
                return 1;
            }

            num_points++;
        }

        fclose(input_file);

        if (num_points == 0) {
            printf("An Error Has Occurred\n");
            cleanup_points(points, num_points);
            return 1;
        }

        {
            double** A = compute_similarity_matrix(points, num_points, dimension);
            if (!A) {
                printf("An Error Has Occurred\n");
                cleanup_points(points, num_points);
                return 1;
            }

            if (strcmp(goal, "sym") == 0) {
                print_matrix(A, num_points, num_points);
            }
            else if (strcmp(goal, "ddg") == 0) {
                double* D = compute_degree_matrix(A, num_points);
                if (!D) {
                    printf("An Error Has Occurred\n");
                    free_matrix(A, num_points);
                    cleanup_points(points, num_points);
                    return 1;
                }
                print_vector_as_diagonal(D, num_points);
                free(D);
            }
            else if (strcmp(goal, "norm") == 0) {
                double* D = compute_degree_matrix(A, num_points);
                if (!D) {
                    printf("An Error Has Occurred\n");
                    free_matrix(A, num_points);
                    cleanup_points(points, num_points);
                    return 1;
                }
                W = compute_normalized_matrix(A, D, num_points);
                if (!W) {
                    printf("An Error Has Occurred\n");
                    free(D);
                    free_matrix(A, num_points);
                    cleanup_points(points, num_points);
                    return 1;
                }
                print_matrix(W, num_points, num_points);

                free_matrix(W, num_points);
                free(D);
            }

            free_matrix(A, num_points);
        }

        cleanup_points(points, num_points);
    }

    return 0;
}
