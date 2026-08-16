#ifndef SYMNMF_H
#define SYMNMF_H

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

// Function declarations
vector* parse_point(char* line, int dimension);
int count_dimensions(char* line);
char* read_line_dynamic(FILE* file);

double** allocate_matrix(int rows, int cols);
void free_matrix(double** mat, int rows);
void matmul(double** A, double** B, double** result, int n, int m, int p);
void transpose(double** A, double** AT, int rows, int cols);
double frobenius_diff(double** H1, double** H2, int n, int k);
double random_double(double min, double max);

double** compute_similarity_matrix(vector** points, int n, int dimension);
double* compute_degree_matrix(double** A, int n);
double** compute_normalized_matrix(double** A, double* D, int n);
double** symnmf(double** W, double** H_init, int n, int k);

void print_matrix(double** matrix, int rows, int cols);
void print_vector_as_diagonal(double* diag, int n);
void cleanup_points(vector** points, int num_points);

#endif // SYMNMF_H
