// Enhanced symnmfmodule.c with all required functions
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

// Helper function to convert points from Python to C
static vector** convert_points_from_python(PyObject* points_list, int* n, int* dimension) {
    *n = PyList_Size(points_list);
    if (*n == 0) return NULL;
    
    PyObject* first_point = PyList_GetItem(points_list, 0);
    *dimension = PyList_Size(first_point);
    
    vector** points = malloc(*n * sizeof(vector*));
    if (!points) return NULL;
    
    for (int i = 0; i < *n; i++) {
        PyObject* point_list = PyList_GetItem(points_list, i);
        points[i] = malloc(sizeof(vector));
        if (!points[i]) {
            cleanup_points(points, i);
            return NULL;
        }
        
        points[i]->cord = malloc(*dimension * sizeof(double));
        points[i]->dimension = *dimension;
        if (!points[i]->cord) {
            cleanup_points(points, i + 1);
            return NULL;
        }
        
        for (int j = 0; j < *dimension; j++) {
            PyObject* coord = PyList_GetItem(point_list, j);
            points[i]->cord[j] = PyFloat_AsDouble(coord);
        }
    }
    
    return points;
}

// Helper function to convert matrix from Python to C
static double** convert_matrix_from_python(PyObject* matrix_list, int rows, int cols) {
    double** matrix = allocate_matrix(rows, cols);
    if (!matrix) return NULL;
    
    for (int i = 0; i < rows; i++) {
        PyObject* row = PyList_GetItem(matrix_list, i);
        for (int j = 0; j < cols; j++) {
            PyObject* val = PyList_GetItem(row, j);
            matrix[i][j] = PyFloat_AsDouble(val);
        }
    }
    
    return matrix;
}

// Helper function to convert matrix from C to Python
static PyObject* matrix_to_python(double** matrix, int rows, int cols) {
    PyObject* result = PyList_New(rows);
    if (!result) return NULL;
    
    for (int i = 0; i < rows; i++) {
        PyObject* row = PyList_New(cols);
        if (!row) {
            Py_DECREF(result);
            return NULL;
        }
        
        for (int j = 0; j < cols; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(matrix[i][j]));
        }
        PyList_SetItem(result, i, row);
    }
    
    return result;
}

// Helper function to convert vector to Python list
static PyObject* vector_to_python(double* vector, int size) {
    PyObject* result = PyList_New(size);
    if (!result) return NULL;
    
    for (int i = 0; i < size; i++) {
        PyList_SetItem(result, i, PyFloat_FromDouble(vector[i]));
    }
    
    return result;
}

static PyObject* py_sym(PyObject* self, PyObject* args) {
    PyObject* points_list;
    
    if (!PyArg_ParseTuple(args, "O", &points_list)) {
        return NULL;
    }
    
    if (!PyList_Check(points_list)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list of points");
        return NULL;
    }
    
    int n, dimension;
    vector** points = convert_points_from_python(points_list, &n, &dimension);
    if (!points) {
        PyErr_SetString(PyExc_MemoryError, "Failed to convert points");
        return NULL;
    }
    
    double** A = compute_similarity_matrix(points, n, dimension);
    cleanup_points(points, n);
    
    if (!A) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to compute similarity matrix");
        return NULL;
    }
    
    PyObject* result = matrix_to_python(A, n, n);
    free_matrix(A, n);
    
    return result;
}

static PyObject* py_ddg(PyObject* self, PyObject* args) {
    PyObject* points_list;
    
    if (!PyArg_ParseTuple(args, "O", &points_list)) {
        return NULL;
    }
    
    if (!PyList_Check(points_list)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list of points");
        return NULL;
    }
    
    int n, dimension;
    vector** points = convert_points_from_python(points_list, &n, &dimension);
    if (!points) {
        PyErr_SetString(PyExc_MemoryError, "Failed to convert points");
        return NULL;
    }
    
    double** A = compute_similarity_matrix(points, n, dimension);
    cleanup_points(points, n);
    
    if (!A) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to compute similarity matrix");
        return NULL;
    }
    
    double* D = compute_degree_matrix(A, n);
    free_matrix(A, n);
    
    if (!D) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to compute degree matrix");
        return NULL;
    }
    
    PyObject* result = vector_to_python(D, n);
    free(D);
    
    return result;
}

static PyObject* py_norm(PyObject* self, PyObject* args) {
    PyObject* points_list;
    
    if (!PyArg_ParseTuple(args, "O", &points_list)) {
        return NULL;
    }
    
    if (!PyList_Check(points_list)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list of points");
        return NULL;
    }
    
    int n, dimension;
    vector** points = convert_points_from_python(points_list, &n, &dimension);
    if (!points) {
        PyErr_SetString(PyExc_MemoryError, "Failed to convert points");
        return NULL;
    }
    
    double** A = compute_similarity_matrix(points, n, dimension);
    cleanup_points(points, n);
    
    if (!A) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to compute similarity matrix");
        return NULL;
    }
    
    double* D = compute_degree_matrix(A, n);
    if (!D) {
        free_matrix(A, n);
        PyErr_SetString(PyExc_RuntimeError, "Failed to compute degree matrix");
        return NULL;
    }
    
    double** W = compute_normalized_matrix(A, D, n);
    free_matrix(A, n);
    free(D);
    
    if (!W) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to compute normalized matrix");
        return NULL;
    }
    
    PyObject* result = matrix_to_python(W, n, n);
    free_matrix(W, n);
    
    return result;
}

static PyObject* py_symnmf(PyObject* self, PyObject* args) {
    PyObject* W_list;
    PyObject* H_init_list;
    int k;

    if (!PyArg_ParseTuple(args, "OOi", &W_list, &H_init_list, &k)) {
        return NULL;
    }

    if (!PyList_Check(W_list) || !PyList_Check(H_init_list)) {
        PyErr_SetString(PyExc_TypeError, "Arguments must be lists");
        return NULL;
    }

    int n = PyList_Size(W_list);
    if (n == 0) {
        PyErr_SetString(PyExc_ValueError, "Input matrix cannot be empty");
        return NULL;
    }

    if (k <= 0 || k >= n) {
        PyErr_SetString(PyExc_ValueError, "k must be between 1 and n-1");
        return NULL;
    }

    // Convert W from Python to C
    double** W = convert_matrix_from_python(W_list, n, n);
    if (!W) {
        PyErr_SetString(PyExc_MemoryError, "Failed to convert W matrix");
        return NULL;
    }

    // Convert H_init from Python to C
    double** H_init = convert_matrix_from_python(H_init_list, n, k);
    if (!H_init) {
        free_matrix(W, n);
        PyErr_SetString(PyExc_MemoryError, "Failed to convert H_init matrix");
        return NULL;
    }

    // Run the symnmf algorithm with initialized H
    double** H_result = symnmf(W, H_init, n, k);
    free_matrix(W, n);
    free_matrix(H_init, n);

    if (!H_result) {
        PyErr_SetString(PyExc_RuntimeError, "symnmf computation failed");
        return NULL;
    }

    PyObject* result = matrix_to_python(H_result, n, k);
    free_matrix(H_result, n);
    
    return result;
}

// Module methods
static PyMethodDef SymnmfMethods[] = {
    {"sym", py_sym, METH_VARARGS, "Compute similarity matrix"},
    {"ddg", py_ddg, METH_VARARGS, "Compute degree matrix diagonal"},
    {"norm", py_norm, METH_VARARGS, "Compute normalized similarity matrix"},
    {"symnmf", py_symnmf, METH_VARARGS, "Perform Symmetric Non-negative Matrix Factorization"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf",
    "Symmetric Non-negative Matrix Factorization module",
    -1,
    SymnmfMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmfmodule);
}