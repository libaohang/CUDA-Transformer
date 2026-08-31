#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// Row-major host-side matrix helpers. All matrices are flat float arrays
// of size rows*cols, indexed as data[row * cols + col].

// ---------- Initialization ----------

inline void matrixRandomInit(std::vector<float>& mat, int rows, int cols,
                              float lo = -1.0f, float hi = 1.0f,
                              unsigned int seed = 42) {
    mat.resize(static_cast<size_t>(rows) * cols);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& v : mat) v = dist(gen);
}

inline void matrixZeroInit(std::vector<float>& mat, int rows, int cols) {
    mat.assign(static_cast<size_t>(rows) * cols, 0.0f);
}

inline void matrixIdentityInit(std::vector<float>& mat, int n) {
    matrixZeroInit(mat, n, n);
    for (int i = 0; i < n; ++i) mat[i * n + i] = 1.0f;
}

// ---------- Printing (only sane for small matrices) ----------

inline void matrixPrint(const std::vector<float>& mat, int rows, int cols,
                         const char* name = nullptr) {
    if (name) printf("%s (%dx%d):\n", name, rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            printf("%8.4f ", mat[r * cols + c]);
        }
        printf("\n");
    }
}

// ---------- Comparison ----------

struct MatrixCompareResult {
    bool passed;
    float max_abs_diff;
    float max_rel_diff;
    int first_mismatch_row;
    int first_mismatch_col;
};

// Compares two matrices with combined absolute/relative tolerance, since
// pure absolute tolerance fails on large values and pure relative tolerance
// blows up near zero. Mirrors the logic behind numpy's allclose.
inline MatrixCompareResult matrixCompare(const std::vector<float>& a,
                                          const std::vector<float>& b,
                                          int rows, int cols,
                                          float atol = 1e-3f,
                                          float rtol = 1e-3f) {
    MatrixCompareResult result{true, 0.0f, 0.0f, -1, -1};

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            size_t idx = static_cast<size_t>(r) * cols + c;
            float diff = std::fabs(a[idx] - b[idx]);
            float threshold = atol + rtol * std::fabs(b[idx]);

            if (diff > result.max_abs_diff) result.max_abs_diff = diff;
            float rel = diff / (std::fabs(b[idx]) + 1e-8f);
            if (rel > result.max_rel_diff) result.max_rel_diff = rel;

            if (diff > threshold && result.passed) {
                result.passed = false;
                result.first_mismatch_row = r;
                result.first_mismatch_col = c;
            }
        }
    }
    return result;
}

inline void matrixCompareReport(const MatrixCompareResult& res,
                                 const char* label = "Comparison") {
    printf("%s: %s (max_abs_diff=%.6e, max_rel_diff=%.6e)\n",
           label, res.passed ? "PASSED" : "FAILED",
           res.max_abs_diff, res.max_rel_diff);
    if (!res.passed) {
        printf("  First mismatch at (%d, %d)\n",
               res.first_mismatch_row, res.first_mismatch_col);
    }
}