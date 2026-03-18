#include "chrono_cudss/ChSolverCuDSS.h"

#include <iostream>

namespace chrono {


#define CUDSS_CHECK(call)                                                           
    do {                                                                            
        cudssStatus_t _s = (call);                                                  
        if (_s != CUDSS_STATUS_SUCCESS) {                                           
            m_last_status = _s;                                                     
            return false;                                                           
        }                                                                           
    } while (0)

// ChSolverCuDSS
ChSolverCuDSS::ChSolverCuDSS(int device_id) : m_device(device_id) {
    cudaSetDevice(m_device);
    cudssCreate(&m_handle);
    cudssConfigCreate(&m_config);
    cudssDataCreate(m_handle, &m_data);
}

ChSolverCuDSS::~ChSolverCuDSS() {
    if (m_mat_b) cudssMatrixDestroy(m_mat_b);
    if (m_mat_x) cudssMatrixDestroy(m_mat_x);
    if (m_mat_A) cudssMatrixDestroy(m_mat_A);

    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_values);
    cudaFree(d_col_ind);
    cudaFree(d_row_ptr);

    cudssDataDestroy(m_handle, m_data);
    cudssConfigDestroy(m_config);
    cudssDestroy(m_handle);
}

bool ChSolverCuDSS::FactorizeMatrix(bool analyze) {
    cudaSetDevice(m_device);

    const int n   = (int)m_mat.rows();
    const int nnz = (int)m_mat.nonZeros();

    const bool realloc = (n != m_n || nnz != m_nnz);
    if (realloc) {
        cudaFree(d_row_ptr); cudaFree(d_col_ind); cudaFree(d_values);
        cudaFree(d_x);       cudaFree(d_b);

        cudaMalloc((void**)&d_row_ptr, (n + 1) * sizeof(int));
        cudaMalloc((void**)&d_col_ind, nnz     * sizeof(int));
        cudaMalloc((void**)&d_values,  nnz     * sizeof(double));
        cudaMalloc((void**)&d_x,       n       * sizeof(double));
        cudaMalloc((void**)&d_b,       n       * sizeof(double));

        m_n   = n;
        m_nnz = nnz;
    }

    cudaMemcpy(d_row_ptr, m_mat.outerIndexPtr(), (n + 1) * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, m_mat.innerIndexPtr(), nnz     * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_values,  m_mat.valuePtr(),      nnz     * sizeof(double), cudaMemcpyHostToDevice);

    if (analyze || realloc) {
        if (m_mat_A) { cudssMatrixDestroy(m_mat_A); m_mat_A = nullptr; }

        CUDSS_CHECK(cudssMatrixCreateCsr(
            &m_mat_A,
            n, n, nnz,
            d_row_ptr, d_col_ind, d_values,
            CUDSS_INDEX_32I, CUDSS_INDEX_32I,
            CUDSS_BASE_ZERO,
            CUDA_R_64F,
            CUDSS_MTYPE_GENERAL,
            CUDSS_MVIEW_FULL));

        CUDSS_CHECK(cudssExecute(m_handle, CUDSS_PHASE_ANALYSIS,
                                 m_config, m_data, m_mat_A, nullptr, nullptr));

        CUDSS_CHECK(cudssExecute(m_handle, CUDSS_PHASE_FACTORIZATION,
                                 m_config, m_data, m_mat_A, nullptr, nullptr));
    } else {
        CUDSS_CHECK(cudssExecute(m_handle, CUDSS_PHASE_REFACTORIZATION,
                                 m_config, m_data, m_mat_A, nullptr, nullptr));
    }

    return true;
}

bool ChSolverCuDSS::SolveSystem() {
    cudaSetDevice(m_device);

    const int n = m_n;
    cudaMemcpy(d_b, m_rhs.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    // Recreate dense vector descriptors (hopefully avoids stale pointer issues[need to test])
    if (m_mat_x) { cudssMatrixDestroy(m_mat_x); m_mat_x = nullptr; }
    if (m_mat_b) { cudssMatrixDestroy(m_mat_b); m_mat_b = nullptr; }

    CUDSS_CHECK(cudssMatrixCreateDn(&m_mat_x, n, 1, n, d_x, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(&m_mat_b, n, 1, n, d_b, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));

    CUDSS_CHECK(cudssExecute(m_handle, CUDSS_PHASE_SOLVE,
                             m_config, m_data, m_mat_A, m_mat_x, m_mat_b));

    // Copy solution
    cudaMemcpy(m_sol.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost);

    return true;
}

void ChSolverCuDSS::PrintErrorMessage() {
    std::cout << "ChSolverCuDSS: cuDSS error status " << (int)m_last_status << std::endl;
}

ChSolverComplexCuDSS::ChSolverComplexCuDSS(int device_id) : m_device(device_id) {
    cudaSetDevice(m_device);
    cudssCreate(&m_handle);
    cudssConfigCreate(&m_config);
    cudssDataCreate(m_handle, &m_data);
}

ChSolverComplexCuDSS::~ChSolverComplexCuDSS() {
    if (m_mat_b) cudssMatrixDestroy(m_mat_b);
    if (m_mat_x) cudssMatrixDestroy(m_mat_x);
    if (m_mat_A) cudssMatrixDestroy(m_mat_A);

    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_values);
    cudaFree(d_col_ind);
    cudaFree(d_row_ptr);

    cudssDataDestroy(m_handle, m_data);
    cudssConfigDestroy(m_config);
    cudssDestroy(m_handle);
}

bool ChSolverComplexCuDSS::FactorizeMatrix() {
    cudaSetDevice(m_device);

    const int n   = (int)m_mat.rows();
    const int nnz = (int)m_mat.nonZeros();

    const bool realloc = (n != m_n || nnz != m_nnz);
    if (realloc) {
        cudaFree(d_row_ptr); cudaFree(d_col_ind); cudaFree(d_values);
        cudaFree(d_x);       cudaFree(d_b);

        cudaMalloc((void**)&d_row_ptr, (n + 1) * sizeof(int));
        cudaMalloc((void**)&d_col_ind, nnz     * sizeof(int));
        cudaMalloc((void**)&d_values,  nnz     * sizeof(cuDoubleComplex));
        cudaMalloc((void**)&d_x,       n       * sizeof(cuDoubleComplex));
        cudaMalloc((void**)&d_b,       n       * sizeof(cuDoubleComplex));

        m_n   = n;
        m_nnz = nnz;
    }

    cudaMemcpy(d_row_ptr, m_mat.outerIndexPtr(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, m_mat.innerIndexPtr(), nnz     * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_values,  m_mat.valuePtr(),      nnz     * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    if (realloc) {
        if (m_mat_A) { cudssMatrixDestroy(m_mat_A); m_mat_A = nullptr; }

        CUDSS_CHECK(cudssMatrixCreateCsr(
            &m_mat_A,
            n, n, nnz,
            d_row_ptr, d_col_ind, d_values,
            CUDSS_INDEX_32I, CUDSS_INDEX_32I,
            CUDSS_BASE_ZERO,
            CUDA_C_64F,
            CUDSS_MTYPE_GENERAL,
            CUDSS_MVIEW_FULL));

        CUDSS_CHECK(cudssExecute(m_handle, CUDSS_PHASE_ANALYSIS,
                                 m_config, m_data, m_mat_A, nullptr, nullptr));
    }

    CUDSS_CHECK(cudssExecute(m_handle, CUDSS_PHASE_FACTORIZATION,
                             m_config, m_data, m_mat_A, nullptr, nullptr));

    return true;
}

bool ChSolverComplexCuDSS::SolveSystem(const ChVectorDynamic<std::complex<double>>& b) {
    cudaSetDevice(m_device);

    const int n = m_n;

    cudaMemcpy(d_b, b.data(), n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    if (m_mat_x) { cudssMatrixDestroy(m_mat_x); m_mat_x = nullptr; }
    if (m_mat_b) { cudssMatrixDestroy(m_mat_b); m_mat_b = nullptr; }

    CUDSS_CHECK(cudssMatrixCreateDn(&m_mat_x, n, 1, n, d_x, CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(&m_mat_b, n, 1, n, d_b, CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR));

    CUDSS_CHECK(cudssExecute(m_handle, CUDSS_PHASE_SOLVE,
                             m_config, m_data, m_mat_A, m_mat_x, m_mat_b));

    cudaMemcpy(m_sol.data(), d_x, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    return true;
}

void ChSolverComplexCuDSS::PrintErrorMessage() {
    std::cout << "ChSolverComplexCuDSS: cuDSS error status " << (int)m_last_status << std::endl;
}

}