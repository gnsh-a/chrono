#ifndef CHSOLVERCUDSS_H
#define CHSOLVERCUDSS_H

#include "chrono_cudss/ChApiCuDSS.h"
#include "chrono/solver/ChDirectSolverLS.h"
#include "chrono/solver/ChDirectSolverLScomplex.h"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cudss.h>

namespace chrono {

class ChApiCuDSS ChSolverCuDSS : public ChDirectSolverLS {
  public:
    ChSolverCuDSS(int device_id = 0);

    ~ChSolverCuDSS();

    virtual Type GetType() const override { return Type::CUSTOM; }

    cudssHandle_t GetCuDSSHandle() { return m_handle; }

  private:
    virtual bool FactorizeMatrix(bool analyze) override;
    virtual bool SolveSystem() override;
    virtual void PrintErrorMessage() override;

    // cuDSS objects
    cudssHandle_t m_handle = nullptr;  // cuDSS library handle
    cudssConfig_t m_config = nullptr;  // solver configuration
    cudssData_t   m_data   = nullptr;  // symbolic/numeric factor storage
    cudssMatrix_t m_mat_A  = nullptr;  // CSR matrix descriptor
    cudssMatrix_t m_mat_x  = nullptr;  // dense solution vector descriptor
    cudssMatrix_t m_mat_b  = nullptr;  // dense rhs vector descriptor

    // Device (GPU) buffers
    int*    d_row_ptr = nullptr;
    int*    d_col_ind = nullptr;
    double* d_values  = nullptr;
    double* d_x       = nullptr;
    double* d_b       = nullptr;

    int m_n   = 0;
    int m_nnz = 0;

    int           m_device      = 0;
    cudssStatus_t m_last_status = CUDSS_STATUS_SUCCESS;
};

class ChApiCuDSS ChSolverComplexCuDSS : public ChDirectSolverLScomplex {
  public:
    ChSolverComplexCuDSS(int device_id = 0);

    ~ChSolverComplexCuDSS();

    /// Get the underlying cuDSS library handle.
    cudssHandle_t GetCuDSSHandle() { return m_handle; }

  private:
    virtual bool FactorizeMatrix() override;
    virtual bool SolveSystem(const ChVectorDynamic<std::complex<double>>& b) override;
    virtual void PrintErrorMessage() override;

    // cuDSS objects
    cudssHandle_t m_handle = nullptr;
    cudssConfig_t m_config = nullptr;
    cudssData_t   m_data   = nullptr;
    cudssMatrix_t m_mat_A  = nullptr;
    cudssMatrix_t m_mat_x  = nullptr;
    cudssMatrix_t m_mat_b  = nullptr;

    // Device (GPU) buffers
    int*             d_row_ptr = nullptr;
    int*             d_col_ind = nullptr;
    cuDoubleComplex* d_values  = nullptr;
    cuDoubleComplex* d_x       = nullptr;
    cuDoubleComplex* d_b       = nullptr;

    int m_n   = 0;
    int m_nnz = 0;

    int           m_device      = 0;
    cudssStatus_t m_last_status = CUDSS_STATUS_SUCCESS;
};

}

#endif