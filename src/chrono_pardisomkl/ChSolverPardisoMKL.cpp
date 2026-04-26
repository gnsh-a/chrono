// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Dario Mangoni, Radu Serban
// =============================================================================

#include "chrono_pardisomkl/ChSolverPardisoMKL.h"
#include "chrono/utils/ChOpenMP.h"

#include <iostream>

namespace chrono {

ChSolverPardisoMKL::ChSolverPardisoMKL(unsigned int num_threads) {
    mkl_set_num_threads(num_threads);
}

bool ChSolverPardisoMKL::FactorizeMatrix(bool analyze) {
    if (analyze) {
        if (verbose)
            std::cout << "  PARDISO Analysis Phase... " << std::flush;
        m_timer_analyze.reset();
        m_timer_analyze.start();
        m_engine.analyzePattern(m_mat);
        m_timer_analyze.stop();
        if (verbose)
            std::cout << "Done in " << m_timer_analyze.GetTimeSeconds() * 1000.0 << " ms." << std::endl;

        auto err_A = m_engine.info();
        if (err_A != Eigen::Success) {
            std::cerr << "Pardiso ANALYZE failed with error code " << ComputationInfoString(err_A) << std::endl;
            return false;
        }
    }

    if (verbose)
        std::cout << "  PARDISO Factorization Phase... " << std::flush;
    m_timer_factorize.reset();
    m_timer_factorize.start();
    m_engine.factorize(m_mat);
    m_timer_factorize.stop();
    if (verbose)
        std::cout << "Done in " << m_timer_factorize.GetTimeSeconds() * 1000.0 << " ms." << std::endl;

    auto err_F = m_engine.info();
    if (err_F != Eigen::Success) {
        std::cerr << "Pardiso FACTORIZE failed with error code " << ComputationInfoString(err_F) << std::endl;
        return false;
    }

    return true;
}

bool ChSolverPardisoMKL::SolveSystem() {
    if (verbose)
        std::cout << "  PARDISO Solve Phase... " << std::flush;
    m_timer_solve.reset();
    m_timer_solve.start();
    m_sol = m_engine.solve(m_rhs);
    m_timer_solve.stop();
    if (verbose)
        std::cout << "Done in " << m_timer_solve.GetTimeSeconds() * 1000.0 << " ms." << std::endl;

    return (m_engine.info() == Eigen::Success);
}

void ChSolverPardisoMKL::PrintErrorMessage() {
    // There are only three possible return codes (see manageErrorCode in Eigen's PardisoSupport.h)
    switch (m_engine.info()) {
        case Eigen::Success:
            std::cout << "PardisoMKL: computation was successful" << std::endl;
            break;
        case Eigen::NumericalIssue:
            std::cout << "PardisoMKL: provided data did not satisfy the prerequisites" << std::endl;
            break;
        case Eigen::InvalidInput:
            std::cout << "PardisoMKL: inputs are invalid, or the algorithm has been improperly called" << std::endl;
            break;
        case Eigen::NoConvergence:
            // Not a possible error for Pardiso
            break;
    }
}

//----------------------------------------------------------------------------------

ChSolverComplexPardisoMKL::ChSolverComplexPardisoMKL(unsigned int num_threads) {
    mkl_set_num_threads(num_threads);
}

bool ChSolverComplexPardisoMKL::FactorizeMatrix() {
    m_engine.compute(m_mat);
    return (m_engine.info() == Eigen::Success);
}

bool ChSolverComplexPardisoMKL::SolveSystem(const ChVectorDynamic<std::complex<double>>& b) {
    m_sol = m_engine.solve(b);
    return (m_engine.info() == Eigen::Success);
}

void ChSolverComplexPardisoMKL::PrintErrorMessage() {
    // There are only three possible return codes (see manageErrorCode in Eigen's PardisoSupport.h)
    switch (m_engine.info()) {
        case Eigen::Success:
            std::cout << "PardisoMKL: computation was successful" << std::endl;
            break;
        case Eigen::NumericalIssue:
            std::cout << "PardisoMKL: provided data did not satisfy the prerequisites" << std::endl;
            break;
        case Eigen::InvalidInput:
            std::cout << "PardisoMKL: inputs are invalid, or the algorithm has been improperly called" << std::endl;
            break;
        case Eigen::NoConvergence:
            // Not a possible error for Pardiso
            break;
    }
}

}  // end of namespace chrono
