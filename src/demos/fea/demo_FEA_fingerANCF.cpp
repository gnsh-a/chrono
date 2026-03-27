// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2024 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Ganesh Arivoli <arivoli@wisc.edu>, Json Zhou <zzhou292@wisc.edu>
//
// ANCF tendon-driven finger demo
//
// A 3-joint finger (MCP, PIP, DIP) with ANCF beam phalanges connected by
// revolute joints, actuated by an ANCF cable tendon routed along the palmar
// side through sliding pulleys. Pulling the tendon causes the finger to curl.
//
// =============================================================================

#include "chrono/physics/ChSystemSMC.h"
#include "chrono/physics/ChBody.h"
#include "chrono/physics/ChLinkRevolute.h"

#include "chrono/fea/ChElementBeamANCF_3243.h"
#include "chrono/fea/ChElementCableANCF.h"
#include "chrono/fea/ChBeamSectionCable.h"
#include "chrono/fea/ChMesh.h"
#include "chrono/fea/ChLinkNodeFrame.h"

#include "chrono/assets/ChVisualShapeFEA.h"
#include "chrono/assets/ChVisualShapeBox.h"
#include "chrono/assets/ChVisualShapeSphere.h"
#include "chrono/assets/ChVisualShapeCylinder.h"

#include "chrono_pardisomkl/ChSolverPardisoMKL.h"

#ifdef CHRONO_VSG
    #define USE_VISUALIZATION
    #include "FEAvisualization.h"
#endif

using namespace chrono;
using namespace chrono::fea;

int main(int argc, char* argv[]) {
    // Pass --no-vis to run headless
    bool use_vis = true;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--no-vis")
            use_vis = false;
    }

    std::cout << "ANCF Tendon-Driven Finger Demo" << std::endl;
    std::cout << "Chrono version: " << CHRONO_VERSION << std::endl;

    // -------------------------------------------------------------------------
    // Finger geometry
    // -------------------------------------------------------------------------
    double L1 = 0.045;   // proximal phalanx length (m) - Buryanov & Kotiuk 2010
    double L2 = 0.028;   // middle phalanx length (m)
    double L3 = 0.020;   // distal phalanx length (m)
    double L_total = L1 + L2 + L3;

    // Per-phalanx cross-sections (width x thickness in m) - literature midshaft values
    double pp_width = 0.011;  double pp_thick = 0.009;   // proximal: 11 x 9 mm
    double mp_width = 0.010;  double mp_thick = 0.008;   // middle:   10 x 8 mm
    double dp_width = 0.007;  double dp_thick = 0.006;   // distal:    7 x 6 mm

    double palmar_offset = 0.004;   // tendon offset from bone centroid (m) - An et al. 1983
    double tendon_diameter = 0.004; // tendon equiv diameter (m) - Manske & Lesker 1983
    double L_pull = 0.02;           // free cable length before MCP (m)

    // -------------------------------------------------------------------------
    // Material properties
    // -------------------------------------------------------------------------
    double bone_rho = 1800.0;   // bone density (kg/m^3)
    double bone_E = 1e6;        // bone Young's modulus (Pa) - reduced for visible deformation
    double bone_nu = 0.3;       // bone Poisson's ratio
    double bone_k1 = 10.0 * (1.0 + bone_nu) / (12.0 + 11.0 * bone_nu);
    double bone_k2 = bone_k1;

    double tendon_E = 1.2e9;   // tendon Young's modulus (Pa)
    double tendon_damp = 0.01;  // Rayleigh damping

    // Actuation timing
    double F_pull_max = -20.0;  // max pull force (N), in -X direction
    double t_ramp = 3.0;       // ramp-up time (s)
    double t_hold = 3.0;       // hold at max force (s)
    double t_release = 3.0;    // ramp-down time (s)
    double t_settle = 3.0;     // settle back to straight (s)
    double t_end = t_ramp + t_hold + t_release + t_settle;

    // -------------------------------------------------------------------------
    // Create system
    // -------------------------------------------------------------------------
    ChSystemSMC sys;
    sys.SetGravitationalAcceleration(ChVector3d(0, 0, 0));

    // Solver
    auto solver = chrono_types::make_shared<ChSolverPardisoMKL>();
    solver->UseSparsityPatternLearner(true);
    solver->LockSparsityPattern(true);
    solver->SetVerbose(false);
    sys.SetSolver(solver);

    // Timestepper
    sys.SetTimestepperType(ChTimestepper::Type::EULER_IMPLICIT_LINEARIZED);

    // -------------------------------------------------------------------------
    // Create meshes (one per phalanx + one for cable, for separate colors)
    // -------------------------------------------------------------------------
    auto mesh_prox = chrono_types::make_shared<ChMesh>();
    auto mesh_mid = chrono_types::make_shared<ChMesh>();
    auto mesh_dist = chrono_types::make_shared<ChMesh>();
    auto mesh_cable = chrono_types::make_shared<ChMesh>();
    mesh_prox->SetAutomaticGravity(false);
    mesh_mid->SetAutomaticGravity(false);
    mesh_dist->SetAutomaticGravity(false);
    mesh_cable->SetAutomaticGravity(false);
    sys.Add(mesh_prox);
    sys.Add(mesh_mid);
    sys.Add(mesh_dist);
    sys.Add(mesh_cable);

    // -------------------------------------------------------------------------
    // Bone material
    // -------------------------------------------------------------------------
    auto bone_material = chrono_types::make_shared<ChMaterialBeamANCF>(bone_rho, bone_E, bone_nu, bone_k1, bone_k2);

    // -------------------------------------------------------------------------
    // Adapter rigid bodies (near-zero mass, serve as joint interfaces)
    // -------------------------------------------------------------------------
    double adapter_mass = 0.001;
    ChVector3d adapter_inertia(1e-9, 1e-9, 1e-9);

    // Ground (metacarpal, fixed)
    auto ground = chrono_types::make_shared<ChBody>();
    ground->SetFixed(true);
    ground->SetPos(ChVector3d(0, 0, 0));
    sys.AddBody(ground);
    auto ground_vis = chrono_types::make_shared<ChVisualShapeBox>(0.01, 0.015, 0.015);
    ground->AddVisualShape(ground_vis, ChFrame<>(ChVector3d(-0.005, 0, 0)));

    // body_A: distal end of proximal phalanx / proximal end of middle phalanx
    auto body_A = chrono_types::make_shared<ChBody>();
    body_A->SetMass(adapter_mass);
    body_A->SetInertiaXX(adapter_inertia);
    body_A->SetPos(ChVector3d(L1, 0, 0));
    sys.AddBody(body_A);
    body_A->AddVisualShape(chrono_types::make_shared<ChVisualShapeSphere>(0.001));

    // body_B: distal end of middle phalanx / proximal end of distal phalanx
    auto body_B = chrono_types::make_shared<ChBody>();
    body_B->SetMass(adapter_mass);
    body_B->SetInertiaXX(adapter_inertia);
    body_B->SetPos(ChVector3d(L1 + L2, 0, 0));
    sys.AddBody(body_B);
    body_B->AddVisualShape(chrono_types::make_shared<ChVisualShapeSphere>(0.001));

    // body_C: tip of distal phalanx
    auto body_C = chrono_types::make_shared<ChBody>();
    body_C->SetMass(adapter_mass);
    body_C->SetInertiaXX(adapter_inertia);
    body_C->SetPos(ChVector3d(L1 + L2 + L3, 0, 0));
    sys.AddBody(body_C);
    auto jointC_vis = chrono_types::make_shared<ChVisualShapeSphere>(0.004);
    jointC_vis->SetColor(ChColor(0.75f, 0.75f, 0.75f));
    body_C->AddVisualShape(jointC_vis);

    // -------------------------------------------------------------------------
    // Revolute joints (Z-axis rotation -> flexion in XY plane)
    // -------------------------------------------------------------------------
    // Joint axis visualization: cylinder along revolute axis (Z)
    auto joint_cyl = chrono_types::make_shared<ChVisualShapeCylinder>(0.003, 0.015);
    joint_cyl->SetColor(ChColor(0.75f, 0.75f, 0.75f));

    auto joint_MCP = chrono_types::make_shared<ChLinkRevolute>();
    joint_MCP->Initialize(ground, body_A, ChFrame<>(ChVector3d(0, 0, 0)));
    joint_MCP->AddVisualShape(joint_cyl);
    sys.AddLink(joint_MCP);

    auto joint_PIP = chrono_types::make_shared<ChLinkRevolute>();
    joint_PIP->Initialize(body_A, body_B, ChFrame<>(ChVector3d(L1, 0, 0)));
    joint_PIP->AddVisualShape(joint_cyl);
    sys.AddLink(joint_PIP);

    auto joint_DIP = chrono_types::make_shared<ChLinkRevolute>();
    joint_DIP->Initialize(body_B, body_C, ChFrame<>(ChVector3d(L1 + L2, 0, 0)));
    joint_DIP->AddVisualShape(joint_cyl);
    sys.AddLink(joint_DIP);

    // -------------------------------------------------------------------------
    // ANCF beam phalanges (1 element each, ChElementBeamANCF_3243)
    // -------------------------------------------------------------------------

    // Helper: create a beam segment and weld its end nodes to two bodies
    auto create_phalanx = [&](std::shared_ptr<ChMesh> mesh, ChVector3d start_pos, ChVector3d end_pos,
                              double thick, double width,
                              std::shared_ptr<ChBody> body_start, std::shared_ptr<ChBody> body_end) {
        auto node_start = chrono_types::make_shared<ChNodeFEAxyzDDD>(start_pos);
        auto node_end = chrono_types::make_shared<ChNodeFEAxyzDDD>(end_pos);
        mesh->AddNode(node_start);
        mesh->AddNode(node_end);

        auto element = chrono_types::make_shared<ChElementBeamANCF_3243>();
        element->SetNodes(node_start, node_end);
        element->SetDimensions((end_pos - start_pos).Length(), thick, width);
        element->SetMaterial(bone_material);
        element->SetAlphaDamp(0.0);
        mesh->AddElement(element);

        // Pin start node to body_start (position only)
        auto link_pos_start = chrono_types::make_shared<ChLinkNodeFrame>();
        link_pos_start->Initialize(node_start, body_start);
        sys.Add(link_pos_start);

        // Pin end node to body_end (position only)
        auto link_pos_end = chrono_types::make_shared<ChLinkNodeFrame>();
        link_pos_end->Initialize(node_end, body_end);
        sys.Add(link_pos_end);
    };

    // Proximal phalanx: ground -> body_A
    create_phalanx(mesh_prox, ChVector3d(0, 0, 0), ChVector3d(L1, 0, 0), pp_thick, pp_width, ground, body_A);

    // Middle phalanx: body_A -> body_B
    create_phalanx(mesh_mid, ChVector3d(L1, 0, 0), ChVector3d(L1 + L2, 0, 0), mp_thick, mp_width, body_A, body_B);

    // Distal phalanx: body_B -> body_C
    create_phalanx(mesh_dist, ChVector3d(L1 + L2, 0, 0), ChVector3d(L1 + L2 + L3, 0, 0), dp_thick, dp_width, body_B, body_C);

    // -------------------------------------------------------------------------
    // ANCF cable tendon
    // -------------------------------------------------------------------------
    auto cable_section = chrono_types::make_shared<ChBeamSectionCable>();
    cable_section->SetDiameter(tendon_diameter);
    cable_section->SetYoungModulus(tendon_E);
    cable_section->SetRayleighDamping(tendon_damp);

    // Cable nodes (all on palmar side, offset in -Y)
    double yo = palmar_offset;
    ChVector3d slope_init(1, 0, 0);

    auto cn_pull = chrono_types::make_shared<ChNodeFEAxyzD>(ChVector3d(-L_pull, yo, 0), slope_init);
    auto cn0 = chrono_types::make_shared<ChNodeFEAxyzD>(ChVector3d(0, yo, 0), slope_init);
    auto cn1 = chrono_types::make_shared<ChNodeFEAxyzD>(ChVector3d(L1, yo, 0), slope_init);
    auto cn2 = chrono_types::make_shared<ChNodeFEAxyzD>(ChVector3d(L1 + L2, yo, 0), slope_init);
    auto cn3 = chrono_types::make_shared<ChNodeFEAxyzD>(ChVector3d(L1 + L2 + L3, yo, 0), slope_init);

    mesh_cable->AddNode(cn_pull);
    mesh_cable->AddNode(cn0);
    mesh_cable->AddNode(cn1);
    mesh_cable->AddNode(cn2);
    mesh_cable->AddNode(cn3);

    // Cable elements
    auto ce_pull = chrono_types::make_shared<ChElementCableANCF>();
    ce_pull->SetNodes(cn_pull, cn0);
    ce_pull->SetSection(cable_section);
    mesh_cable->AddElement(ce_pull);

    auto ce0 = chrono_types::make_shared<ChElementCableANCF>();
    ce0->SetNodes(cn0, cn1);
    ce0->SetSection(cable_section);
    mesh_cable->AddElement(ce0);

    auto ce1 = chrono_types::make_shared<ChElementCableANCF>();
    ce1->SetNodes(cn1, cn2);
    ce1->SetSection(cable_section);
    mesh_cable->AddElement(ce1);

    auto ce2 = chrono_types::make_shared<ChElementCableANCF>();
    ce2->SetNodes(cn2, cn3);
    ce2->SetSection(cable_section);
    mesh_cable->AddElement(ce2);

    // -------------------------------------------------------------------------
    // Cable constraints: sliding pulleys + tip weld
    // -------------------------------------------------------------------------

    // MCP pulley: cable node cn0 slides along ground's local X, locked in Y/Z
    auto pulley_MCP = chrono_types::make_shared<ChLinkNodeFrameGeneric>(false, true, true);
    pulley_MCP->Initialize(cn0, ground);
    sys.Add(pulley_MCP);

    // PIP pulley: cable node cn1 slides along body_A's local X
    auto pulley_PIP = chrono_types::make_shared<ChLinkNodeFrameGeneric>(false, true, true);
    pulley_PIP->Initialize(cn1, body_A);
    sys.Add(pulley_PIP);

    // DIP pulley: cable node cn2 slides along body_B's local X
    auto pulley_DIP = chrono_types::make_shared<ChLinkNodeFrameGeneric>(false, true, true);
    pulley_DIP->Initialize(cn2, body_B);
    sys.Add(pulley_DIP);

    // Tip weld: cable node cn3 fully pinned to body_C
    auto weld_tip = chrono_types::make_shared<ChLinkNodeFrame>();
    weld_tip->Initialize(cn3, body_C);
    sys.Add(weld_tip);

    // -------------------------------------------------------------------------
    // Visualization (optional)
    // -------------------------------------------------------------------------
#ifdef USE_VISUALIZATION
    std::shared_ptr<ChVisualSystem> vis;
    if (use_vis) {
        // Helper: add colored surface + node visualization to a mesh
        auto add_mesh_vis = [](std::shared_ptr<ChMesh> m, ChColor color) {
            auto vis_surf = chrono_types::make_shared<ChVisualShapeFEA>();
            vis_surf->SetFEMdataType(ChVisualShapeFEA::DataType::SURFACE);
            vis_surf->SetWireframe(false);
            vis_surf->SetSmoothFaces(true);
            vis_surf->SetDefaultMeshColor(color);
            m->AddVisualShapeFEA(vis_surf);

            auto vis_nodes = chrono_types::make_shared<ChVisualShapeFEA>();
            vis_nodes->SetFEMglyphType(ChVisualShapeFEA::GlyphType::NODE_DOT_POS);
            vis_nodes->SetFEMdataType(ChVisualShapeFEA::DataType::NONE);
            vis_nodes->SetSymbolsThickness(0.003);
            vis_nodes->SetDefaultSymbolsColor(ChColor(0.0f, 0.0f, 0.0f));
            m->AddVisualShapeFEA(vis_nodes);
        };

        // Proximal phalanx - blue
        add_mesh_vis(mesh_prox, ChColor(0.2f, 0.4f, 0.9f));
        // Middle phalanx - green
        add_mesh_vis(mesh_mid, ChColor(0.2f, 0.8f, 0.3f));
        // Distal phalanx - orange
        add_mesh_vis(mesh_dist, ChColor(1.0f, 0.6f, 0.1f));
        // Cable/tendon - red
        add_mesh_vis(mesh_cable, ChColor(0.9f, 0.1f, 0.1f));

        vis = CreateVisualizationSystem(ChVisualSystem::Type::VSG, CameraVerticalDir::Y, sys, "ANCF Tendon-Driven Finger",
                                        ChVector3d(L_total / 2, 0.05, 0.15),
                                        ChVector3d(L_total / 2, 0.01, 0));

    }
#endif

    // -------------------------------------------------------------------------
    // Static equilibrium solve (settle constraints before dynamics)
    // -------------------------------------------------------------------------
    sys.DoStaticLinear();

    // -------------------------------------------------------------------------
    // Simulation loop
    // -------------------------------------------------------------------------
    double timestep = 1e-3;
    double t_print = 0.0;
    double print_interval = 0.1;
    std::cout << "\nSimulation: ramp " << t_ramp << "s -> hold " << t_hold << "s -> release " << t_release
              << "s -> settle " << t_settle << "s  (total " << t_end << "s)\n"
              << std::endl;

    while (sys.GetChTime() < t_end) {
#ifdef USE_VISUALIZATION
        if (vis && !vis->Run())
            break;
#endif
        double t = sys.GetChTime();

        // Compute tendon force: ramp up -> hold -> ramp down -> zero
        double F_scale = 0.0;
        if (t < t_ramp) {
            F_scale = t / t_ramp;
        } else if (t < t_ramp + t_hold) {
            F_scale = 1.0;
        } else if (t < t_ramp + t_hold + t_release) {
            F_scale = 1.0 - (t - t_ramp - t_hold) / t_release;
        }
        cn_pull->SetForce(ChVector3d(F_pull_max * F_scale, 0, 0));

        // Print joint info periodically
        if (t >= t_print) {
            std::cout << "t=" << t << "s  |  body_A=(" << body_A->GetPos().x() << ", " << body_A->GetPos().y()
                      << ")  body_B=(" << body_B->GetPos().x() << ", " << body_B->GetPos().y() << ")  body_C=("
                      << body_C->GetPos().x() << ", " << body_C->GetPos().y() << ")" << std::endl;
            t_print += print_interval;
        }

#ifdef USE_VISUALIZATION
        if (vis) {
            vis->BeginScene();
            vis->Render();
            vis->EndScene();
        }
#endif

        sys.DoStepDynamics(timestep);
    }

    return 0;
}
