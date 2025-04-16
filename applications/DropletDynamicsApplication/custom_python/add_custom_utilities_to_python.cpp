//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Mohammad R. Hashemi
//


// System includes

// External includes
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>  // This is needed for py::array_t

// Project includes
#include "includes/define.h"
#include "custom_python/add_custom_utilities_to_python.h"

#include "spaces/ublas_space.h"
#include "linear_solvers/linear_solver.h"

#include "custom_utilities/contact_angle_evaluator.h"

#include "custom_utilities/intersection_points_utility.h"  // Include for IntersectionPointsUtility
#include "droplet_dynamics_application_variables.h"  // Include for IntersectionPointData

// Aw 9.4: include the curvature fitting calculation utility; tells the compiler to use this class from the corresponding header file
#include "custom_utilities/curvature_fitting_utility.h"

// AW 10.4: include the normal computation utility
#include "custom_utilities/normal_computation_utility.h"



namespace Kratos {
namespace Python {

void AddCustomUtilitiesToPython(pybind11::module& m)
{
    namespace py = pybind11;

    typedef UblasSpace<double, CompressedMatrix, Vector> SparseSpaceType;
    typedef UblasSpace<double, Matrix, Vector> LocalSpaceType;
    typedef LinearSolver<SparseSpaceType, LocalSpaceType > LinearSolverType;

    py::class_<ContactAngleEvaluator, ContactAngleEvaluator::Pointer, Process>(m,"ContactAngleEvaluatorProcess")
    .def(py::init<ModelPart&>())
    .def(py::init<ModelPart&, Parameters& >());

    // Register intersection points data and utility
    py::class_<IntersectionPointData>(m, "IntersectionPointData")
        .def(py::init<>())
        .def_readwrite("elementId", &IntersectionPointData::elementId)
        .def_readwrite("pointId", &IntersectionPointData::pointId)
        .def_property("coordinates",
            [](IntersectionPointData& self) { return py::array_t<double>(3, &self.coordinates[0]); },
            [](IntersectionPointData& self, py::array_t<double> arr) {
                for (int i = 0; i < 3; i++) self.coordinates[i] = arr.at(i);
            });
    
    py::class_<KratosDropletDynamics::IntersectionPointsUtility>(m, "IntersectionPointsUtility")
        .def_static("CollectElementIntersectionPoints", &KratosDropletDynamics::IntersectionPointsUtility::CollectElementIntersectionPoints)
        .def_static("ClearIntersectionPoints", &KratosDropletDynamics::IntersectionPointsUtility::ClearIntersectionPoints)
        .def_static("GetIntersectionPoints", &KratosDropletDynamics::IntersectionPointsUtility::GetIntersectionPoints, py::return_value_policy::reference)
        .def_static("SaveIntersectionPointsToFile", &KratosDropletDynamics::IntersectionPointsUtility::SaveIntersectionPointsToFile)
        // .def_static("AddIntersectionPoint", &KratosDropletDynamics::IntersectionPointsUtility::AddIntersectionPoint)
        .def_static("ExtractIntersectionPointsFromSplitter", &KratosDropletDynamics::IntersectionPointsUtility::ExtractIntersectionPointsFromSplitter)
        .def_static("DiagnosticOutput", &KratosDropletDynamics::IntersectionPointsUtility::DiagnosticOutput)
        .def_static("ProcessIntersectionPointsAndFitCurves", &KratosDropletDynamics::IntersectionPointsUtility::ProcessIntersectionPointsAndFitCurves)
        .def_static("ProcessIntersectionPointsAndFitCurvesparabola", &KratosDropletDynamics::IntersectionPointsUtility::ProcessIntersectionPointsAndFitCurvesparabola);

    
    // AW 9.4: makes it callable from python
    py::class_<KratosDropletDynamics::CurvatureFittingUtility>(m, "CurvatureFittingUtility")
    .def_static(
        "ComputeFittedCurvatures",
        &KratosDropletDynamics::CurvatureFittingUtility::ComputeFittedCurvatures,
        py::arg("parabola_filename"),
        py::arg("circle_filename"),
        py::arg("intersection_points_filename"),
         // AW 15.4: additional files added
        py::arg("original_neighbours_filename"),
        py::arg("rotated_neighbours_filename"),
        py::arg("output_csv") = "element_curvatures_simplified.csv"
    )
    .def_static(
        "LoadCurvatureCSV",
        &KratosDropletDynamics::CurvatureFittingUtility::LoadCurvatureCSV,
        py::arg("csv_filename")
    )
    .def_static(
        "GetFittedParabolaCurvature",
        &KratosDropletDynamics::CurvatureFittingUtility::GetFittedParabolaCurvature,
        py::arg("element_id")
    );

    // AW 10.4: makes it callable from Python
    py::class_<KratosDropletDynamics::NormalComputationUtility>(m, "NormalComputationUtility")
    .def_static(
        "ComputeAveragedNormals",
        &KratosDropletDynamics::NormalComputationUtility::ComputeAveragedNormals,
        py::arg("parabola_file"),
        py::arg("intersection_file"),
        py::arg("output_csv") = "averaged_normals.csv"
    )
    .def_static(
        "LoadNormalCSV",
        &KratosDropletDynamics::NormalComputationUtility::LoadNormalCSV,
        py::arg("csv_filename")
    )
    .def_static(
        "GetFittedNormal",
        &KratosDropletDynamics::NormalComputationUtility::GetFittedNormal,
        py::arg("element_id"),
        py::return_value_policy::reference  // return by reference to avoid copies
    );
    


}

} // namespace Python.
} // Namespace Kratos
