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

#if defined(KRATOS_PYTHON)
// External includes
#include <pybind11/pybind11.h>


// Project includes
#include "includes/define_python.h"
#include "droplet_dynamics_application.h"
#include "droplet_dynamics_application_variables.h"
#include "custom_python/add_custom_utilities_to_python.h"


namespace Kratos {
namespace Python {

PYBIND11_MODULE(KratosDropletDynamicsApplication,m)
{
    namespace py = pybind11;

    py::class_<KratosDropletDynamicsApplication,
        KratosDropletDynamicsApplication::Pointer,
        KratosApplication>(m, "KratosDropletDynamicsApplication")
        .def(py::init<>())
        ;

    AddCustomUtilitiesToPython(m);

    //registering variables in python
    KRATOS_REGISTER_IN_PYTHON_3D_VARIABLE_WITH_COMPONENTS(m,EXT_INT_FORCE)
    
    //Auxiliary variable to store maximum element size (h_{max})
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m,NODAL_H_MAX)

    // Smoothed surface to calculate DISTANCE_GRADIENT
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m,DISTANCE_AUX);
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m,DISTANCE_AUX2);
    KRATOS_REGISTER_IN_PYTHON_3D_VARIABLE_WITH_COMPONENTS(m,DISTANCE_GRADIENT_AUX);

    // Parallel levelset distance calculator needs an AREA_VARIABLE_AUX
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m,AREA_VARIABLE_AUX);

    // A variable to check if node is on cut element (maybe in a layer farther for future!)
    //KRATOS_REGISTER_IN_PYTHON_VARIABLE(m, IS_NEAR_CUT)

    // Contact line calculation
    KRATOS_REGISTER_IN_PYTHON_3D_VARIABLE_WITH_COMPONENTS(m,NORMAL_VECTOR);
    KRATOS_REGISTER_IN_PYTHON_3D_VARIABLE_WITH_COMPONENTS(m,TANGENT_VECTOR);
    KRATOS_REGISTER_IN_PYTHON_3D_VARIABLE_WITH_COMPONENTS(m,CONTACT_VECTOR);
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m, CONTACT_ANGLE);
    KRATOS_REGISTER_IN_PYTHON_3D_VARIABLE_WITH_COMPONENTS(m,CONTACT_VECTOR_MICRO);
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m, CONTACT_ANGLE_MICRO);
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m, CONTACT_VELOCITY);

    ////////////////////
    // AW 14.5: adapted to include latest adaptation by Alireza
    KRATOS_REGISTER_IN_PYTHON_3D_VARIABLE_WITH_COMPONENTS( m, VELOCITY_X_GRADIENT)
    KRATOS_REGISTER_IN_PYTHON_3D_VARIABLE_WITH_COMPONENTS( m, VELOCITY_Y_GRADIENT)
    KRATOS_REGISTER_IN_PYTHON_3D_VARIABLE_WITH_COMPONENTS( m, VELOCITY_Z_GRADIENT)
    KRATOS_REGISTER_IN_PYTHON_VARIABLE( m, DISTANCE_GRADIENT_DIVERGENCE)
    KRATOS_REGISTER_IN_PYTHON_3D_VARIABLE_WITH_COMPONENTS( m, DISTANCE_GRADIENT_SIZE_GRADIENT)
    ///////////////
    // AW 14.5: adaptation end

    // Enriched pressure is an array of NumNodes components defined for elements. Access it using Element.GetValue()
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m, ENRICHED_PRESSURE_1)
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m, ENRICHED_PRESSURE_2)
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m, ENRICHED_PRESSURE_3)
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m, ENRICHED_PRESSURE_4)

    // Last known velocity and pressure to recalculate the last increment
    KRATOS_REGISTER_IN_PYTHON_3D_VARIABLE_WITH_COMPONENTS(m, VELOCITY_STAR)
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m, PRESSURE_STAR)

    // Pressure gradient to calculate its jump over interface
    // KRATOS_REGISTER_IN_PYTHON_3D_VARIABLE_WITH_COMPONENTS(m, PRESSURE_GRADIENT_AUX)

    // Level-set convective velocity
    KRATOS_REGISTER_IN_PYTHON_3D_VARIABLE_WITH_COMPONENTS(m, CONVECTIVE_VELOCITY)

    // AW 21.4: Quasi-static contact line model
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m, quasi_static_contact_angle)
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m, theta_equilibrium_hydrophilic)
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m, theta_equilibrium_hydrophobic)
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m, penalty_coefficient)
    // AW 2.6
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m, X_threshold)
    // AW 25.6
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m, smooth_scaling)
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m, smooth_scaling_lower_threshold)
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m, smooth_scaling_upper_threshold)

    // AW 28.4:fitted curvature and normals
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m,  FITTED_CURVATURE)
    KRATOS_REGISTER_IN_PYTHON_3D_VARIABLE_WITH_COMPONENTS(m, FITTED_NORMAL)

    // AW 2.5:fitted curvature and normals at the Gauss Points
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m,  CURVATURE_FITTED_GAUSS1)
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m,  CURVATURE_FITTED_GAUSS2)
    KRATOS_REGISTER_IN_PYTHON_3D_VARIABLE_WITH_COMPONENTS(m, NORMAL_FITTED_GAUSS1)
    KRATOS_REGISTER_IN_PYTHON_3D_VARIABLE_WITH_COMPONENTS(m, NORMAL_FITTED_GAUSS2)

    // AW 19.5: Fitting control variables
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m, FittingType)
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m, UsePartialFitting)
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m, FittingElementIds)
    KRATOS_REGISTER_IN_PYTHON_VARIABLE(m, NormalEvaluationMode)

}

} // namespace Python.
} // namespace Kratos.

#endif // KRATOS_PYTHON defined
