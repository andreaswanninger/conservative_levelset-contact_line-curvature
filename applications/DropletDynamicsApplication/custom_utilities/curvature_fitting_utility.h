//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Andreas Thomas Wanninger
//
//  AW 9.4: added utility for curvature computation

#ifndef KRATOS_CURVATURE_FITTING_UTILITY_H_INCLUDED
#define KRATOS_CURVATURE_FITTING_UTILITY_H_INCLUDED

// System includes
#include <string>
#include <unordered_map>

// Project includes
#include "includes/define.h"

namespace Kratos {
namespace KratosDropletDynamics {

class CurvatureFittingUtility
{
public:
    KRATOS_API(DROPLET_DYNAMICS_APPLICATION)

    static void ComputeFittedCurvatures(
        const std::string& rParabolaFilename,
        const std::string& rCircleFilename,
        const std::string& rIntersectionFilename,
        // AW 15.4: additionally use these input files for neighbouring points
        const std::string& rOriginalNeighboursFileName,
        const std::string& rRotatedNeighboursFileName,
        const std::string& rOutputCSV = "element_curvatures_simplified.csv");

    static void LoadCurvatureCSV(const std::string& rCSVFile);
    static std::pair<double, bool> GetFittedParabolaCurvature(std::size_t ElementId);

private:
    static double ComputeParabolaCurvature(double a, double b, double x);
    static double ComputeRadiusCurvature(double radius);
    // AW 15.4: new method for the rotated fitting
    static double ComputeRotatedParabolaCurvature(double a, double b, double y);

    // Prior MISSING STATIC MEMBER DECLARATION
    static std::unordered_map<std::size_t, double> mParabolaCurvatureByElement;
    static std::unordered_map<std::size_t, bool> mElementWasRotated;
};

} // namespace KratosDropletDynamics
} // namespace Kratos

#endif // KRATOS_CURVATURE_FITTING_UTILITY_H_INCLUDED
