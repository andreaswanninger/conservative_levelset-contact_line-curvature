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
//  AW 10.4: added utility for normal computation

// AW 10.4: include guards: prevent the header from being included multiple times in the same translation unit
#ifndef NORMAL_COMPUTATION_UTILITY_H
#define NORMAL_COMPUTATION_UTILITY_H

#include <string>
#include <map>                          // Needed for std::map
#include "containers/array_1d.h"        // Needed for array_1d


// AW 10.4: declares namespace
namespace Kratos { 
namespace KratosDropletDynamics
{
    // AW 10.4: defines the utility class NormalComputationUtility
    class NormalComputationUtility
    {
    public:
        static void ComputeAveragedNormals(
            const std::string& parabola_file,
            const std::string& intersection_file,
            const std::string& rotated_points_file,
            const std::string& output_csv);

        static void LoadNormalCSV(const std::string& csv_filename);
        static const array_1d<double, 3>& GetFittedNormal(int element_id);

    private:
        static std::map<int, array_1d<double, 3>> msFittedNormals;
    };
}
} // namespace 

#endif // NORMAL_COMPUTATION_UTILITY_H
