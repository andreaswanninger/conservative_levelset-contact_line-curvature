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

// AW 9.4: added utility for curvature computation

#include "curvature_fitting_utility.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace Kratos {
namespace KratosDropletDynamics {

double CurvatureFittingUtility::ComputeParabolaCurvature(double a, double b, double x)
{
    const double dx = 2.0 * a * x + b;
    const double denom = std::pow(1.0 + dx * dx, 1.5);
    return denom != 0.0 ? std::abs(2.0 * a) / denom : 0.0;
}

double CurvatureFittingUtility::ComputeRadiusCurvature(double radius)
{
    return radius != 0.0 ? 1.0 / radius : std::numeric_limits<double>::infinity();
}

void CurvatureFittingUtility::ComputeFittedCurvatures(
    const std::string& rParabolaFilename,
    const std::string& rCircleFilename,
    const std::string& rIntersectionFilename,
    const std::string& rOutputCSV)
{   
    std::unordered_map<int, std::vector<double>> element_x_values;
    std::unordered_map<int, double> avg_x_map;

    // === Read intersection_points.txt ===
    {
        std::ifstream file(rIntersectionFilename);
        std::string line;
        std::getline(file, line); // Skip header

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string token;
            std::getline(ss, token, '\t'); int element_id = std::stoi(token);
            std::getline(ss, token, '\t'); // point_id
            std::getline(ss, token, '\t'); double x = std::stod(token);
            // Skip Y, Z
            element_x_values[element_id].push_back(x);
        }

        for (const auto& [id, x_vals] : element_x_values) {
            double sum = 0.0;
            for (double x : x_vals) sum += x;
            avg_x_map[id] = sum / x_vals.size();
        }
    }

    // === Read element_curves_parabola.txt ===
    std::unordered_map<int, std::pair<double, double>> parabola_coeffs;
    {
        std::ifstream file(rParabolaFilename);
        std::string line;
        std::getline(file, line); // Header

        // Figure out column indices from header
        std::unordered_map<std::string, int> header_map;
        std::stringstream header_stream(line);
        std::string token;
        int idx = 0;
        while (std::getline(header_stream, token, '\t')) {
            header_map[token] = idx++;
        }

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::vector<std::string> fields;
            while (std::getline(ss, token, '\t')) {
                fields.push_back(token);
            }
            int element_id = std::stoi(fields[header_map["Element_ID"]]);
            double a = std::stod(fields[header_map["a(x²)"]]);
            double b = std::stod(fields[header_map["b(x)"]]);
            parabola_coeffs[element_id] = std::make_pair(a, b);
        }
    }

    // === Read element_curves.txt ===
    std::unordered_map<int, double> radius_map;
    {
        std::ifstream file(rCircleFilename);
        std::string line;
        std::getline(file, line); // Header

        std::unordered_map<std::string, int> header_map;
        std::stringstream header_stream(line);
        std::string token;
        int idx = 0;
        while (std::getline(header_stream, token, '\t')) {
            header_map[token] = idx++;
        }

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::vector<std::string> fields;
            while (std::getline(ss, token, '\t')) {
                fields.push_back(token);
            }
            int element_id = std::stoi(fields[header_map["Element_ID"]]);
            double radius = std::stod(fields[header_map["radius"]]);
            radius_map[element_id] = radius;
        }
    }

    // === Write output file ===
    std::ofstream out(rOutputCSV);
    out << std::fixed << std::setprecision(12);
    out << "Element_ID,X,kappa_parabola,kappa_radius\n";

    for (const auto& [id, x_vals] : element_x_values) {
        double kp = std::numeric_limits<double>::quiet_NaN();
        double kr = std::numeric_limits<double>::quiet_NaN();
    
        // Only use elements with exactly 2 intersection points
        if (x_vals.size() == 2 && parabola_coeffs.count(id)) {
            auto [a, b] = parabola_coeffs[id];
            double x1 = x_vals[0];
            double x2 = x_vals[1];
    
            const int num_points = 10;
            double step = (x2 - x1) / (num_points + 1);
            double sum_kappa = 0.0;
    
            for (int i = 1; i <= num_points; ++i) {
                double xi = x1 + i * step;
                sum_kappa += ComputeParabolaCurvature(a, b, xi);
            }
    
            kp = sum_kappa / num_points;
        }
    
        if (radius_map.count(id)) {
            kr = ComputeRadiusCurvature(radius_map[id]);
        }
    
        // For debug: print if curvature looks unphysical
        if (!std::isnan(kp) && (kp < 0.0 || kp > 10000.0)) {
            std::cout << "⚠️ Warning: Element " << id << " has suspicious parabola curvature: " << kp << std::endl;
        }
    
        // Output average x for traceability
        double x_avg = 0.0;
        for (double x : x_vals) x_avg += x;
        x_avg /= x_vals.size();
    
        out << id << "," << x_avg << "," << kp << "," << kr << "\n";
    }

    std::cout << "[CurvatureFittingUtility] Curvature computation complete. Output written to " << rOutputCSV << std::endl;
}

std::unordered_map<std::size_t, double> CurvatureFittingUtility::mParabolaCurvatureByElement;

void CurvatureFittingUtility::LoadCurvatureCSV(const std::string& rCSVFile)
{
    mParabolaCurvatureByElement.clear();

    std::ifstream file(rCSVFile);
    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::getline(ss, token, ','); std::size_t elem_id = std::stoul(token);
        std::getline(ss, token, ','); /* x */
        std::getline(ss, token, ','); double kappa_parabola = std::stod(token);
        mParabolaCurvatureByElement[elem_id] = kappa_parabola;
    }
}

double CurvatureFittingUtility::GetFittedParabolaCurvature(std::size_t ElementId)
{
    auto it = mParabolaCurvatureByElement.find(ElementId);
    if (it != mParabolaCurvatureByElement.end())
        return it->second;
    return std::numeric_limits<double>::quiet_NaN(); // fallback if not found
}


} // namespace KratosDropletDynamics
} // namespace Kratos
