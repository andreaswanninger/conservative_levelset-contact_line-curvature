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

// AW 15.4: new method added to compute curvature from rotated fitting
double CurvatureFittingUtility::ComputeRotatedParabolaCurvature(double a, double b, double y)
{
    const double dx_dy = 2.0 * a * y + b;
    const double denom = std::pow(1.0 + dx_dy * dx_dy, 1.5);
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
    // AW 15.4: additionally use these input files for neighbouring points
    const std::string& rOriginalNeighboursFileName,
    const std::string& rRotatedNeighboursFileName,
    const std::string& rOutputCSV)
{   
    std::unordered_map<int, std::vector<double>> element_x_values;
    // AW 15.4: new map needed to also compute with rotated fitting
    std::unordered_map<int, std::pair<std::pair<double, double>, std::pair<double, double>>> intersection_map; 
    std::unordered_map<int, double> avg_x_map;
    // AW 15.4: new maps needed for the neighbouring (original + rotated) points
    std::unordered_map<int, std::vector<std::pair<double, double>>> rotated_neighbors;
    std::unordered_map<int, std::vector<std::pair<double, double>>> original_neighbors;

    // AW 15.4
    // --- Read element_points_rotated.txt ---
    {
        std::ifstream file(rRotatedNeighboursFileName);
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            int id;
            double x, y;
            ss >> id >> x >> y;
            rotated_neighbors[id].emplace_back(y, x); // y is first because we fit x = f(y)
        }
    }

    // --- Read element_points_original.txt ---
    {
        std::ifstream file(rOriginalNeighboursFileName);
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            int id;
            double x, y;
            ss >> id >> x >> y;
            original_neighbors[id].emplace_back(x, y);
        }
    }


    // === Read intersection_points.txt ===
    {
        std::ifstream file(rIntersectionFilename);
        std::string line;
        std::getline(file, line); // Skip header
        
        // AW 15.4: new code block for rotated fitting
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string token;
            std::getline(ss, token, '\t'); int element_id = std::stoi(token);
            std::getline(ss, token, '\t'); int point_id;
            std::getline(ss, token, '\t'); double x = std::stod(token);
            std::getline(ss, token, '\t'); double y = std::stod(token);
            std::getline(ss, token, '\t'); // z

            element_x_values[element_id].push_back(x);
            if (point_id == 1)
                intersection_map[element_id].second = std::make_pair(x, y);
            else
                intersection_map[element_id].first = std::make_pair(x, y);
        }
        // AW 15.4: old code block, outcommented
        /* while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string token;
            std::getline(ss, token, '\t'); int element_id = std::stoi(token);
            std::getline(ss, token, '\t'); // point_id
            std::getline(ss, token, '\t'); double x = std::stod(token);
            // Skip Y, Z
            element_x_values[element_id].push_back(x);
        } */

        for (const auto& [id, x_vals] : element_x_values) {
            double sum = 0.0;
            for (double x : x_vals) sum += x;
            avg_x_map[id] = sum / x_vals.size();
        }
    }

    // AW 15.4: new block, incl rotated fitting
    // === Read element_curves_parabola.txt ===
    std::unordered_map<int, std::pair<double, double>> parabola_coeffs;
    std::unordered_map<int, bool> is_rotated_map;
    std::unordered_map<int, std::pair<double, double>> rotated_coeffs;
    std::unordered_map<int, double> rotated_y_min_map;
    std::unordered_map<int, double> rotated_y_range_map;


    {
        std::ifstream file(rParabolaFilename);
        std::string line;
        std::getline(file, line); // Header

        // Parse column indices
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
            bool rotated = std::stoi(fields[header_map["Rotated"]]) == 1;
            double a = std::stod(fields[header_map["a(x^2)"]]);
            double b = std::stod(fields[header_map["b(x)"]]);

            parabola_coeffs[element_id] = std::make_pair(a, b);
            is_rotated_map[element_id] = rotated;

            if (rotated) {
                double rot_a = std::stod(fields[header_map["rot_a"]]);
                double rot_b = std::stod(fields[header_map["rot_b"]]);
                rotated_coeffs[element_id] = std::make_pair(rot_a, rot_b);
                double rot_ymin = std::stod(fields[header_map["rot_ymin"]]);
                double rot_yrange = std::stod(fields[header_map["rot_yrange"]]);
                rotated_y_min_map[element_id] = rot_ymin;
                rotated_y_range_map[element_id] = rot_yrange;
            }
        }
    }


    // AW 15.4: old block, outcommented
    /* // === Read element_curves_parabola.txt ===
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
    } */

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

    // AW 15.4: allows using the neighbours for original, unrotaed points or not
    const bool use_original_neighbors = false;

    for (const auto& [id, x_vals] : element_x_values) {
        double kp = std::numeric_limits<double>::quiet_NaN();
        double kr = std::numeric_limits<double>::quiet_NaN();
    
        if (parabola_coeffs.count(id)) {
            double sum_kappa = 0.0;
        
            if (is_rotated_map.count(id) && is_rotated_map[id] && rotated_neighbors.count(id)) {
                const auto& [rot_a, rot_b] = rotated_coeffs[id];
                const auto& points = rotated_neighbors[id];
        
                std::cout << "\n[DEBUG] Element " << id << " is ROTATED (using " << points.size() << " rotated neighbor points)" << std::endl;
                std::cout << "  rot_a: " << rot_a << ", rot_b: " << rot_b << std::endl;
        
                for (std::size_t j = 0; j < points.size(); ++j) {
                    double y_rot = points[j].first;
                    double kappa = ComputeRotatedParabolaCurvature(rot_a, rot_b, y_rot);
                    sum_kappa += kappa;
        
                    std::cout << "    [j=" << j << "] y_rot: " << y_rot << ", kappa: " << kappa << std::endl;
                }
        
                kp = sum_kappa / points.size();
                std::cout << "  --> Averaged ROTATED curvature: " << kp << std::endl;
        
            } 
            else if (original_neighbors.count(id)) {
                const auto& [a, b] = parabola_coeffs[id];
            
                if (use_original_neighbors) {
                    // === Using ORIGINAL neighbor points ===
                    const auto& points = original_neighbors[id];
                    std::cout << "\n[DEBUG] Element " << id << " is NON-ROTATED (using " << points.size() << " original neighbor points)" << std::endl;
                    std::cout << "  a: " << a << ", b: " << b << std::endl;
            
                    for (std::size_t i = 0; i < points.size(); ++i) {
                        double x = points[i].first;
                        double kappa = ComputeParabolaCurvature(a, b, x);
                        sum_kappa += kappa;
            
                        std::cout << "    [i=" << i << "] x: " << x << ", kappa: " << kappa << std::endl;
                    }
            
                    kp = sum_kappa / points.size();
                    std::cout << "  --> Averaged NON-ROTATED curvature (with neighbors): " << kp << std::endl;
                } else if (element_x_values.count(id) && element_x_values[id].size() == 2) {
                    // === Fallback: interpolate between intersection points ===
                    double num_points = 100.0;
                    double x1 = element_x_values[id][0];
                    double x2 = element_x_values[id][1];
                    double step = (x2 - x1) / (num_points + 1);
            
                    std::cout << "\n[DEBUG] Element " << id << " is NON-ROTATED (interpolating between x1=" << x1 << ", x2=" << x2 << ")" << std::endl;
                    std::cout << "  a: " << a << ", b: " << b << std::endl;
            
                    for (int i = 1; i <= num_points; ++i) {
                        double xi = x1 + i * step;
                        double kappa = ComputeParabolaCurvature(a, b, xi);
                        sum_kappa += kappa;
            
                        std::cout << "    [i=" << i << "] xi: " << xi << ", kappa: " << kappa << std::endl;
                    }
            
                    kp = sum_kappa / num_points;
                    std::cout << "  --> Averaged NON-ROTATED curvature (from intersections): " << kp << std::endl;
                }
            }
        }
        
    
        if (radius_map.count(id)) {
            kr = ComputeRadiusCurvature(radius_map[id]);
        }
    
        // Average x
        double x_avg = 0.0;
        for (double x : x_vals) x_avg += x;
        x_avg /= x_vals.size();
    
        // Debugging output
        if (!std::isnan(kp) && (kp < 0.0 || kp > 10000.0)) {
            std::cout << "⚠️ Warning: Element " << id << " has suspicious parabola curvature: " << kp << std::endl;
        }
    
        std::cout << "[CurvatureFittingUtility] Element " << id
                  << " | Rotated: " << (is_rotated_map.count(id) && is_rotated_map[id] ? "yes" : "no")
                  << " | Averaged kappa_parabola: " << kp << std::endl;
    
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
