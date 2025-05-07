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
    // compute the first derivative of the polynomial
    const double dx = 2.0 * a * x + b;
    // computes the denominator of the curvature formula
    const double denom = std::pow(1.0 + dx * dx, 1.5);
    // Uses ternary operator to avoid division by zero
    return denom != 0.0 ? std::abs(2.0 * a) / denom : 0.0;
}

// AW 15.4: new method added to compute curvature from rotated fitting
// Idea: compute curvature from the fitted parabola in rotated space, defined as  x=ay2+by+c
double CurvatureFittingUtility::ComputeRotatedParabolaCurvature(double a, double b, double y)
{
    // compute the first derivative of the polynomial
    const double dx_dy = 2.0 * a * y + b;
    // computes the denominator of the curvature formula
    const double denom = std::pow(1.0 + dx_dy * dx_dy, 1.5);
    // Uses ternary operator to avoid division by zero
    return denom != 0.0 ? std::abs(2.0 * a) / denom : 0.0;
}

// computes curvature of a circle, which is straightforward knowing the radius
double CurvatureFittingUtility::ComputeRadiusCurvature(double radius)
{
    return radius != 0.0 ? 1.0 / radius : std::numeric_limits<double>::infinity();
}

void CurvatureFittingUtility::ComputeFittedCurvatures(
    // the file containing the fitted parabola coefficients per element id
    const std::string& rParabolaFilename,
    // the file containing the fitted circle coefficients per element id
    const std::string& rCircleFilename,
    // the file containing the intersection points per element id
    const std::string& rIntersectionFilename,
    // AW 15.4: new file containing the original neighbours (depending on the chosen neighbourhood depth) per element id
    const std::string& rOriginalNeighboursFileName,
    // AW 15.4: new file containing the rotated neighbours (depending on the chosen neighbourhood depth) per element id
    const std::string& rRotatedNeighboursFileName,
    // the (to be filled) output file containing the curvature per element id
    const std::string& rOutputCSV)
{   
    // Usage of hash maps to store key-element combinations (facilitates fast look-up)
    // this map stores the elemental x-values per element id
    std::unordered_map<int, std::vector<double>> element_x_values;
    // AW 18.4: this map stores the elemental y-values per element id
    std::unordered_map<int, std::vector<double>> element_y_values;
    // AW 15.4: this stores the intersection points per element id
    std::unordered_map<int, std::pair<std::pair<double, double>, std::pair<double, double>>> intersection_map; 
    // this map stores the average x-value per element id
    std::unordered_map<int, double> avg_x_map;
    // AW 15.4: new maps needed for the neighbouring (original + rotated) points
    // this map stores the rotated neighbours (depending on the chosen neighbourhood depth) per element id
    std::unordered_map<int, std::vector<std::pair<double, double>>> rotated_neighbors;
    // this map stores the original neighbours (depending on the chosen neighbourhood depth) per element id
    std::unordered_map<int, std::vector<std::pair<double, double>>> original_neighbors;

    // AW 15.4
    // --- Read element_points_rotated.txt ---
    {
        // read from the corresponding file line by line
        std::ifstream file(rRotatedNeighboursFileName);
        std::string line;
        // read each line from the file at once 
        while (std::getline(file, line)) {
            // Wrap the line in a stringstream to easily extract individual fields
            std::stringstream ss(line);
            // extract element id, as well as x,y coordinates from the current line
            int id;
            double x, y;
            ss >> id >> x >> y;
            rotated_neighbors[id].emplace_back(y, x); // y is first because we fit x = f(y)
        }
    }

    // --- Read element_points_original.txt ---
    {
        // (see comments for prior reading)
        std::ifstream file(rOriginalNeighboursFileName);
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            int id;
            double x, y;
            ss >> id >> x >> y;
            original_neighbors[id].emplace_back(x, y); // x is first because we fit y = f(x)
        }
    }


    // === Read intersection_points.txt ===
    {
        // std::ifstream is used to read from file line by line
        std::ifstream file(rIntersectionFilename);
        std::string line;
        std::getline(file, line); // Skip header by disregarding first line
        
        // AW 15.4: changed due to rotational fitting consideration
        // Loop through all remaining lines in the file — each line corresponds to one intersection point on an element
        while (std::getline(file, line)) {
            // Prepare to split the line using tab characters (\t) by putting it into a std::stringstream
            std::stringstream ss(line);
            std::string token;
            // extracts first character, converts it to integer and stores
            std::getline(ss, token, '\t'); int element_id = std::stoi(token);
            // AW 16.4: changed to first convert before assigning point id 
            std::getline(ss, token, '\t'); int point_id = std::stoi(token);
            // for coordinates, conversion into double
            std::getline(ss, token, '\t'); double x = std::stod(token);
            std::getline(ss, token, '\t'); double y = std::stod(token);
            // skips the z-coordinate for this (2d) problem
            std::getline(ss, token, '\t'); // z
            
            // store elemental id and corresponding x-coordinate
            element_x_values[element_id].push_back(x);
            element_y_values[element_id].push_back(y);
            // store x,y coordinates in the intersection map, depending on which of the intersection points per element it is
            if (point_id == 1)
                intersection_map[element_id].second = std::make_pair(x, y);
            else
                intersection_map[element_id].first = std::make_pair(x, y);
        }

        // Loop over all elements that had intersection points
        for (const auto& [id, x_vals] : element_x_values) {
            // Compute and store the average x-coordinate of the intersection points for each element in the avg_x_map
            double sum = 0.0;
            for (double x : x_vals) sum += x;
            avg_x_map[id] = sum / x_vals.size();
        }
    }

    // AW 15.4: new block, incl rotated fitting
    //  Maps element_id → (a, b) for the parabola equation
    std::unordered_map<int, std::pair<double, double>> parabola_coeffs;
    // Stores whether each element had a coordinate rotation applied during fitting
    std::unordered_map<int, bool> is_rotated_map;

    // === Read element_curves_parabola.txt ===
    {
        std::ifstream file(rParabolaFilename);
        std::string line;
        std::getline(file, line); // Header

        // Parse column indices
        // Builds a header_map that maps column names like "a(x^2)", "Element_ID", etc. to their column indices. This makes the code robust to column order
        std::unordered_map<std::string, int> header_map;
        std::stringstream header_stream(line);
        std::string token;
        int idx = 0;
        while (std::getline(header_stream, token, '\t')) {
            header_map[token] = idx++;
        }

        // Reads each line and splits it by tabs into a fields vector
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::vector<std::string> fields;
            while (std::getline(ss, token, '\t')) {
                fields.push_back(token);
            }
            
            // Parses values from the line based on the header map
            int element_id = std::stoi(fields[header_map["Element_ID"]]);
            bool rotated = std::stoi(fields[header_map["Rotated"]]) == 1;
            double a = std::stod(fields[header_map["a(x^2)"]]);
            double b = std::stod(fields[header_map["b(x)"]]);
            
            // Adds (a, b) and rotation flag to the appropriate maps
            parabola_coeffs[element_id] = std::make_pair(a, b);
            is_rotated_map[element_id] = rotated;
        }
    }

    // === Read element_curves.txt ===
    // Declares a hash map that stores the radius of the circle fitted to each element
    std::unordered_map<int, double> radius_map;
    {
        std::ifstream file(rCircleFilename);
        std::string line;
        std::getline(file, line); // Header

        // builds header map (see comments in prior block)
        std::unordered_map<std::string, int> header_map;
        std::stringstream header_stream(line);
        std::string token;
        int idx = 0;
        while (std::getline(header_stream, token, '\t')) {
            header_map[token] = idx++;
        }

        // stores necessary information per element
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
    //  Opens the output file stream (given by rOutputCSV) to write results into
    std::ofstream out(rOutputCSV);
    // writes floats in fixed point notation and uses high precision
    out << std::fixed << std::setprecision(12);
    // writes the header line of the csv file
    out << "Element_ID,X,kappa_parabola,kappa_radius,Rotated\n";

    // AW 15.4: allows using the neighbours for original, unrotated points or not
    const bool use_original_neighbors = false;
    const bool use_rotated_neighbors = true;
    // AW 18.4: allows either computing the curvature only on the input points (true), or additionally on n points in between
    const bool interpolate_curvature_on_original_points_only = true;
    const bool interpolate_curvature_on_rotated_points_only = true;

    // loops over each element id and its corresponding x values of the intersection points
    for (const auto& [id, x_vals] : element_x_values) {
        // start with undefined curvature (nan); updated if available;
        // idea: trace back which curvatures could not be updated in case nan appears in csv file
        double kp = std::numeric_limits<double>::quiet_NaN();
        double kr = std::numeric_limits<double>::quiet_NaN();
        
        // initial check if the current element has parabola coefficients
        if (parabola_coeffs.count(id)) {
            // Initializes an accumulator to sum all computed curvature values
            double sum_kappa = 0.0;
            const auto& [a, b] = parabola_coeffs[id];
            
            if (is_rotated_map.count(id) && is_rotated_map[id]) {
                const auto& points = rotated_neighbors[id];
                
                // AW 24.4: print statement removed
                /* std::cout << "\n[DEBUG] Element " << id << " is ROTATED" << std::endl;
                std::cout << "  rot_a: " << a << ", rot_b: " << b << std::endl; */
            
                if (use_rotated_neighbors) {
                    if (interpolate_curvature_on_rotated_points_only) {
                        // === Option 1: Compute directly on rotated neighbor points ===
                        for (std::size_t j = 0; j < points.size(); ++j) {
                            double y_rot = points[j].first;
                            double kappa = ComputeRotatedParabolaCurvature(a, b, y_rot);
                            sum_kappa += kappa;
                            // AW 24.4: print statement removed
                            // std::cout << "    [j=" << j << "] y_rot: " << y_rot << ", kappa: " << kappa << std::endl;
                        }
                        kp = sum_kappa / points.size();
                        // AW 24.4: print statement removed
                        // std::cout << "  --> Averaged ROTATED curvature (with neighbors): " << kp << std::endl;
                    } else {
                        // === Option 2: Interpolate between all consecutive neighbor points ===
                        const int num_interp_points = 100;
                        int count = 0;
            
                        for (std::size_t j = 1; j < points.size(); ++j) {
                            double y1 = points[j - 1].first;
                            double y2 = points[j].first;
                            double step = (y2 - y1) / (num_interp_points + 1);
            
                            for (int k = 1; k <= num_interp_points; ++k) {
                                double y_interp = y1 + k * step;
                                double kappa = ComputeRotatedParabolaCurvature(a, b, y_interp);
                                sum_kappa += kappa;
                                ++count;
            
                                /* std::cout << "    [seg j=" << j << ", k=" << k << "] y: " << y_interp 
                                          << ", kappa: " << kappa << std::endl; */
                            }
                        }
            
                        kp = (count > 0) ? (sum_kappa / count) : 0.0;
                        // AW 24.4: print statement removed
                        // std::cout << "  --> Averaged ROTATED curvature (interpolated between neighbors): " << kp << std::endl;
                    }
                } 
                else if (!use_rotated_neighbors) {
                    double y1 = element_y_values[id][0];
                    double y2 = element_y_values[id][1];
                    
                    // AW 24.4: print statement removed
                    // std::cout << "  --> ROTATED fallback on intersection points: y1 = " << y1 << ", y2 = " << y2 << std::endl;
            
                    if (interpolate_curvature_on_rotated_points_only) {
                        double kappa1 = ComputeRotatedParabolaCurvature(a, b, y1);
                        double kappa2 = ComputeRotatedParabolaCurvature(a, b, y2);
                        kp = 0.5 * (kappa1 + kappa2);
                        
                        // AW 24.4: print statement removed
                        /* std::cout << "    [Points-only] kappa1: " << kappa1 << ", kappa2: " << kappa2 << std::endl;
                        std::cout << "  --> Averaged ROTATED curvature (2-point): " << kp << std::endl; */
                    } else {
                        const int num_points = 100;
                        double step = (y2 - y1) / (num_points + 1);
                        sum_kappa = 0.0;
            
                        for (int j = 1; j <= num_points; ++j) {
                            double y_interp = y1 + j * step;
                            double kappa = ComputeRotatedParabolaCurvature(a, b, y_interp);
                            sum_kappa += kappa;
            
                            // AW 24.4: print statement removed
                            // std::cout << "    [j=" << j << "] y: " << y_interp << ", kappa: " << kappa << std::endl;
                        }
            
                        kp = sum_kappa / num_points;
                        // AW 24.4: print statement removed
                        // std::cout << "  --> Averaged ROTATED curvature (interpolated between intersections): " << kp << std::endl;
                    }
                }
            }
            // unrotated case
            else {
                // AW 24.4: print statement removed
                // std::cout << "Entering unrotated fitting" << std::endl;
                if (use_original_neighbors) {
                    const auto& points = original_neighbors[id];
                    
                    if (interpolate_curvature_on_original_points_only) {
                        // === Option 1: Compute directly on neighbor points ===
                        for (std::size_t i = 0; i < points.size(); ++i) {
                            double x = points[i].first;
                            double kappa = ComputeParabolaCurvature(a, b, x);
                            sum_kappa += kappa;
                            
                            // AW 24.4: print statement removed
                            // std::cout << "    [i=" << i << "] x: " << x << ", kappa: " << kappa << std::endl;
                        }
                        kp = sum_kappa / points.size();
                        // AW 24.4: print statement removed
                        // std::cout << "  --> Averaged NON-ROTATED curvature (with neighbors): " << kp << std::endl;
                    } else {
                        // === Option 2: Interpolate between all consecutive neighbor points ===
                        const int num_interp_points = 100;
                        int count = 0;
                
                        for (std::size_t i = 1; i < points.size(); ++i) {
                            double x1 = points[i - 1].first;
                            double x2 = points[i].first;
                
                            double step = (x2 - x1) / (num_interp_points + 1);
                
                            for (int j = 1; j <= num_interp_points; ++j) {
                                double xi = x1 + j * step;
                                double kappa = ComputeParabolaCurvature(a, b, xi);
                                sum_kappa += kappa;
                                ++count;
                
                                /* std::cout << "    [seg i=" << i << ", j=" << j << "] xi: " << xi 
                                        << ", kappa: " << kappa << std::endl; */
                            }
                        }
                
                        kp = (count > 0) ? (sum_kappa / count) : 0.0;
                        // AW 24.4: print statement removed
                        // std::cout << "  --> Averaged NON-ROTATED curvature (interpolated between neighbors): " << kp << std::endl;
                    }
                    // in case curvature shall only be evaluated between intersection points, do this:
                    // Do not use neighbors;
                    // Instead, check that there are exactly 2 x-values (from intersection points)
                    }   
                else if (! use_original_neighbors) {
                    // === Fallback: interpolate between intersection points ===
                    double x1 = element_x_values[id][0];
                    double x2 = element_x_values[id][1];
                    
                    // AW 24.4: print statement removed
                    /* std::cout << "\n[DEBUG] Element " << id << " is NON-ROTATED (interpolating between x1=" << x1 << ", x2=" << x2 << ")" << std::endl;
                    std::cout << "  a: " << a << ", b: " << b << std::endl; */
                
                    if (interpolate_curvature_on_original_points_only) {
                        // === Option 1: Interpolate only on the two intersection points ===
                        double kappa1 = ComputeParabolaCurvature(a, b, x1);
                        double kappa2 = ComputeParabolaCurvature(a, b, x2);
                        kp = 0.5 * (kappa1 + kappa2);
                        
                        // AW 24.4: print statement removed
                        /* std::cout << "    [Points-only] kappa1: " << kappa1 << ", kappa2: " << kappa2 << std::endl;
                        std::cout << "  --> Averaged NON-ROTATED curvature (2-point): " << kp << std::endl; */
                    } else {
                        // === Option 2: Interpolate on 100 points between the intersections ===
                        double num_points = 100.0;
                        double step = (x2 - x1) / (num_points + 1);
                        double sum_kappa = 0.0;
                    
                        for (int i = 1; i <= num_points; ++i) {
                            double xi = x1 + i * step;
                            double kappa = ComputeParabolaCurvature(a, b, xi);
                            sum_kappa += kappa;
                            
                            // AW 24.4: print statement removed
                            // std::cout << "    [i=" << i << "] xi: " << xi << ", kappa: " << kappa << std::endl;
                        }
                
                    kp = sum_kappa / num_points;
                    // AW 24.4: print statement removed
                    // std::cout << "  --> Averaged NON-ROTATED curvature (from intersections): " << kp << std::endl;
                    }
                }
            }
        
        // for the spherical fit based curvature computation, check if the radius is available for the current element id
        if (radius_map.count(id)) {
            kr = ComputeRadiusCurvature(radius_map[id]);
        }
    
        // Average x;
        // Compute the average x-position of the intersection points (typically 2 per element)
        // this gives a representative x-coordinate for plotting the curvature later
        double x_avg = 0.0;
        for (double x : x_vals) x_avg += x;
        x_avg /= x_vals.size();
    
        // Debugging output
        if (!std::isnan(kp) && (kp < 0.0 || kp > 10000.0)) {
            // AW 24.4: print statement removed
            //std::cout << "⚠️ Warning: Element " << id << " has suspicious parabola curvature: " << kp << std::endl;
        }
        
        // AW 24.4: print statement removed
        /* std::cout << "[CurvatureFittingUtility] Element " << id
                  << " | Rotated: " << (is_rotated_map[id] ? "yes" : "no")
                  << " | Averaged kappa_parabola: " << kp << std::endl; */
    
        out << id << "," << x_avg << "," << kp << "," << kr << "," << (is_rotated_map.count(id) && is_rotated_map[id] ? 1 : 0) << "\n";
    }
    
    // AW 24.4: print statement removed
    // std::cout << "[CurvatureFittingUtility] Curvature computation complete. Output written to " << rOutputCSV << std::endl;
}
}

// static class member declaration that maps the computed curvature from the parabola to the element id
// map provides fast (c++ level) access without having to read the csv over and over again (only once per time step needed)
std::unordered_map<std::size_t, double> CurvatureFittingUtility::mParabolaCurvatureByElement;
std::unordered_map<std::size_t, bool> CurvatureFittingUtility::mElementWasRotated;


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
        std::getline(ss, token, ','); bool is_rotated = (token == "1");
        mParabolaCurvatureByElement[elem_id] = kappa_parabola;
        mElementWasRotated[elem_id] = is_rotated;
    }
}

std::pair<double, bool> CurvatureFittingUtility::GetFittedParabolaCurvature(std::size_t ElementId)
{
    auto it_curv = mParabolaCurvatureByElement.find(ElementId);
    auto it_rot = mElementWasRotated.find(ElementId);

    if (it_curv != mParabolaCurvatureByElement.end() && it_rot != mElementWasRotated.end()) {
        return {it_curv->second, it_rot->second};
    }

    return {std::numeric_limits<double>::quiet_NaN(), false};
}


} // namespace KratosDropletDynamics
   } // namespace Kratos
