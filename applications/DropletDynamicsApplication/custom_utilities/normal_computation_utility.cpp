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

// AW 10.4: added utility for normal computation

// AW 10.4: includes the header file 
#include "normal_computation_utility.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include <iomanip>

// AW 10.4: Opens the same namespace defined in the .h file again for the implementation
namespace Kratos { 
namespace KratosDropletDynamics
{

// AW 10.4: struct to store parabola coefficients y=ax2+bx+c
struct Coefficients
{
    double a, b, c;
};

// AW 10.4: struct for basic 2D point
struct Point
{
    double x, y;
};


void NormalComputationUtility::ComputeAveragedNormals(
    const std::string& parabola_file,
    const std::string& intersection_file,
    const std::string& output_csv)
{
    std::map<int, Coefficients> curve_map;
    // AW 15.4: additionally needed maps for the rotated fitting
    std::map<int, bool> rotated_flag;
    std::map<int, std::pair<double, double>> rotated_ab;
    std::map<int, std::pair<Point, Point>> intersection_map;

    // AW 15.4: new reading block to also consider rotated fitting
    std::ifstream infile1(parabola_file);
    std::string line;
    std::getline(infile1, line); // Skip header

    std::map<std::string, int> header_map;
    std::stringstream header_stream(line);
    std::string token;
    int col = 0;
    while (std::getline(header_stream, token, '\t')) {
        header_map[token] = col++;
    }

    while (std::getline(infile1, line)) {
        std::istringstream ss(line);
        std::vector<std::string> fields;
        while (std::getline(ss, token, '\t')) {
            fields.push_back(token);
        }

        int elem_id = std::stoi(fields[header_map["Element_ID"]]);
        double a = std::stod(fields[header_map["a(x^2)"]]);
        double b = std::stod(fields[header_map["b(x)"]]);
        double c = std::stod(fields[header_map["c"]]);
        curve_map[elem_id] = {a, b, c};

        bool rotated = std::stoi(fields[header_map["Rotated"]]) == 1;
        rotated_flag[elem_id] = rotated;

        if (rotated) {
            double rot_a = std::stod(fields[header_map["rot_a"]]);
            double rot_b = std::stod(fields[header_map["rot_b"]]);
            rotated_ab[elem_id] = {rot_a, rot_b};
        }
    }

    // Load parabola coefficients
    /* std::ifstream infile1(parabola_file);
    std::string line;
    std::getline(infile1, line); // skip header
    while (std::getline(infile1, line))
    {
        std::istringstream iss(line);
        int elem_id;
        double a, b, c;
        std::string temp;
        std::getline(iss, temp, '\t'); elem_id = std::stoi(temp);
        for (int i = 0; i < 2; ++i) std::getline(iss, temp, '\t'); // skip 2 columns
        std::getline(iss, temp, '\t'); a = std::stod(temp);
        std::getline(iss, temp, '\t'); b = std::stod(temp);
        std::getline(iss, temp, '\t'); c = std::stod(temp);
        curve_map[elem_id] = {a, b, c};
    } */

    // Load intersection points
    std::ifstream infile2(intersection_file);
    std::getline(infile2, line); // Skip header line

    while (std::getline(infile2, line))
    {
        std::istringstream iss(line);
        std::string token;
        int elem_id = -1, pt_id = -1;
        double x = 0.0, y = 0.0, z = 0.0;

        // Split line using tab delimiter
        std::getline(iss, token, '\t'); elem_id = std::stoi(token);
        std::getline(iss, token, '\t'); pt_id   = std::stoi(token);
        std::getline(iss, token, '\t'); x       = std::stod(token);
        std::getline(iss, token, '\t'); y       = std::stod(token);
        std::getline(iss, token, '\t'); z       = std::stod(token);

        // Store into the map
        if (intersection_map.find(elem_id) == intersection_map.end())
        {
            intersection_map[elem_id] = {{x, y}, {0, 0}};
        }

        if (pt_id == 1)
            intersection_map[elem_id].second = {x, y};
        else
            intersection_map[elem_id].first = {x, y};
    }

    // Open output CSV
    std::ofstream outfile(output_csv);
    outfile << "Element_ID,Avg_Nx,Avg_Ny\n";

    // AW 15.4: new code block to also consider rotated fitting
    for (const auto& [elem_id, coeffs] : curve_map)
    {   
        if (intersection_map.find(elem_id) == intersection_map.end())
            continue;

        const auto& [p1, p2] = intersection_map[elem_id];

        // Use midpoint for a robust and meaningful result
        double avg_nx = 0.0;
        double avg_ny = 0.0;

        if (rotated_flag.count(elem_id) && rotated_flag[elem_id]) {
            // === ROTATED FIT ===
            double y_center = 0.5; // Optional: improve later using real rotated y
            const auto& [ra, rb] = rotated_ab[elem_id];

            // Compute rotated-space derivative dx/dy
            double dx_dy = 2.0 * ra * y_center + rb;

            // Rotated-space normal: (1, -dx_dy)
            double nx_rot = 1.0;
            double ny_rot = -dx_dy;
            double norm_rot = std::sqrt(nx_rot * nx_rot + ny_rot * ny_rot);
            nx_rot /= norm_rot;
            ny_rot /= norm_rot;

            // Rotate back to original system (45° CCW)
            const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
            avg_nx = inv_sqrt2 * (nx_rot - ny_rot);
            avg_ny = inv_sqrt2 * (nx_rot + ny_rot);

        } else {
            // === NORMAL FIT ===
            double x_mid = 0.5 * (p1.x + p2.x);
            double dydx = 2.0 * coeffs.a * x_mid + coeffs.b;
            double norm = std::sqrt(1.0 + dydx * dydx);
            avg_nx = -dydx / norm;
            avg_ny = 1.0 / norm;
        }

        // Final normalization (optional, for safety)
        double len = std::sqrt(avg_nx * avg_nx + avg_ny * avg_ny);
        if (len > 1e-12) {
            avg_nx /= len;
            avg_ny /= len;
        }

        // === Output result ===
        outfile << elem_id << "," << std::setprecision(12) << avg_nx << "," << avg_ny << "\n";
    }

    // AW 15.4: old code block, commented
    /* for (const auto& [elem_id, coeffs] : curve_map)
    {   
        if (intersection_map.find(elem_id) == intersection_map.end())
            continue;

        const auto& [p1, p2] = intersection_map[elem_id];

        // Use midpoint for a robust and meaningful result
        double x_mid = 0.5 * (p1.x + p2.x);
        double dydx = 2.0 * coeffs.a * x_mid + coeffs.b;
        double norm = std::sqrt(1.0 + dydx * dydx);

        double avg_nx = -dydx / norm;
        double avg_ny = 1.0 / norm;

        // Optional: normalize the average vector
        double len = std::sqrt(avg_nx * avg_nx + avg_ny * avg_ny);
        if (len > 1e-12) {
            avg_nx /= len;
            avg_ny /= len;
        }

          // === Debug prints ===
         std::cout << "Element ID: " << elem_id << std::endl;
        std::cout << "  Coefficients: a = " << coeffs.a << ", b = " << coeffs.b << ", c = " << coeffs.c << std::endl;
        std::cout << "  Intersection Points: x1 = " << p1.x << ", x2 = " << p2.x << std::endl;
        std::cout << "  Midpoint x = " << x_mid << std::endl;
        std::cout << "  Derivative f'(x) = " << dydx << std::endl;
        std::cout << "  Raw normal (not yet normalized): (" << -dydx << ", 1.0)" << std::endl;
        std::cout << "  Normalized normal: (" << avg_nx << ", " << avg_ny << ")" << std::endl;
        std::cout << std::endl; 

        outfile << elem_id << "," << std::setprecision(12) << avg_nx << "," << avg_ny << "\n";
    } */

    // std::cout << "Averaged normals written to " << output_csv << std::endl;
}

// === NEW ===
// Define static member
std::map<int, array_1d<double, 3>> NormalComputationUtility::msFittedNormals;

void NormalComputationUtility::LoadNormalCSV(const std::string& csv_filename)
{
    msFittedNormals.clear();

    std::ifstream infile(csv_filename);
    std::string line;
    std::getline(infile, line); // skip header

    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::string token;
        int elem_id;
        double nx, ny;

        std::getline(iss, token, ','); elem_id = std::stoi(token);
        std::getline(iss, token, ','); nx = std::stod(token);
        std::getline(iss, token, ','); ny = std::stod(token);

        array_1d<double, 3> normal;
        normal[0] = nx;
        normal[1] = ny;
        normal[2] = 0.0; // default Z = 0 for 2D

        msFittedNormals[elem_id] = normal;
    }

    // std::cout << "Loaded " << msFittedNormals.size() << " fitted normals from " << csv_filename << std::endl;
}

const array_1d<double, 3>& NormalComputationUtility::GetFittedNormal(int element_id)
{
    auto it = msFittedNormals.find(element_id);
    if (it == msFittedNormals.end())
    {
        static array_1d<double, 3> default_normal = ZeroVector(3);
        std::cout << "⚠️  Warning: Fitted normal not found for element " << element_id << std::endl;
        return default_normal;
    }

    return it->second;
}

}
} // namespace 
