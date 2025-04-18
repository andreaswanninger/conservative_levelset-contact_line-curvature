//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Alireza
//
//

// intersection_points_utility.cpp
#include "intersection_points_utility.h"
#include "modified_shape_functions/modified_shape_functions.h"
#include "utilities/divide_geometry.h"
#include "droplet_dynamics_application_variables.h"
#include <fstream>
#include "custom_elements/droplet_dynamics_element.h"
#include "droplet_dynamics_application_variables.h"
#include "../../FluidDynamicsApplication/custom_utilities/two_fluid_navier_stokes_data.h"
// AW 14.4
#include "../../LinearSolversApplication/external_libraries/eigen3/Eigen/Dense"
using Eigen::Matrix3d;
using Eigen::Vector3d;




namespace Kratos
{
namespace KratosDropletDynamics
{
    // Define the global container for intersection points
    std::vector<IntersectionPointData> g_IntersectionPointsContainer;

void IntersectionPointsUtility::CollectElementIntersectionPoints(Element::Pointer pElement)
{
    // Get the geometry and distance values from the element
    auto p_geom = pElement->pGetGeometry();
    
    // Only proceed if the element is properly initialized
    if (!p_geom) return;
    
    // Get the distance values from the element's nodes
    Vector nodal_distances;
    nodal_distances.resize(p_geom->size());
    
    for (unsigned int i = 0; i < p_geom->size(); ++i) {
        nodal_distances[i] = (*p_geom)[i].FastGetSolutionStepValue(DISTANCE);
    }

    // Check if the element is actually split by the interface
    bool is_split = false;
    const double sign_threshold = 1e-14;
    int pos_count = 0, neg_count = 0;
    for (unsigned int i = 0; i < p_geom->size(); ++i) {
        if (nodal_distances[i] > sign_threshold) {
            pos_count++;
        } else if (nodal_distances[i] < -sign_threshold) {
            neg_count++;
        }
    }
    
    // Element is split only if it has both positive and negative distance values
    is_split = (pos_count > 0 && neg_count > 0);
    
    // Only proceed with intersection calculations if the element is actually split
    if (!is_split) {
        return; // Skip this element as it's not split by the interface
    }
    
    // Structure nodes info
    Vector structure_node_id = ZeroVector(p_geom->size());
    // for (unsigned int i_node = 0; i_node < p_geom->size(); i_node++) {
    //     if ((*p_geom)[i_node].Is(BOUNDARY)) {
    //         structure_node_id[i_node] = 1.0;
    //     }
    // }
    
    // Create the modified shape functions utility
    ModifiedShapeFunctions::Pointer p_modified_sh_func;
    
    // Create the appropriate modified shape functions based on geometry type
    if (p_geom->GetGeometryType() == GeometryData::KratosGeometryType::Kratos_Triangle2D3) {
        p_modified_sh_func = Kratos::make_shared<Triangle2D3ModifiedShapeFunctions>(p_geom, nodal_distances, structure_node_id);
    } 
    else if (p_geom->GetGeometryType() == GeometryData::KratosGeometryType::Kratos_Tetrahedra3D4) {
        p_modified_sh_func = Kratos::make_shared<Tetrahedra3D4ModifiedShapeFunctions>(p_geom, nodal_distances, structure_node_id);
    }
    
    if (p_modified_sh_func) {
        // Get the splitting utility
        auto p_splitting_util = p_modified_sh_func->pGetSplittingUtil();
        
        if (p_splitting_util) {
            try {
                // Force generation of the intersection skin
                if (p_geom->GetGeometryType() == GeometryData::KratosGeometryType::Kratos_Triangle2D3) {
                    auto p_triangle_splitter = dynamic_cast<DivideTriangle2D3<Node>*>(p_splitting_util.get());
                    if (p_triangle_splitter) {
                        p_triangle_splitter->GenerateIntersectionsSkin();
                        
                        // Extract real intersection points
                        ExtractIntersectionPointsFromSplitter(p_triangle_splitter, pElement->Id());
                    }
                }
                else if (p_geom->GetGeometryType() == GeometryData::KratosGeometryType::Kratos_Tetrahedra3D4) {
                    auto p_tetra_splitter = dynamic_cast<DivideTetrahedra3D4<Node>*>(p_splitting_util.get());
                    if (p_tetra_splitter) {
                        p_tetra_splitter->GenerateIntersectionsSkin();
                        
                        // Extract real intersection points
                        ExtractIntersectionPointsFromSplitter(p_tetra_splitter, pElement->Id());
                    }
                }
                
                std::cout << "Processed interface points for element " << pElement->Id() << std::endl;
            }
            catch (std::exception& e) {
                std::cerr << "Error processing element " << pElement->Id() 
                          << ": " << e.what() << std::endl;
            }
        }
    }
}
    
    void IntersectionPointsUtility::ClearIntersectionPoints()
    {
        g_IntersectionPointsContainer.clear();
    }
    
    const std::vector<IntersectionPointData>& IntersectionPointsUtility::GetIntersectionPoints()
    {
        return g_IntersectionPointsContainer;
    }
    
void IntersectionPointsUtility::SaveIntersectionPointsToFile(const std::string& filename)
{
    std::cout << "Saving " << g_IntersectionPointsContainer.size() << " intersection points to file: " << filename << std::endl;
    
    // Rest of your original function...
    std::ofstream outFile(filename);
    
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    outFile << std::fixed << std::setprecision(15);  
    outFile << "Element_ID\tPoint_ID\tX\tY\tZ" << std::endl;
    
    for (const auto& point : g_IntersectionPointsContainer) {
        outFile << point.elementId << "\t" 
               << point.pointId << "\t"
               << point.coordinates[0] << "\t" 
               << point.coordinates[1] << "\t" 
               << point.coordinates[2] << std::endl;
    }
    
    outFile.close();
    std::cout << "Successfully wrote " << g_IntersectionPointsContainer.size() << " points to " << filename << std::endl;
}

//     void IntersectionPointsUtility::AddIntersectionPoint(int elementId, int pointId, const array_1d<double, 3>& coordinates)
// {
//     IntersectionPointData point;
//     point.elementId = elementId;
//     point.pointId = pointId;
    
//     point.coordinates[0] = coordinates[0];
//     point.coordinates[1] = coordinates[1];
//     point.coordinates[2] = coordinates[2];
    
//     g_IntersectionPointsContainer.push_back(point);
// }

void IntersectionPointsUtility::DiagnosticOutput(const ModelPart& rModelPart)
{
    int total_elements = rModelPart.NumberOfElements();
    int split_elements = 0;
    
    for (auto& element : rModelPart.Elements()) {
        auto p_geom = element.pGetGeometry();
        if (!p_geom) continue;
        
        // Get the distance values
        Vector nodal_distances;
        nodal_distances.resize(p_geom->size());
        for (unsigned int i = 0; i < p_geom->size(); ++i) {
            nodal_distances[i] = (*p_geom)[i].FastGetSolutionStepValue(DISTANCE);
        }
        
        // Check if element is split
        const double sign_threshold = 1e-14;
        int pos_count = 0, neg_count = 0;
        for (unsigned int i = 0; i < p_geom->size(); ++i) {
            if (nodal_distances[i] > sign_threshold) {
                pos_count++;
            } else if (nodal_distances[i] < -sign_threshold) {
                neg_count++;
            }
        }
        
        bool is_split = (pos_count > 0 && neg_count > 0);
        
        if (is_split) {
            split_elements++;
        }
    }
    
    std::cout << "Diagnostic Output:" << std::endl;
    std::cout << "Total Elements: " << total_elements << std::endl;
    std::cout << "Split Elements: " << split_elements << std::endl;
    std::cout << "Points Collected: " << g_IntersectionPointsContainer.size() << std::endl;
}
void IntersectionPointsUtility::ExtractIntersectionPointsFromSplitter(DivideGeometry<Node>* p_splitter, int elementId)
{
    if (!p_splitter) return;
    
    try {
        // Try to get interface points using the GetInterfacePoints method
        auto interface_points = p_splitter->GetInterfacePoints();
        
        if(interface_points.size() > 0) {
            std::cout << "Found " << interface_points.size() << " interface points for element " << elementId << std::endl;
            
            // Add each interface point to our container
            int point_count = 0;
            for (size_t i = 0; i < interface_points.size(); ++i) {
                if (interface_points[i]) {  // Make sure the pointer is valid
                    IntersectionPointData point;
                    point.elementId = elementId;
                    point.pointId = point_count++;
                    
                    // Copy coordinates from the IndexedPoint
                    const auto& coords = interface_points[i]->Coordinates();
                    point.coordinates[0] = coords[0];
                    point.coordinates[1] = coords[1];
                    point.coordinates[2] = coords[2];
                    
                    g_IntersectionPointsContainer.push_back(point);
                }
            }
            
            std::cout << "Added " << point_count << " real intersection points from element " << elementId << std::endl;
        } else {
            std::cout << "No interface points found for element " << elementId << std::endl;
            
    //         // Fallback to test points if no interface points were found
    //         for (int i = 0; i < 2; i++) {
    //             IntersectionPointData point;
    //             point.elementId = elementId;
    //             point.pointId = i;
                
    //             // Sample coordinates
    //             point.coordinates[0] = 0.5 + 0.1*i;  // X
    //             point.coordinates[1] = 0.5 - 0.1*i;  // Y
    //             point.coordinates[2] = 0.0;          // Z (0 for 2D)
                
    //             g_IntersectionPointsContainer.push_back(point);
    //         }
            
    //         std::cout << "Used test points for element " << elementId << " (no interface points found)" << std::endl;
               }
        } catch (std::exception& e) {
        std::cerr << "Error extracting interface points: " << e.what() << std::endl;
        
        // // Fallback to test points if there's an error
        // for (int i = 0; i < 2; i++) {
        //     IntersectionPointData point;
        //     point.elementId = elementId;
        //     point.pointId = i;
            
        //     // Sample coordinates
        //     point.coordinates[0] = 0.5 + 0.1*i;  // X
        //     point.coordinates[1] = 0.5 - 0.1*i;  // Y
        //     point.coordinates[2] = 0.0;          // Z (0 for 2D)
            
        //     g_IntersectionPointsContainer.push_back(point);
        // }
        
        // std::cout << "Fell back to test points for element " << elementId << " due to error: " << e.what() << std::endl;
        }
}
////////////////////////////////////////////////
// void IntersectionPointsUtility::ProcessIntersectionPointsAndFitCurves(const std::string& output_file)
// {
//     // Get all intersection points
//     const auto& points = g_IntersectionPointsContainer;
    
//     if (points.empty()) {
//         std::cout << "No intersection points available for curve fitting." << std::endl;
//         return;
//     }
    
//     std::cout << "Starting circle fitting with " << points.size() << " intersection points." << std::endl;
    
//     // First, create a map of points by their coordinates
//     // This will help us identify which points are shared between elements
//     std::map<std::pair<double, double>, std::vector<int>> point_to_elements;
//     std::map<int, std::vector<IntersectionPointData>> element_points;
    
//     // Group points by element and build point->elements mapping
//     for (const auto& point : points) {
//         int elemId = point.elementId;
        
//         // Round coordinates to handle floating point precision
//         double x = std::round(point.coordinates[0] * 10000000.0) / 10000000.0;
//         double y = std::round(point.coordinates[1] * 10000000.0) / 10000000.0;
//         std::pair<double, double> coord_key(x, y);
        
//         // Add this element to the list for this point
//         point_to_elements[coord_key].push_back(elemId);
        
//         // Add this point to the element's list
//         element_points[elemId].push_back(point);
//     }
    
//     // Find connected element pairs (elements that share intersection points)
//     std::map<int, std::set<int>> element_neighbors;
    
//     for (const auto& [coord, elements] : point_to_elements) {
//         // If this point belongs to multiple elements, they are neighbors
//         for (size_t i = 0; i < elements.size(); ++i) {
//             for (size_t j = i+1; j < elements.size(); ++j) {
//                 int elem1 = elements[i];
//                 int elem2 = elements[j];
                
//                 // Mark as neighbors
//                 element_neighbors[elem1].insert(elem2);
//                 element_neighbors[elem2].insert(elem1);
//             }
//         }
//     }
    
//     // Now, for each element, fit a circle using its points and its neighbors' points
//     struct CircleCoefficients {
//         double a;  // x-center
//         double b;  // y-center
//         double c;  // radius squared
//     };
    
//     std::map<int, CircleCoefficients> elementFits;
    
//     for (const auto& [elemId, neighbors] : element_neighbors) {
//         // Collect all points from this element and its neighbors
//         std::vector<IntersectionPointData> combined_points = element_points[elemId];
        
//         for (int neighborId : neighbors) {
//             // Add neighbor's points
//             combined_points.insert(combined_points.end(), 
//                                  element_points[neighborId].begin(), 
//                                  element_points[neighborId].end());
//         }
        
//         // Remove duplicate points
//         std::map<std::pair<double, double>, IntersectionPointData> unique_points;
//         for (const auto& point : combined_points) {
//             double x = std::round(point.coordinates[0] * 10000000.0) / 10000000.0;
//             double y = std::round(point.coordinates[1] * 10000000.0) / 10000000.0;
//             std::pair<double, double> key(x, y);
//             unique_points[key] = point;
//         }
        
//         // Convert back to vector
//         combined_points.clear();
//         for (const auto& [_, point] : unique_points) {
//             combined_points.push_back(point);
//         }
        
//         // We need at least 3 points to fit a circle
//         if (combined_points.size() >= 3) {
//             // Circle fitting using algebraic approach
//             // For a circle (x-a)^2 + (y-b)^2 = c, we can expand to:
//             // x^2 - 2ax + a^2 + y^2 - 2by + b^2 = c
//             // x^2 + y^2 = 2ax + 2by - a^2 - b^2 + c
//             // x^2 + y^2 = 2ax + 2by + d, where d = -a^2 - b^2 + c
            
//             // Set up matrices for least squares fitting
//             double sum_x = 0.0, sum_y = 0.0;
//             double sum_x2 = 0.0, sum_y2 = 0.0;
//             double sum_xy = 0.0;  // sum of x*y
//             double sum_x2y2 = 0.0;  // sum of (x^2 + y^2)
//             double sum_x3 = 0.0, sum_xy2 = 0.0;
//             double sum_x2y = 0.0, sum_y3 = 0.0;
            
//             for (const auto& point : combined_points) {
//                 double x = point.coordinates[0];
//                 double y = point.coordinates[1];
                
//                 double x2 = x * x;
//                 double y2 = y * y;
                
//                 sum_x += x;
//                 sum_y += y;
//                 sum_x2 += x2;
//                 sum_y2 += y2;
//                 sum_xy += x * y;
//                 sum_x2y2 += (x2 + y2);
//                 sum_x3 += x * x2;
//                 sum_xy2 += x * y2;
//                 sum_x2y += x2 * y;
//                 sum_y3 += y * y2;
//             }
            
//             int n = combined_points.size();
            
//             // Create the system of equations
//             Matrix A(3, 3);
//             Vector b(3);
            
//             A(0, 0) = sum_x2;    A(0, 1) = sum_xy;    A(0, 2) = sum_x;
//             A(1, 0) = sum_xy;    A(1, 1) = sum_y2;    A(1, 2) = sum_y;
//             A(2, 0) = sum_x;     A(2, 1) = sum_y;     A(2, 2) = n;
            
//             // For equation: x^2 + y^2 = 2ax + 2by + d
//             // The right side is x^2 + y^2
//             b[0] = sum_x3 + sum_xy2;  // sum of x * (x^2 + y^2)
//             b[1] = sum_x2y + sum_y3;  // sum of y * (x^2 + y^2)
//             b[2] = sum_x2y2;          // sum of (x^2 + y^2)
            
//             // Solve using Cramer's rule
//             double det = A(0, 0) * (A(1, 1) * A(2, 2) - A(2, 1) * A(1, 2)) -
//                          A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) +
//                          A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));

//             // No early termination for small determinants
            
//             Matrix A1 = A, A2 = A, A3 = A;
            
//             for (int i = 0; i < 3; i++) {
//                 A1(i, 0) = b[i];
//                 A2(i, 1) = b[i];
//                 A3(i, 2) = b[i];
//             }
            
//             double det1 = A1(0, 0) * (A1(1, 1) * A1(2, 2) - A1(2, 1) * A1(1, 2)) -
//                           A1(0, 1) * (A1(1, 0) * A1(2, 2) - A1(1, 2) * A1(2, 0)) +
//                           A1(0, 2) * (A1(1, 0) * A1(2, 1) - A1(1, 1) * A1(2, 0));
                          
//             double det2 = A2(0, 0) * (A2(1, 1) * A2(2, 2) - A2(2, 1) * A2(1, 2)) -
//                           A2(0, 1) * (A2(1, 0) * A2(2, 2) - A2(1, 2) * A2(2, 0)) +
//                           A2(0, 2) * (A2(1, 0) * A2(2, 1) - A2(1, 1) * A2(2, 0));
                          
//             double det3 = A3(0, 0) * (A3(1, 1) * A3(2, 2) - A3(2, 1) * A3(1, 2)) -
//                           A3(0, 1) * (A3(1, 0) * A3(2, 2) - A3(1, 2) * A3(2, 0)) +
//                           A3(0, 2) * (A3(1, 0) * A3(2, 1) - A3(1, 1) * A3(2, 0));
            
//             // Solve for parameters in the form 2ax + 2by + d = x^2 + y^2
//             double twoA = det1 / det;
//             double twoB = det2 / det;
//             double d = det3 / det;
            
//             CircleCoefficients fit;
//             fit.a = twoA / 2.0;  // center x-coordinate
//             fit.b = twoB / 2.0;  // center y-coordinate
            
//             // Calculate radius squared (c)
//             // From d = -a^2 - b^2 + c, we get:
//             // c = d + a^2 + b^2
//             fit.c = d + fit.a * fit.a + fit.b * fit.b;
            
//             // Store the fit
//             elementFits[elemId] = fit;
            
//             // Calculate the actual radius for display
//             double radius = std::sqrt(fit.c);
            
//             std::cout << "Element " << elemId 
//                       << " with " << combined_points.size() 
//                       << " points (including neighbors): (x-" << fit.a 
//                       << ")² + (y-" << fit.b << ")² = " << fit.c 
//                       << " (radius = " << radius << ")" << std::endl;
//         } else {
//             std::cout << "Element " << elemId 
//                       << " still has only " << combined_points.size() 
//                       << " unique points (less than 3) - cannot fit circle." << std::endl;
//         }
//     }
    
//     // Save results to file
//     std::ofstream outFile(output_file);
    
//     if (!outFile.is_open()) {
//         std::cerr << "Error: Could not open file " << output_file << " for writing." << std::endl;
//         return;
//     }
    
//     outFile << "Element_ID\tNum_Points\ta(center_x)\tb(center_y)\tc(radius_squared)\tradius\n";
    
//     for (const auto& fit_pair : elementFits) {
//         int elemId = fit_pair.first;
//         const auto& fit = fit_pair.second;
//         int numPoints = element_points[elemId].size();
//         double radius = std::sqrt(fit.c);
        
//         outFile << elemId << "\t" 
//                 << numPoints << "\t"
//                 << fit.a << "\t" 
//                 << fit.b << "\t" 
//                 << fit.c << "\t"
//                 << radius << "\n";
//     }
    
//     outFile.close();
    
//     std::cout << "Saved " << elementFits.size() << " element circle fits to " << output_file << std::endl;
// }
/////////////////////////////////////////////////
// void IntersectionPointsUtility::ProcessIntersectionPointsAndFitCurves(const std::string& output_file)
// {
//     // Get all intersection points
//     const auto& points = g_IntersectionPointsContainer;
    
//     if (points.empty()) {
//         std::cout << "No intersection points available for circle fitting." << std::endl;
//         return;
//     }
    
//     // Configuration parameters
//     const int MIN_POINTS_FOR_CIRCLE_FIT = 3;  // Absolute minimum needed for circle
//     const int TARGET_POINTS = 6;              // Target number of points for each element
//     const int NEIGHBOR_EXPANSION_LEVEL = 3;   // Expand to n-hop neighbors
    
//     std::cout << "Starting circle fitting with " << points.size() << " intersection points." << std::endl;
//     std::cout << "Using exactly " << TARGET_POINTS << " points per element where possible." << std::endl;
    
//     // Group points by element
//     std::map<int, std::vector<IntersectionPointData>> element_points;
//     // Create a map of points by their coordinates
//     std::map<std::pair<double, double>, std::vector<int>> point_to_elements;
    
//     for (const auto& point : points) {
//         int elemId = point.elementId;
        
//         // Round coordinates to handle floating point precision
//         double x = std::round(point.coordinates[0] * 10000000.0) / 10000000.0;
//         double y = std::round(point.coordinates[1] * 10000000.0) / 10000000.0;
//         std::pair<double, double> coord_key(x, y);
        
//         // Add this element to the list for this point
//         point_to_elements[coord_key].push_back(elemId);
        
//         // Add this point to the element's list
//         element_points[elemId].push_back(point);
//     }
    
//     // Find element neighbors (elements that share intersection points)
//     std::map<int, std::set<int>> element_neighbors;
    
//     for (const auto& [coord, elements] : point_to_elements) {
//         // If this point belongs to multiple elements, they are neighbors
//         for (size_t i = 0; i < elements.size(); ++i) {
//             for (size_t j = i+1; j < elements.size(); ++j) {
//                 element_neighbors[elements[i]].insert(elements[j]);
//                 element_neighbors[elements[j]].insert(elements[i]);
//             }
//         }
//     }
    
//     // Expand the neighborhood to n-hop neighbors
//     std::cout << "Expanding neighborhood with " << NEIGHBOR_EXPANSION_LEVEL << " hops..." << std::endl;
//     std::map<int, std::set<int>> expanded_neighbors = element_neighbors;
    
//     for (int hop = 2; hop <= NEIGHBOR_EXPANSION_LEVEL; hop++) {
//         std::map<int, std::set<int>> next_level_neighbors = expanded_neighbors;
        
//         for (const auto& [elemId, current_neighbors] : expanded_neighbors) {
//             for (int neighbor : current_neighbors) {
//                 for (int next_hop : expanded_neighbors[neighbor]) {
//                     if (next_hop != elemId && !expanded_neighbors[elemId].count(next_hop)) {
//                         next_level_neighbors[elemId].insert(next_hop);
//                     }
//                 }
//             }
//         }
        
//         expanded_neighbors = next_level_neighbors;
//         std::cout << "Completed " << hop << "-hop neighborhood expansion." << std::endl;
//     }
    
//     // Structure to hold circle fit coefficients
//     struct CircleCoefficients {
//         double a;  // x-center
//         double b;  // y-center
//         double c;  // radius squared
//     };
    
//     // Maps to store results
//     std::map<int, CircleCoefficients> elementFits;
//     std::map<int, int> elementTotalPoints;
    
//     // Process each element
//     for (const auto& [elemId, neighbors] : expanded_neighbors) {
//         // Get original points for this element
//         std::vector<IntersectionPointData> original_points = element_points[elemId];
//         int original_point_count = original_points.size();
        
//         // Create a pool of neighbor points
//         std::vector<IntersectionPointData> neighbor_points;
//         for (int neighborId : neighbors) {
//             neighbor_points.insert(neighbor_points.end(), 
//                                  element_points[neighborId].begin(), 
//                                  element_points[neighborId].end());
//         }
        
//         // Remove duplicates and points shared with original set
//         std::map<std::pair<double, double>, IntersectionPointData> unique_neighbor_points;
//         for (const auto& point : neighbor_points) {
//             double x = std::round(point.coordinates[0] * 10000000.0) / 10000000.0;
//             double y = std::round(point.coordinates[1] * 10000000.0) / 10000000.0;
//             std::pair<double, double> key(x, y);
            
//             // Skip points that are in the original set
//             bool is_in_original = false;
//             for (const auto& orig_point : original_points) {
//                 double ox = std::round(orig_point.coordinates[0] * 10000000.0) / 10000000.0;
//                 double oy = std::round(orig_point.coordinates[1] * 10000000.0) / 10000000.0;
//                 if (ox == x && oy == y) {
//                     is_in_original = true;
//                     break;
//                 }
//             }
            
//             if (!is_in_original) {
//                 unique_neighbor_points[key] = point;
//             }
//         }
        
//         // Create a vector of unique neighbor points
//         neighbor_points.clear();
//         for (const auto& [_, point] : unique_neighbor_points) {
//             neighbor_points.push_back(point);
//         }
        
//         // Build the set of points for circle fitting
//         std::vector<IntersectionPointData> combined_points = original_points;
        
//         // Add only enough points to reach the target
//         int points_to_take = std::min((int)neighbor_points.size(), 
//                                      TARGET_POINTS - original_point_count);
        
//         for (int i = 0; i < points_to_take; i++) {
//             combined_points.push_back(neighbor_points[i]);
//         }
        
//         int points_from_neighbors = points_to_take;
        
//         // Only fit if we have enough points
//         if (combined_points.size() >= MIN_POINTS_FOR_CIRCLE_FIT) {
//             std::cout << "Element " << elemId 
//                       << " has exactly " << combined_points.size() 
//                       << " points for circle fitting (" 
//                       << original_point_count << " original + " 
//                       << points_from_neighbors << " from neighbors)." << std::endl;
            
//             // Prepare matrices for least squares fitting
//             double sum_x = 0.0, sum_y = 0.0;
//             double sum_x2 = 0.0, sum_y2 = 0.0;
//             double sum_xy = 0.0;
//             double sum_x2y2 = 0.0;  // sum of (x^2 + y^2)
//             double sum_x3 = 0.0, sum_xy2 = 0.0;
//             double sum_x2y = 0.0, sum_y3 = 0.0;
            
//             for (const auto& point : combined_points) {
//                 double x = point.coordinates[0];
//                 double y = point.coordinates[1];
                
//                 double x2 = x * x;
//                 double y2 = y * y;
                
//                 sum_x += x;
//                 sum_y += y;
//                 sum_x2 += x2;
//                 sum_y2 += y2;
//                 sum_xy += x * y;
//                 sum_x2y2 += (x2 + y2);
//                 sum_x3 += x * x2;
//                 sum_xy2 += x * y2;
//                 sum_x2y += x2 * y;
//                 sum_y3 += y * y2;
//             }
            
//             int n = combined_points.size();
            
//             // Set up the system of equations: x^2 + y^2 = 2ax + 2by + d
//             Matrix A(3, 3);
//             Vector b(3);
            
//             A(0, 0) = sum_x2;    A(0, 1) = sum_xy;    A(0, 2) = sum_x;
//             A(1, 0) = sum_xy;    A(1, 1) = sum_y2;    A(1, 2) = sum_y;
//             A(2, 0) = sum_x;     A(2, 1) = sum_y;     A(2, 2) = n;
            
//             b[0] = sum_x3 + sum_xy2;  // sum of x * (x^2 + y^2)
//             b[1] = sum_x2y + sum_y3;  // sum of y * (x^2 + y^2)
//             b[2] = sum_x2y2;          // sum of (x^2 + y^2)
            
//             // Solve using Cramer's rule
//             double det = A(0, 0) * (A(1, 1) * A(2, 2) - A(2, 1) * A(1, 2)) -
//                          A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) +
//                          A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
            
//             Matrix A1 = A, A2 = A, A3 = A;
            
//             for (int i = 0; i < 3; i++) {
//                 A1(i, 0) = b[i];
//                 A2(i, 1) = b[i];
//                 A3(i, 2) = b[i];
//             }
            
//             double det1 = A1(0, 0) * (A1(1, 1) * A1(2, 2) - A1(2, 1) * A1(1, 2)) -
//                           A1(0, 1) * (A1(1, 0) * A1(2, 2) - A1(1, 2) * A1(2, 0)) +
//                           A1(0, 2) * (A1(1, 0) * A1(2, 1) - A1(1, 1) * A1(2, 0));
                          
//             double det2 = A2(0, 0) * (A2(1, 1) * A2(2, 2) - A2(2, 1) * A2(1, 2)) -
//                           A2(0, 1) * (A2(1, 0) * A2(2, 2) - A2(1, 2) * A2(2, 0)) +
//                           A2(0, 2) * (A2(1, 0) * A2(2, 1) - A2(1, 1) * A2(2, 0));
                          
//             double det3 = A3(0, 0) * (A3(1, 1) * A3(2, 2) - A3(2, 1) * A3(1, 2)) -
//                           A3(0, 1) * (A3(1, 0) * A3(2, 2) - A3(1, 2) * A3(2, 0)) +
//                           A3(0, 2) * (A3(1, 0) * A3(2, 1) - A3(1, 1) * A3(2, 0));
            
//             // Solve for parameters
//             double twoA = det1 / det;
//             double twoB = det2 / det;
//             double d = det3 / det;
            
//             CircleCoefficients fit;
//             fit.a = twoA / 2.0;  // x-center
//             fit.b = twoB / 2.0;  // y-center
//             fit.c = d + fit.a * fit.a + fit.b * fit.b;  // radius squared
            
//             // Store results
//             elementFits[elemId] = fit;
//             elementTotalPoints[elemId] = combined_points.size();
            
//             // Calculate error on original points
//             double radius = std::sqrt(fit.c);
//             double total_error = 0.0;
            
//             for (const auto& point : original_points) {
//                 double x = point.coordinates[0];
//                 double y = point.coordinates[1];
//                 double dist_squared = (x - fit.a) * (x - fit.a) + (y - fit.b) * (y - fit.b);
//                 total_error += std::abs(dist_squared - fit.c);
//             }
            
//             double avg_error = total_error / (original_points.empty() ? 1.0 : original_points.size());
//             double reliability = std::min(1.0, (double)combined_points.size() / 6.0);
            
//             std::cout << "Element " << elemId 
//                       << " fitted with " << combined_points.size() 
//                       << " points: (x-" << fit.a 
//                       << ")² + (y-" << fit.b << ")² = " << fit.c 
//                       << " (radius = " << radius 
//                       << ", reliability = " << std::fixed << std::setprecision(2) << reliability * 100.0 << "%)" 
//                       << std::endl;
//             std::cout << "    Average fit error on original points: " << avg_error << std::endl;
//         } else {
//             std::cout << "Element " << elemId 
//                       << " has only " << combined_points.size() 
//                       << " unique points (less than " << MIN_POINTS_FOR_CIRCLE_FIT 
//                       << " required) - cannot perform circle fitting." << std::endl;
//         }
//     }
    
//     // Write results to file
//     std::ofstream outFile(output_file);
    
//     if (!outFile.is_open()) {
//         std::cerr << "Error: Could not open file " << output_file << " for writing." << std::endl;
//         return;
//     }
    
//     outFile << "Element_ID\tNum_Original_Points\tTotal_Points\ta(center_x)\tb(center_y)\tc(radius_squared)\tradius\tavg_error\treliability\n";
    
//     for (const auto& [elemId, fit] : elementFits) {
//         int numPoints = element_points[elemId].size();
//         int totalPoints = elementTotalPoints[elemId];
//         double radius = std::sqrt(fit.c);
        
//         // Calculate error
//         double total_error = 0.0;
//         for (const auto& point : element_points[elemId]) {
//             double x = point.coordinates[0];
//             double y = point.coordinates[1];
//             double dist_squared = (x - fit.a) * (x - fit.a) + (y - fit.b) * (y - fit.b);
//             total_error += std::abs(dist_squared - fit.c);
//         }
        
//         double avg_error = total_error / (numPoints > 0 ? numPoints : 1.0);
//         double reliability = std::min(1.0, (double)totalPoints / 6.0);
        
//         outFile << elemId << "\t" 
//                 << numPoints << "\t"
//                 << totalPoints << "\t"
//                 << fit.a << "\t" 
//                 << fit.b << "\t" 
//                 << fit.c << "\t"
//                 << radius << "\t"
//                 << avg_error << "\t"
//                 << reliability << "\n";
//     }
    
//     outFile.close();
    
//     std::cout << "Saved " << elementFits.size() << " element circle fits to " << output_file << std::endl;
//     std::cout << "Each element used exactly " << TARGET_POINTS << " points where possible." << std::endl;
// }

// includes the <iomanip> header which allows manipulators like std::setprecision, which control the number of decimal places for storing the output results in files
#include <iomanip>  // For std::setprecision


void IntersectionPointsUtility::ProcessIntersectionPointsAndFitCurves(const std::string& output_file)
{
    // Get all intersection points 
    // grabs a reference to the global container g_IntersectionPointsContainer which stores all interface (cut) points
    const auto& points = g_IntersectionPointsContainer;
    
    // debug message in case the intersection points container is empty
    if (points.empty()) {
        std::cout << "No intersection points available for circle fitting." << std::endl;
        return;
    }
    
    // Configuration parameters
    // AW 14.4: neighbourhood expansion changed to 3
    const int MIN_POINTS_FOR_CIRCLE_FIT = 3;  // Absolute minimum needed for circle
    const int NEIGHBOR_EXPANSION_LEVEL = 3;   // Expand to n-hop neighbors
    
    std::cout << "Starting circle fitting with " << points.size() << " intersection points." << std::endl;
    std::cout << "Using all available points from 2-hop neighborhoods." << std::endl;
    
    // Group points by element
    std::map<int, std::vector<IntersectionPointData>> element_points;
    // Create a map of points by their coordinates
    std::map<std::pair<double, double>, std::vector<int>> point_to_elements;
    
    for (const auto& point : points) {
        int elemId = point.elementId;
        
        // Round coordinates to handle floating point precision
        double x = std::round(point.coordinates[0] * 1.0E14) / 1.0E14;
        double y = std::round(point.coordinates[1] * 1.0E14) / 1.0E14;
        std::pair<double, double> coord_key(x, y);
        
        // Add this element to the list for this point
        point_to_elements[coord_key].push_back(elemId);
        
        // Add this point to the element's list
        element_points[elemId].push_back(point);
    }
    
    // Find element neighbors (elements that share intersection points)
    std::map<int, std::set<int>> element_neighbors;
    
    for (const auto& [coord, elements] : point_to_elements) {
        // If this point belongs to multiple elements, they are neighbors
        for (size_t i = 0; i < elements.size(); ++i) {
            for (size_t j = i+1; j < elements.size(); ++j) {
                element_neighbors[elements[i]].insert(elements[j]);
                element_neighbors[elements[j]].insert(elements[i]);
            }
        }
    }
    
    // Expand the neighborhood to n-hop neighbors
    std::cout << "Expanding neighborhood with " << NEIGHBOR_EXPANSION_LEVEL << " hops..." << std::endl;
    std::map<int, std::set<int>> expanded_neighbors = element_neighbors;
    
    for (int hop = 2; hop <= NEIGHBOR_EXPANSION_LEVEL; hop++) {
        std::map<int, std::set<int>> next_level_neighbors = expanded_neighbors;
        
        for (const auto& [elemId, current_neighbors] : expanded_neighbors) {
            for (int neighbor : current_neighbors) {
                // for (int next_hop : expanded_neighbors[neighbor]) {
                for (int next_hop : element_neighbors[neighbor]) {
                    if (next_hop != elemId && !expanded_neighbors[elemId].count(next_hop)) {
                        next_level_neighbors[elemId].insert(next_hop);
                    }
                }
            }
        }
        
        expanded_neighbors = next_level_neighbors;
        std::cout << "Completed " << hop << "-hop neighborhood expansion." << std::endl;
    }
    
    // Structure to hold circle fit coefficients
    struct CircleCoefficients {
        double a;  // x-center
        double b;  // y-center
        double c;  // radius squared
    };
    
    // Maps to store results
    std::map<int, CircleCoefficients> elementFits;
    std::map<int, int> elementTotalPoints;
    
    // Process each element
    for (const auto& [elemId, neighbors] : expanded_neighbors) {
        // Get original points for this element
        std::vector<IntersectionPointData> original_points = element_points[elemId];
        int original_point_count = original_points.size();
        
        // Create a pool of neighbor points
        std::vector<IntersectionPointData> neighbor_points;
        for (int neighborId : neighbors) {
            neighbor_points.insert(neighbor_points.end(), 
                                 element_points[neighborId].begin(), 
                                 element_points[neighborId].end());
        }
        
        // Remove duplicates and points shared with original set
        std::map<std::pair<double, double>, IntersectionPointData> unique_neighbor_points;
        for (const auto& point : neighbor_points) {
            double x = std::round(point.coordinates[0] * 1.0E14) / 1.0E14;
            double y = std::round(point.coordinates[1] * 1.0E14) / 1.0E14;
            std::pair<double, double> key(x, y);
            
            // Skip points that are in the original set
            bool is_in_original = false;
            for (const auto& orig_point : original_points) {
                double ox = std::round(orig_point.coordinates[0] * 1.0E14) / 1.0E14;
                double oy = std::round(orig_point.coordinates[1] * 1.0E14) / 1.0E14;
                if (ox == x && oy == y) {
                    is_in_original = true;
                    break;
                }
            }
            
            if (!is_in_original) {
                unique_neighbor_points[key] = point;
            }
        }
        
        // Create a vector of unique neighbor points
        neighbor_points.clear();
        for (const auto& [_, point] : unique_neighbor_points) {
            neighbor_points.push_back(point);
        }
        
        // Build the set of points for circle fitting
        std::vector<IntersectionPointData> combined_points = original_points;
        
        // Add all neighbor points
        combined_points.insert(combined_points.end(), neighbor_points.begin(), neighbor_points.end());
        
        int points_from_neighbors = neighbor_points.size();
        
        // Only fit if we have enough points
        if (combined_points.size() >= MIN_POINTS_FOR_CIRCLE_FIT) {
            std::cout << "Element " << elemId 
                      << " has " << combined_points.size() 
                      << " points for circle fitting (" 
                      << original_point_count << " original + " 
                      << points_from_neighbors << " from neighbors)." << std::endl;
            
            // Prepare matrices for least squares fitting
            double sum_x = 0.0, sum_y = 0.0;
            double sum_x2 = 0.0, sum_y2 = 0.0;
            double sum_xy = 0.0;
            double sum_x2y2 = 0.0;  // sum of (x^2 + y^2)
            double sum_x3 = 0.0, sum_xy2 = 0.0;
            double sum_x2y = 0.0, sum_y3 = 0.0;
            
            for (const auto& point : combined_points) {
                double x = point.coordinates[0];
                double y = point.coordinates[1];
                
                double x2 = x * x;
                double y2 = y * y;
                
                sum_x += x;
                sum_y += y;
                sum_x2 += x2;
                sum_y2 += y2;
                sum_xy += x * y;
                sum_x2y2 += (x2 + y2);
                sum_x3 += x * x2;
                sum_xy2 += x * y2;
                sum_x2y += x2 * y;
                sum_y3 += y * y2;
            }
            
            int n = combined_points.size();
            
            // Set up the system of equations: x^2 + y^2 = 2ax + 2by + d
            Matrix A(3, 3);
            Vector b(3);
            
            A(0, 0) = sum_x2;    A(0, 1) = sum_xy;    A(0, 2) = sum_x;
            A(1, 0) = sum_xy;    A(1, 1) = sum_y2;    A(1, 2) = sum_y;
            A(2, 0) = sum_x;     A(2, 1) = sum_y;     A(2, 2) = n;
            
            b[0] = sum_x3 + sum_xy2;  // sum of x * (x^2 + y^2)
            b[1] = sum_x2y + sum_y3;  // sum of y * (x^2 + y^2)
            b[2] = sum_x2y2;          // sum of (x^2 + y^2)
            
            // Solve using Cramer's rule
            double det = A(0, 0) * (A(1, 1) * A(2, 2) - A(2, 1) * A(1, 2)) -
                         A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) +
                         A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
            
            Matrix A1 = A, A2 = A, A3 = A;
            
            for (int i = 0; i < 3; i++) {
                A1(i, 0) = b[i];
                A2(i, 1) = b[i];
                A3(i, 2) = b[i];
            }
            
            double det1 = A1(0, 0) * (A1(1, 1) * A1(2, 2) - A1(2, 1) * A1(1, 2)) -
                          A1(0, 1) * (A1(1, 0) * A1(2, 2) - A1(1, 2) * A1(2, 0)) +
                          A1(0, 2) * (A1(1, 0) * A1(2, 1) - A1(1, 1) * A1(2, 0));
                          
            double det2 = A2(0, 0) * (A2(1, 1) * A2(2, 2) - A2(2, 1) * A2(1, 2)) -
                          A2(0, 1) * (A2(1, 0) * A2(2, 2) - A2(1, 2) * A2(2, 0)) +
                          A2(0, 2) * (A2(1, 0) * A2(2, 1) - A2(1, 1) * A2(2, 0));
                          
            double det3 = A3(0, 0) * (A3(1, 1) * A3(2, 2) - A3(2, 1) * A3(1, 2)) -
                          A3(0, 1) * (A3(1, 0) * A3(2, 2) - A3(1, 2) * A3(2, 0)) +
                          A3(0, 2) * (A3(1, 0) * A3(2, 1) - A3(1, 1) * A3(2, 0));
            
            // Solve for parameters
            double twoA = det1 / det;
            double twoB = det2 / det;
            double d = det3 / det;
            
            CircleCoefficients fit;
            fit.a = twoA / 2.0;  // x-center
            fit.b = twoB / 2.0;  // y-center
            fit.c = d + fit.a * fit.a + fit.b * fit.b;  // radius squared
            
            // Store results
            elementFits[elemId] = fit;
            elementTotalPoints[elemId] = combined_points.size();
            
            // Calculate error on original points
            double radius = std::sqrt(fit.c);
            double total_error = 0.0;
            
            for (const auto& point : original_points) {
                double x = point.coordinates[0];
                double y = point.coordinates[1];
                double dist_squared = (x - fit.a) * (x - fit.a) + (y - fit.b) * (y - fit.b);
                total_error += std::abs(dist_squared - fit.c);
            }
            
            double avg_error = total_error / (original_points.empty() ? 1.0 : original_points.size());
            
            std::cout << "Element " << elemId 
                      << " fitted with " << combined_points.size() 
                      << " points: (x-" << fit.a 
                      << ")² + (y-" << fit.b << ")² = " << fit.c 
                      << " (radius = " << radius << ")" 
                      << std::endl;
            std::cout << "    Average fit error on original points: " << avg_error << std::endl;
        } else {
            std::cout << "Element " << elemId 
                      << " has only " << combined_points.size() 
                      << " unique points (less than " << MIN_POINTS_FOR_CIRCLE_FIT 
                      << " required) - cannot perform circle fitting." << std::endl;
        }
    }
    
    // Write results to file
    std::ofstream outFile(output_file);
    
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << output_file << " for writing." << std::endl;
        return;
    }
    
    outFile << "Element_ID\tNum_Original_Points\tTotal_Points\ta(center_x)\tb(center_y)\tc(radius_squared)\tradius\tavg_error\n";
    
    for (const auto& [elemId, fit] : elementFits) {
        int numPoints = element_points[elemId].size();
        int totalPoints = elementTotalPoints[elemId];
        double radius = std::sqrt(fit.c);
        
        // Calculate error
        double total_error = 0.0;
        for (const auto& point : element_points[elemId]) {
            double x = point.coordinates[0];
            double y = point.coordinates[1];
            double dist_squared = (x - fit.a) * (x - fit.a) + (y - fit.b) * (y - fit.b);
            total_error += std::abs(dist_squared - fit.c);
        }
        
        double avg_error = total_error / (numPoints > 0 ? numPoints : 1.0);
        
        outFile << elemId << "\t" 
                << numPoints << "\t"
                << totalPoints << "\t"
                << fit.a << "\t" 
                << fit.b << "\t" 
                << fit.c << "\t"
                << radius << "\t"
                << avg_error << "\n";
    }
    
    outFile.close();
    
    std::cout << "Saved " << elementFits.size() << " element circle fits to " << output_file << std::endl;
    std::cout << "Used all available points from 2-hop neighborhoods." << std::endl;
}

// method ProcessIntersectionPointsAndFitCurves, a static member function of the IntersectionPointsUtility class
// only input file: the output file where the intersection points are written into
void IntersectionPointsUtility::ProcessIntersectionPointsAndFitCurvesparabola(const std::string& output_file)
{
    // Get all intersection points 
    // grabs a reference to the global container g_IntersectionPointsContainer which stores all interface (cut) points
    const auto& points = g_IntersectionPointsContainer;
    
    // debug message in case the intersection points container is empty
    if (points.empty()) {
        std::cout << "No intersection points available for curve fitting." << std::endl;
        return;
    }
    
    // Configuration parameters
    // AW 14.4: neighbourhood expansion changed to 6
    const int MIN_POINTS_FOR_CURVE_FIT = 3;  // Absolute minimum needed for quadratic fit
    const int NEIGHBOR_EXPANSION_LEVEL = 6;   // Expand to n-hop neighbors
    
    // debug message pointing out the start of the quadratic curve fitting process
    std::cout << "Starting quadratic curve fitting with " << points.size() << " intersection points." << std::endl;
    std::cout << "Using all available points from 2-hop neighborhoods." << std::endl;
    
    // Group points by element
    // element_points maps element IDs to a list of IntersectionPointData objects
    std::map<int, std::vector<IntersectionPointData>> element_points;
    // Create a map of points by their coordinates
    // point_to_elements maps each (x, y) coordinate pair to the list of element IDs that contain that point
    std::map<std::pair<double, double>, std::vector<int>> point_to_elements;
    
    // Iterates over each intersection point in the global container points which contains the intersection points per element
    for (const auto& point : points) {
        // Get the element ID of the point
        int elemId = point.elementId;
        
        // Round coordinates to handle floating point precision
        double x = std::round(point.coordinates[0] * 1.0E14) / 1.0E14;
        double y = std::round(point.coordinates[1] * 1.0E14) / 1.0E14;
        std::pair<double, double> coord_key(x, y);
        
        // Add this element to the list for this point
        // reverse map lets us quickly find which elements share a given coordinate
        point_to_elements[coord_key].push_back(elemId);
        
        // Add this point to the element's list
        // forward map stores this point in the list of intersection points for the current elemen
        element_points[elemId].push_back(point);
    }
    
    // Find element neighbors (elements that share intersection points)
    // map will hold, for each element ID, the set of neighboring element IDs
    std::map<int, std::set<int>> element_neighbors;
    
    // Loops over each coordinate key and its corresponding list of element IDs from the reverse map
    for (const auto& [coord, elements] : point_to_elements) {
        // If this point belongs to multiple elements, they are neighbors
        for (size_t i = 0; i < elements.size(); ++i) {
            for (size_t j = i+1; j < elements.size(); ++j) {
                // Adds a bidirectional connection between the two neighboring elements
                element_neighbors[elements[i]].insert(elements[j]);
                element_neighbors[elements[j]].insert(elements[i]);
            }
        }
    }
    
    // Expand the neighborhood to n-hop neighbors
    std::cout << "Expanding neighborhood with " << NEIGHBOR_EXPANSION_LEVEL << " hops..." << std::endl;
    std::map<int, std::set<int>> expanded_neighbors = element_neighbors;
    
    // This loop performs successive expansions
    // On each hop, it builds up a larger neighbor set
    for (int hop = 2; hop <= NEIGHBOR_EXPANSION_LEVEL; hop++) {
        std::map<int, std::set<int>> next_level_neighbors = expanded_neighbors;
        // Loops through all current neighbors of an element
        // For each such neighbor, it looks up that neighbor’s own neighbors in the original 1-hop element_neighbors
        for (const auto& [elemId, current_neighbors] : expanded_neighbors) {
            for (int neighbor : current_neighbors) {
                // Gets the neighbors of that neighbor
                for (int next_hop : element_neighbors[neighbor]) {
                    if (next_hop != elemId && !expanded_neighbors[elemId].count(next_hop)) {
                        // If the new neighbor is not already known and isn't the element itself, add it
                        next_level_neighbors[elemId].insert(next_hop);
                    }
                }
            }
        }
        
        // Replace the current neighbor set with the expanded one
        expanded_neighbors = next_level_neighbors;
        std::cout << "Completed " << hop << "-hop neighborhood expansion." << std::endl;
    }
    
    // Structure to hold quadratic curve fit coefficients (y = ax² + bx + c)
    struct QuadraticCoefficients {
        double a;  // coefficient of x²
        double b;  // coefficient of x
        double c;  // constant term
    };
    
    // Maps to store results
    // Associates each element ID (int) with its fitted quadratic curve coefficients
    std::map<int, QuadraticCoefficients> elementFits;
    // Stores the number of total points (original + neighbor) used per element for fitting
    std::map<int, int> elementTotalPoints;
    // Stores whether a given element’s curve was fit in rotated coordinates
    std::map<int, bool> elementWasRotated;
    // AW 18.4: added this map to avoid computing the points in several loops; instead, just once per element and store in here
    // Stores the full set of (x, y) coordinates used for fitting per element (after neighbor expansion)
    std::map<int, std::vector<std::pair<double, double>>> elementFitPoints;

    // Process each element
    // loop iterates over all elements and their neighbours;
    // hence, everything within this loop is done for every element
    for (const auto& [elemId, neighbors] : expanded_neighbors) {
        // Retrieve the list of intersection points belonging to this specific element
        std::vector<IntersectionPointData> original_points = element_points[elemId];
        // also count how many original points this element has, before adding neighbors
        int original_point_count = original_points.size();
        
        // Create a pool of neighbor points; declares a vector to hold all points from neighboring elements
        std::vector<IntersectionPointData> neighbor_points;
        // For each neighboring element, access its associated intersection points 
        for (int neighborId : neighbors) {
            // builds a complete neighborhood point cloud for curve fitting around elemId
            neighbor_points.insert(neighbor_points.end(), 
                                 element_points[neighborId].begin(), 
                                 element_points[neighborId].end());
        }
        
        // Remove duplicates and points shared with original set
        std::map<std::pair<double, double>, IntersectionPointData> unique_neighbor_points;
        // Loop over all previously collected neighboring points
        for (const auto& point : neighbor_points) {
            // creates a consistent key so that two points that are numerically close but not bitwise identical are treated as the same
            double x = std::round(point.coordinates[0] * 1.0E14) / 1.0E14;
            double y = std::round(point.coordinates[1] * 1.0E14) / 1.0E14;
            std::pair<double, double> key(x, y);
            
            // Skip points that are in the original set
            // Loop through all original points and round them the same way as the neighbor points for comparison
            bool is_in_original = false;
            for (const auto& orig_point : original_points) {
                double ox = std::round(orig_point.coordinates[0] * 1.0E14) / 1.0E14;
                double oy = std::round(orig_point.coordinates[1] * 1.0E14) / 1.0E14;
                // AW 14.4: Tolerance-based comparison to not consider points that are extremely close to each other
                if (std::abs(ox - x) < 1e-10 && std::abs(oy - y) < 1e-10) { 
                    is_in_original = true;
                    break;
                }
            }
            
            // If the current neighbor point was not found in the original points (is_in_original == false), we add it to unique_neighbor_points
            if (!is_in_original) {
                unique_neighbor_points[key] = point;
            }
        // End of the loop over neighbor_points
        }
        
        // Create a vector of unique neighbor points
        // now replace the old neighbor_points list with the filtered version
        neighbor_points.clear();
        for (const auto& [_, point] : unique_neighbor_points) {
            neighbor_points.push_back(point);
        }
        
        // Build the set of points for curve fitting
        // construct the full set of points to be used for curve fitting (starting with the original point set)
        std::vector<IntersectionPointData> combined_points = original_points;
        
        // Add all neighbor points
        combined_points.insert(combined_points.end(), neighbor_points.begin(), neighbor_points.end());
        
        // Stores how many points came specifically from neighboring elements 
        int points_from_neighbors = neighbor_points.size();

        // AW 18.4: store points in the map per element
        // Store the raw coordinates used for fitting (before rotation/scaling) for this element
        // will be overwritten for rotated points
        std::vector<std::pair<double, double>> fit_coords;
        for (const auto& point : combined_points) {
            fit_coords.emplace_back(point.coordinates[0], point.coordinates[1]);
        }
        elementFitPoints[elemId] = fit_coords;

        
        // Only fit if we have enough points
        // Before attempting to fit a curve, the code checks if the total number of points
        // (combined_points, which includes original and neighbor points) is at least the minimum required
        if (combined_points.size() >= MIN_POINTS_FOR_CURVE_FIT) {
            // debug log indicating how many points are used for the fit, including a breakdown of original vs neighbor points 
            std::cout << "Element " << elemId 
                      << " has " << combined_points.size() 
                      << " points for quadratic fitting (" 
                      << original_point_count << " original + " 
                      << points_from_neighbors << " from neighbors)." << std::endl;
            
            // Initialize x_min, x_max, y_min, and y_max to extreme values to later find the bounding box of all points
            double x_min = std::numeric_limits<double>::max();
            double x_max = -std::numeric_limits<double>::max();
            double y_min = std::numeric_limits<double>::max();
            double y_max = -std::numeric_limits<double>::max();

            // Loop over the combined points to find the minimum and maximum x,y values used later for the scaling
            for (const auto& p : combined_points) {
                double x = p.coordinates[0];
                double y = p.coordinates[1];
                x_min = std::min(x_min, x);
                x_max = std::max(x_max, x);
                y_min = std::min(y_min, y);
                y_max = std::max(y_max, y);
            }
            // compute this "initial" (in a sense, that it is based on the unrotated points) range;
            // the initial range is used to determine whether the axes need to be rotated
            double x_range_initial = x_max - x_min;
            double y_range_initial = y_max - y_min;
            // set this tolerance as desired
            double tolerance = 5;
            bool rotate_axes = (y_range_initial / x_range_initial > tolerance);  // Change to one-way check
            if (rotate_axes) {
                std::cout << "    AW 14.4: Rotating axes for element " << elemId << " due to steep gradient." << std::endl;
            }

            // initialize the inverse of the square root needed for the rotation
            const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

            // AW 18.4: Prepare container for storing the (possibly rotated) coordinates
            std::vector<std::pair<double, double>> fit_coords;  // declared before this loop

            // in this loop, the actual min and max x,y values for the scaling are computed, depending on whether rotation was done or not
            for (const auto& p : combined_points) {
                // first, get the coordinates per point
                double x = p.coordinates[0];
                double y = p.coordinates[1];

                if (rotate_axes) {
                    double x_rot = inv_sqrt2 * (x + y);
                    double y_rot = inv_sqrt2 * (y - x);
                    // overwrite the points with the rotated points in case it was rotated
                    x = x_rot;
                    y = y_rot;
                }

                // AW 18.4: store the rotated or unrotated coordinate
                fit_coords.emplace_back(x, y);


                // allows computing the ranges for unrotated and rotated points all in one
                x_min = std::min(x_min, x);
                x_max = std::max(x_max, x);
                y_min = std::min(y_min, y);
                y_max = std::max(y_max, y);
            }

            // AW 18.4: finally, store the coordinates per element
            elementFitPoints[elemId] = fit_coords;

            // Compute the ranges in x and y
            double x_range = x_max - x_min;
            double y_range = y_max - y_min;
            // Clamp small ranges to 1.0 to avoid division by zero or bad scaling in later steps
            if (x_range < 1e-12) x_range = 1.0;
            if (y_range < 1e-12) y_range = 1.0;
            
            
            // scalar sums required to construct the normal equations matrix and RHS for quadratic fitting
            double sum_x = 0.0, sum_y = 0.0;
            double sum_x2 = 0.0, sum_x3 = 0.0, sum_x4 = 0.0;
            double sum_xy = 0.0, sum_x2y = 0.0;
            
            // AW 18.4: Looping over all points from the current element and its neighbors; already rotated or unrotated!
            for (const auto& [x, y] : elementFitPoints[elemId]) {
                // Always compute scaled values after (potential) rotation
                double x_scaled = (x - x_min) / x_range;
                double y_scaled = (y - y_min) / y_range;

                if (!rotate_axes) {
                    double x2 = x_scaled * x_scaled;
                
                    sum_x += x_scaled;
                    sum_y += y_scaled;
                    sum_x2 += x2;
                    sum_x3 += x2 * x_scaled;
                    sum_x4 += x2 * x2;
                    sum_xy += x_scaled * y_scaled;
                    sum_x2y += x2 * y_scaled;
                } else {
                    // Fit x = a*y^2 + b*y + c
                    double y2 = y_scaled * y_scaled;
                
                    sum_x += y_scaled;
                    sum_y += x_scaled;
                    sum_x2 += y2;
                    sum_x3 += y2 * y_scaled;
                    sum_x4 += y2 * y2;
                    sum_xy += y_scaled * x_scaled;
                    sum_x2y += y2 * x_scaled;
                }
            
            }
            
            // Stores the number of total points used in the least-squares fitting (original + neighbors)      
            int n = combined_points.size();

            // Declares the matrix A (3×3) and vector b (3×1) that form the system of normal equations
            Matrix3d A;
            Vector3d b;

            // These are the inner products of the basis functions x2,x,1x2,x,1 with themselves, as derived from the least squares minimization min⁡∥Ac⃗−b∥2min∥Ac−b∥2
            A << sum_x4, sum_x3, sum_x2,
                sum_x3, sum_x2, sum_x,
                sum_x2, sum_x,  n;
            // The right-hand side corresponds to the projections of yy onto the basis functions
            // The right-hand side corresponds to the projections of the dependent variable
            // onto the basis functions. That is:
            // - If unrotated: dependent is y, independent is x
            // - If rotated:   dependent is x, independent is y

            b << sum_x2y, sum_xy, sum_y;

            // Solves the linear system
            // fullPivLu() is a robust LU decomposition with full pivoting from the Eigen library
            Vector3d coeffs = A.fullPivLu().solve(b);

            // declares a variable fit of type QuadraticCoefficients
            QuadraticCoefficients fit;

            double a_s = coeffs[0];
            double b_s = coeffs[1];
            double c_s = coeffs[2];

            // AW 14.4: transform scaled fit back to original coordinate system
            // handles the case when no rotation was needed; So we simply rescale the fitted coefficients back to the original coordinates
            if (!rotate_axes) {
                // AW 14.4: transform the polynomial from scaled coordinates back to original
                fit.a = a_s * y_range / (x_range * x_range);
                fit.b = (b_s * y_range / x_range) - 2.0 * fit.a * x_min;
                fit.c = c_s * y_range + y_min - fit.b * x_min - fit.a * x_min * x_min;

                // Saves the unrotated original/neighbor points to a file for debugging or postprocessing    
                std::ofstream pointOrigFile("element_points_original.txt", std::ios::app);
                if (pointOrigFile.is_open()) {
                    for (const auto& point : combined_points) {
                        double x = point.coordinates[0];
                        double y = point.coordinates[1];
                        pointOrigFile << elemId << "\t" << x << "\t" << y << "\n";
                    }
                    pointOrigFile.close();
                } else {
                    std::cerr << "Error: could not open element_points_original.txt" << std::endl;
                }
            } else {
                // Rescale polynomial from scaled rotated space (x̃ = a_s*ỹ² + b_s*ỹ + c_s)
                // to unscaled rotated space (x = a*y² + b*y + c)
                fit.a = a_s * x_range / (y_range * y_range);
                fit.b = (b_s * x_range / y_range) - 2.0 * fit.a * y_min;
                fit.c = c_s * x_range + x_min - fit.b * y_min - fit.a * y_min * y_min;

                // Saves the unrotated original/neighbor points to a file for debugging or postprocessing    
                std::ofstream pointRotFile("element_points_rotated.txt", std::ios::app);
                if (pointRotFile.is_open()) {
                    for (const auto& [x, y] : elementFitPoints[elemId]) {
                        pointRotFile << elemId << "\t" << x << "\t" << y << "\n";
                    }
                    pointRotFile.close();
                } else {
                    std::cerr << "Error: could not open element_points_rotated.txt" << std::endl;
                }
            }
            
            // AW 15.4: After computing the final fit per element, store rotation info
            if (!rotate_axes) {
                elementWasRotated[elemId] = false;
            } else {
                elementWasRotated[elemId] = true;
            }

            
            // Store results
            elementFits[elemId] = fit;
            elementTotalPoints[elemId] = combined_points.size();
            
            // Calculate error on original points
            double total_error = 0.0;
            
            for (const auto& point : original_points) {
                double x = point.coordinates[0];
                double y = point.coordinates[1];
                if (rotate_axes) {
                    double x_rot = inv_sqrt2 * (x + y);
                    double y_rot = inv_sqrt2 * (y - x);
                    // overwrite the points with the rotated points in case it was rotated
                    x = x_rot;
                    y = y_rot;

                    // Calculate x value from fitted curve
                    double fitted_x = fit.a * y * y + fit.b * y + fit.c;
                    
                    // Error is the vertical distance between point and curve
                    double error = std::abs(x - fitted_x);
                    total_error += error;}
                else {
                    // Calculate x value from fitted curve
                    double fitted_y = fit.a * x * x + fit.b * x + fit.c;
                    
                    // Error is the vertical distance between point and curve
                    double error = std::abs(y - fitted_y);
                    total_error += error;
                }
            }
            
            double avg_error = total_error / (original_points.empty() ? 1.0 : original_points.size());
            

            std::cout << "    Average fit error on original points: " << avg_error << std::endl;

            // AW 14.4 new: Print all (x, y) pairs and their scaled x
            std::cout << "    Fit points (original and scaled x):" << std::endl;
        } else {
            std::cout << "Element " << elemId 
                      << " has only " << combined_points.size() 
                      << " unique points (less than " << MIN_POINTS_FOR_CURVE_FIT 
                      << " required) - cannot perform quadratic curve fitting." << std::endl;
        }
    }
    
    // Write results to file
    std::ofstream outFile(output_file);
    
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << output_file << " for writing." << std::endl;
        return;
    }
    
    // AW 15.4: updated file header
    outFile << "Element_ID\tRotated\tNum_Original_Points\tTotal_Points\t"
        << "a(x^2)\tb(x)\tc\tavg_error\n";
    // AW 15.4: old line Alireza, outcommented
    // outFile << "Element_ID\tNum_Original_Points\tTotal_Points\ta(x²)\tb(x)\tc(const)\tavg_error\n";
    
    for (const auto& [elemId, fit] : elementFits) {
        int numPoints = element_points[elemId].size();
        int totalPoints = elementTotalPoints[elemId];
        
        // Calculate error
        double total_error = 0.0;
        for (const auto& point : element_points[elemId]) {
            double x = point.coordinates[0];
            double y = point.coordinates[1];

            const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

            if (elementWasRotated[elemId]) {
                double x_rot = inv_sqrt2 * (x + y);
                double y_rot = inv_sqrt2 * (y - x);
                // overwrite the points with the rotated points in case it was rotated
                x = x_rot;
                y = y_rot;

                // Calculate x value from fitted curve
                double fitted_x = fit.a * y * y + fit.b * y + fit.c;
                
                // Error is the vertical distance between point and curve
                double error = std::abs(x - fitted_x);
                total_error += error;
            } else {
                // Calculate x value from fitted curve
                double fitted_y = fit.a * x * x + fit.b * x + fit.c;
                              
                // Error is the vertical distance between point and curve
                double error = std::abs(y - fitted_y);
                total_error += error;
            }
        }
        
        double avg_error = total_error / (numPoints > 0 ? numPoints : 1.0);
        // AW 15.4: new input for output file
        bool rotated = elementWasRotated[elemId];

        outFile << elemId << "\t"
        << (rotated ? 1 : 0) << "\t"
        << numPoints << "\t"
        << totalPoints << "\t"
        << fit.a << "\t"
        << fit.b << "\t"
        << fit.c << "\t"
        << avg_error << "\n";
    }
    
    outFile.close();
    
    std::cout << "Saved " << elementFits.size() << " element quadratic curve fits to " << output_file << std::endl;
    std::cout << "Used all available points from 2-hop neighborhoods." << std::endl;
}

void IntersectionPointsUtility::ProcessIntersectionPointsAndFitGeneralConic(const std::string& output_file)
{
    // Get all intersection points
    const auto& points = g_IntersectionPointsContainer;
    
    if (points.empty()) {
        std::cout << "No intersection points available for curve fitting." << std::endl;
        return;
    }
    
    // Configuration parameters
    const int MIN_POINTS_FOR_CURVE_FIT = 5;  // Minimum needed for general conic section
    const int NEIGHBOR_EXPANSION_LEVEL = 3;   // Expand to n-hop neighbors
    
    std::cout << "Starting general conic section fitting with " << points.size() << " intersection points." << std::endl;
    std::cout << "Using all available points from 2-hop neighborhoods." << std::endl;
    
    // Group points by element
    std::map<int, std::vector<IntersectionPointData>> element_points;
    // Create a map of points by their coordinates
    std::map<std::pair<double, double>, std::vector<int>> point_to_elements;
    
    for (const auto& point : points) {
        int elemId = point.elementId;
        
        // Round coordinates to handle floating point precision
        double x = std::round(point.coordinates[0] * 1.0E14) / 1.0E14;
        double y = std::round(point.coordinates[1] * 1.0E14) / 1.0E14;
        std::pair<double, double> coord_key(x, y);
        
        // Add this element to the list for this point
        point_to_elements[coord_key].push_back(elemId);
        
        // Add this point to the element's list
        element_points[elemId].push_back(point);
    }
    
    // Find element neighbors (elements that share intersection points)
    std::map<int, std::set<int>> element_neighbors;
    
    for (const auto& [coord, elements] : point_to_elements) {
        // If this point belongs to multiple elements, they are neighbors
        for (size_t i = 0; i < elements.size(); ++i) {
            for (size_t j = i+1; j < elements.size(); ++j) {
                element_neighbors[elements[i]].insert(elements[j]);
                element_neighbors[elements[j]].insert(elements[i]);
            }
        }
    }
    
    // Expand the neighborhood to n-hop neighbors
    std::cout << "Expanding neighborhood with " << NEIGHBOR_EXPANSION_LEVEL << " hops..." << std::endl;
    std::map<int, std::set<int>> expanded_neighbors = element_neighbors;
    
    for (int hop = 2; hop <= NEIGHBOR_EXPANSION_LEVEL; hop++) {
        std::map<int, std::set<int>> next_level_neighbors = expanded_neighbors;
        
        for (const auto& [elemId, current_neighbors] : expanded_neighbors) {
            for (int neighbor : current_neighbors) {
                // for (int next_hop : expanded_neighbors[neighbor]) {
                for (int next_hop : element_neighbors[neighbor]) {
                    if (next_hop != elemId && !expanded_neighbors[elemId].count(next_hop)) {
                        next_level_neighbors[elemId].insert(next_hop);
                    }
                }
            }
        }
        
        expanded_neighbors = next_level_neighbors;
        std::cout << "Completed " << hop << "-hop neighborhood expansion." << std::endl;
    }
    
    // Structure to hold general conic section coefficients (y² + ax² + bxy + cy + dx + e = 0)
    struct ConicCoefficients {
        double a;  // coefficient of x²
        double b;  // coefficient of xy
        double c;  // coefficient of y
        double d;  // coefficient of x
        double e;  // constant term
    };
    
    // Maps to store results
    std::map<int, ConicCoefficients> elementFits;
    std::map<int, int> elementTotalPoints;
    
    // Process each element
    for (const auto& [elemId, neighbors] : expanded_neighbors) {
        // Get original points for this element
        std::vector<IntersectionPointData> original_points = element_points[elemId];
        int original_point_count = original_points.size();
        
        // Create a pool of neighbor points
        std::vector<IntersectionPointData> neighbor_points;
        for (int neighborId : neighbors) {
            neighbor_points.insert(neighbor_points.end(), 
                                 element_points[neighborId].begin(), 
                                 element_points[neighborId].end());
        }
        
        // Remove duplicates and points shared with original set
        std::map<std::pair<double, double>, IntersectionPointData> unique_neighbor_points;
        for (const auto& point : neighbor_points) {
            double x = std::round(point.coordinates[0] * 1.0E14) / 1.0E14;
            double y = std::round(point.coordinates[1] * 1.0E14) / 1.0E14;
            std::pair<double, double> key(x, y);
            
            // Skip points that are in the original set
            bool is_in_original = false;
            for (const auto& orig_point : original_points) {
                double ox = std::round(orig_point.coordinates[0] * 1.0E14) / 1.0E14;
                double oy = std::round(orig_point.coordinates[1] * 1.0E14) / 1.0E14;
                if (ox == x && oy == y) {
                    is_in_original = true;
                    break;
                }
            }
            
            if (!is_in_original) {
                unique_neighbor_points[key] = point;
            }
        }
        
        // Create a vector of unique neighbor points
        neighbor_points.clear();
        for (const auto& [_, point] : unique_neighbor_points) {
            neighbor_points.push_back(point);
        }
        
        // Build the set of points for curve fitting
        std::vector<IntersectionPointData> combined_points = original_points;
        
        // Add all neighbor points
        combined_points.insert(combined_points.end(), neighbor_points.begin(), neighbor_points.end());
        
        int points_from_neighbors = neighbor_points.size();
        
        // Only fit if we have enough points
        if (combined_points.size() >= MIN_POINTS_FOR_CURVE_FIT) {
            std::cout << "Element " << elemId 
                      << " has " << combined_points.size() 
                      << " points for general conic fitting (" 
                      << original_point_count << " original + " 
                      << points_from_neighbors << " from neighbors)." << std::endl;
            
            // Prepare matrices for least squares fitting of y² + ax² + bxy + cy + dx + e = 0
            // We'll use Matrix class (as in the original function)
            
            // For a general conic, we need to solve for 5 parameters (a, b, c, d, e)
            // We'll rearrange as: y² = -ax² - bxy - cy - dx - e
            Matrix A(5, 5);
            Vector b(5);
            
            // Initialize matrices with zeros
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    A(i, j) = 0.0;
                }
                b[i] = 0.0;
            }
            
            // Fill matrices by summing contributions from each point
            for (const auto& point : combined_points) {
                double x = point.coordinates[0];
                double y = point.coordinates[1];
                
                double x2 = x * x;
                double y2 = y * y;
                double xy = x * y;
                
                // Row 0: x^4 x^3y x^2y x^3 x^2 | x^2y^2
                A(0, 0) += x2 * x2;       // x^4
                A(0, 1) += x2 * x * y;     // x^3y
                A(0, 2) += x2 * y;         // x^2y
                A(0, 3) += x2 * x;         // x^3
                A(0, 4) += x2;             // x^2
                b[0] += x2 * y2;           // x^2y^2
                
                // Row 1: x^3y x^2y^2 xy^2 x^2y xy | xy^3
                A(1, 0) += x2 * x * y;     // x^3y
                A(1, 1) += x2 * y2;        // x^2y^2
                A(1, 2) += x * y2;         // xy^2
                A(1, 3) += x2 * y;         // x^2y
                A(1, 4) += x * y;          // xy
                b[1] += x * y2 * y;        // xy^3
                
                // Row 2: x^2y xy^2 y^2 xy y | y^3
                A(2, 0) += x2 * y;         // x^2y
                A(2, 1) += x * y2;         // xy^2
                A(2, 2) += y2;             // y^2
                A(2, 3) += x * y;          // xy
                A(2, 4) += y;              // y
                b[2] += y2 * y;            // y^3
                
                // Row 3: x^3 x^2y xy x^2 x | xy^2
                A(3, 0) += x2 * x;         // x^3
                A(3, 1) += x2 * y;         // x^2y
                A(3, 2) += x * y;          // xy
                A(3, 3) += x2;             // x^2
                A(3, 4) += x;              // x
                b[3] += x * y2;            // xy^2
                
                // Row 4: x^2 xy y x 1 | y^2
                A(4, 0) += x2;             // x^2
                A(4, 1) += x * y;          // xy
                A(4, 2) += y;              // y
                A(4, 3) += x;              // x
                A(4, 4) += 1.0;            // 1
                b[4] += y2;                // y^2
            }
            
            // Solve the system for the conic coefficients
            // For simplicity, we'll use Cramer's rule as in the original code
            
            double det = determinant5x5(A);
            
            // Create copies of A for solving using Cramer's rule
            Matrix A1 = A, A2 = A, A3 = A, A4 = A, A5 = A;
            
            // Replace columns with b vector
            for (int i = 0; i < 5; i++) {
                A1(i, 0) = b[i];
                A2(i, 1) = b[i];
                A3(i, 2) = b[i];
                A4(i, 3) = b[i];
                A5(i, 4) = b[i];
            }
            
            // Calculate determinants
            double det1 = determinant5x5(A1);
            double det2 = determinant5x5(A2);
            double det3 = determinant5x5(A3);
            double det4 = determinant5x5(A4);
            double det5 = determinant5x5(A5);
            
            // Solve for conic parameters
            ConicCoefficients fit;
            fit.a = -det1 / det;  // coefficient of x²
            fit.b = -det2 / det;  // coefficient of xy
            fit.c = -det3 / det;  // coefficient of y
            fit.d = -det4 / det;  // coefficient of x
            fit.e = -det5 / det;  // constant term
            
            // Store results
            elementFits[elemId] = fit;
            elementTotalPoints[elemId] = combined_points.size();
            
            // Calculate error on original points
            double total_error = 0.0;
            
            for (const auto& point : original_points) {
                double x = point.coordinates[0];
                double y = point.coordinates[1];
                
                // Calculate error as deviation from the conic equation: y² + ax² + bxy + cy + dx + e = 0
                double equation_value = y*y + fit.a*x*x + fit.b*x*y + fit.c*y + fit.d*x + fit.e;
                
                // Error is the absolute value of the equation
                double error = std::abs(equation_value);
                total_error += error;
            }
            
            double avg_error = total_error / (original_points.empty() ? 1.0 : original_points.size());
            
            std::cout << "Element " << elemId 
                      << " fitted with " << combined_points.size() 
                      << " points: y² + " << fit.a << "x² + " 
                      << fit.b << "xy + " << fit.c << "y + " 
                      << fit.d << "x + " << fit.e << " = 0"
                      << std::endl;
            std::cout << "    Average fit error on original points: " << avg_error << std::endl;
        } else {
            std::cout << "Element " << elemId 
                      << " has only " << combined_points.size() 
                      << " unique points (less than " << MIN_POINTS_FOR_CURVE_FIT 
                      << " required) - cannot perform general conic curve fitting." << std::endl;
        }
    }

 

    
    // Write results to file
    std::ofstream outFile(output_file);
    
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << output_file << " for writing." << std::endl;
        return;
    }
    
    outFile << "Element_ID\tNum_Original_Points\tTotal_Points\ta(x²)\tb(xy)\tc(y)\td(x)\te(const)\tavg_error\n";

    
    for (const auto& [elemId, fit] : elementFits) {
        int numPoints = element_points[elemId].size();
        int totalPoints = elementTotalPoints[elemId];
        
        // Calculate error
        double total_error = 0.0;
        for (const auto& point : element_points[elemId]) {
            double x = point.coordinates[0];
            double y = point.coordinates[1];
            
            // Calculate error as deviation from the conic equation
            double equation_value = y*y + fit.a*x*x + fit.b*x*y + fit.c*y + fit.d*x + fit.e;
            
            // Error is the absolute value of the equation
            double error = std::abs(equation_value);
            total_error += error;
        }
        
        double avg_error = total_error / (numPoints > 0 ? numPoints : 1.0);
        
        outFile << elemId << "\t" 
                << numPoints << "\t"
                << totalPoints << "\t"
                << fit.a << "\t" 
                << fit.b << "\t" 
                << fit.c << "\t"
                << fit.d << "\t" 
                << fit.e << "\t"
                << avg_error << "\n";
    }
    
    outFile.close();
    
    std::cout << "Saved " << elementFits.size() << " element general conic curve fits to " << output_file << std::endl;
    std::cout << "Used all available points from 2-hop neighborhoods." << std::endl;
}

// Helper function to calculate determinant of a 5x5 matrix
double IntersectionPointsUtility::determinant5x5(const Matrix& A) {
    // For a general 5x5 determinant, we'll expand along the first row
    // This is not the most efficient method but it's straightforward to implement
    
    double det = 0.0;
    
    for (int j = 0; j < 5; j++) {
        // Create a 4x4 submatrix by excluding row 0 and column j
        Matrix subMatrix(4, 4);
        
        // Fill the submatrix
        for (int r = 0; r < 4; r++) {
            int row = r + 1;  // Skip row 0
            int subCol = 0;
            
            for (int c = 0; c < 5; c++) {
                if (c != j) {
                    subMatrix(r, subCol) = A(row, c);
                    subCol++;
                }
            }
        }
        
        // Calculate sign: (-1)^(i+j)
        double sign = ((j % 2) == 0) ? 1.0 : -1.0;
        
        // Calculate determinant recursively
        det += sign * A(0, j) * determinant4x4(subMatrix);
    }
    
    return det;
}

// Helper function to calculate determinant of a 4x4 matrix
double IntersectionPointsUtility::determinant4x4(const Matrix& A) {
    // Calculate determinant using cofactor expansion
    
    double det = 0.0;
    
    for (int j = 0; j < 4; j++) {
        // Create a 3x3 submatrix by excluding row 0 and column j
        Matrix subMatrix(3, 3);
        
        // Fill the submatrix
        for (int r = 0; r < 3; r++) {
            int row = r + 1;  // Skip row 0
            int subCol = 0;
            
            for (int c = 0; c < 4; c++) {
                if (c != j) {
                    subMatrix(r, subCol) = A(row, c);
                    subCol++;
                }
            }
        }
        
        // Calculate sign: (-1)^(i+j)
        double sign = ((j % 2) == 0) ? 1.0 : -1.0;
        
        // Calculate determinant recursively
        det += sign * A(0, j) * determinant3x3(subMatrix);
    }
    
    return det;
}

// Helper function to calculate determinant of a 3x3 matrix
double IntersectionPointsUtility::determinant3x3(const Matrix& A) {
    return A(0, 0) * (A(1, 1) * A(2, 2) - A(2, 1) * A(1, 2)) -
           A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) +
           A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
}
/////////////////////////////////////////////////////////
}
}
