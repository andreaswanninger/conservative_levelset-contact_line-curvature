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
// AW 14.5: added to be in accordance with latest impl by Alireza
#include <iomanip>  // For std::setprecision
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
    // AW 14.5: these lines added
    std::vector<InterfaceAverageData> InterfaceAveragesUtility::mInterfaceAverageContainer;
    std::vector<IntersectionDataWithNormal> g_IntersectionDataWithNormalContainer;

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
    // AW 14.5: this line changed
    const double sign_threshold = 1e-15;
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
               // std::cout << "Processed interface points for element " << pElement->Id() << std::endl;
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
            // AW 24.4: print statement removed
            // std::cout << "Found " << interface_points.size() << " interface points for element " << elementId << std::endl;
            
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
            // AW 20.5: comment this output
            // std::cout << "Added " << point_count << " real intersection points from element " << elementId << std::endl;
        } else {
            std::cout << "No interface points found for element " << elementId << std::endl;
            
               }
        } catch (std::exception& e) {
        std::cerr << "Error extracting interface points: " << e.what() << std::endl;
        
        
        }
}

///////////////////////////////////////////////
// AW 14.5: added this part for normal averaging
/**
 * @brief Calculate and store intersection lengths for 2D elements
 * @param rModelPart The model part containing elements
 * @return Number of elements with valid intersection lengths
 */
int CalculateAndStoreElementIntersectionLengths(ModelPart& rModelPart)
{
    // Count of elements with valid intersection lengths
    int count = 0;
    
    // Clear any previous intersection points
    KratosDropletDynamics::IntersectionPointsUtility::ClearIntersectionPoints();
    
    // Collect intersection points for all elements
    for (auto it = rModelPart.ElementsBegin(); it != rModelPart.ElementsEnd(); ++it) {
        Element::Pointer pElement = *(it.base());
        KratosDropletDynamics::IntersectionPointsUtility::CollectElementIntersectionPoints(pElement);
    }
    
    // Get all intersection points
    const auto& points = KratosDropletDynamics::IntersectionPointsUtility::GetIntersectionPoints();
    
    // Group points by element ID
    std::map<int, std::vector<array_1d<double, 3>>> elementPoints;
    for (const auto& point : points) {
        elementPoints[point.elementId].push_back(point.coordinates);
    }
    
    // Initialize INTERSECTION_LENGTH_2D to zero for all elements
    for (auto it = rModelPart.ElementsBegin(); it != rModelPart.ElementsEnd(); ++it) {
        it->SetValue(INTERSECTION_LENGTH_2D, 0.0);
    }
    
    // Calculate length for each element and store it
    for (const auto& [elementId, elementPointList] : elementPoints) {
        if (elementPointList.size() == 2) { // For 2D elements
            // Calculate distance between the two points
            double dx = elementPointList[0][0] - elementPointList[1][0];
            double dy = elementPointList[0][1] - elementPointList[1][1];
            double dz = elementPointList[0][2] - elementPointList[1][2];
            
            double length = std::sqrt(dx*dx + dy*dy + dz*dz);
            
            // Find the element and store the value
            auto elem_iterator = rModelPart.Elements().find(elementId);
            if (elem_iterator != rModelPart.Elements().end()) {
                elem_iterator->SetValue(INTERSECTION_LENGTH_2D, length);
                count++;
            }
        }
    }
    
    return count;
}

/**
 * @brief Get intersection length from an element
 * @param rElement Element to get the intersection length from
 * @return Intersection length value
 */
double GetElementIntersectionLength(const Element& rElement)
{
    return rElement.GetValue(INTERSECTION_LENGTH_2D);
}

void InterfaceAveragesUtility::ClearInterfaceAverages()
{
    mInterfaceAverageContainer.clear();
}

const std::vector<InterfaceAverageData>& InterfaceAveragesUtility::GetInterfaceAverages()
{
    return mInterfaceAverageContainer;
}

void InterfaceAveragesUtility::CollectElementInterfaceAverages(Element::Pointer pElement)
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
    
    // Only proceed with calculations if the element is actually split
    if (!is_split) {
        return; // Skip this element as it's not split by the interface
    }
    
    // Structure nodes info (if needed)
    Vector structure_node_id = ZeroVector(p_geom->size());
    
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
        // Prepare variables to store interface data
        Matrix interface_shape_function_neg;
        ModifiedShapeFunctions::ShapeFunctionsGradientsType interface_shape_derivatives_neg;
        Vector interface_weights_neg;
        std::vector<array_1d<double,3>> interface_normals_neg;
        
        try {
            // Calculate interface shape functions, weights, and normals
            p_modified_sh_func->ComputeInterfaceNegativeSideShapeFunctionsAndGradientsValues(
                interface_shape_function_neg,
                interface_shape_derivatives_neg,
                interface_weights_neg,
                GeometryData::IntegrationMethod::GI_GAUSS_2);
                
            p_modified_sh_func->ComputeNegativeSideInterfaceAreaNormals(
                interface_normals_neg,
                GeometryData::IntegrationMethod::GI_GAUSS_2);
        }
        catch (std::exception& e) {
            std::cerr << "Error in element " << pElement->Id() << " computing interface: " << e.what() << std::endl;
            return;
        }
            
        // Normalize interface normals
        for (unsigned int gp = 0; gp < interface_normals_neg.size(); ++gp) {
            const double normal_norm = norm_2(interface_normals_neg[gp]);
            if (normal_norm > 1e-10) {
                interface_normals_neg[gp] /= normal_norm;
            }
        }
        
        // Only proceed if we have interface points
        if (interface_weights_neg.size() > 0) {
            // Create and compute the average interface data
            InterfaceAverageData avg_data;
            avg_data.elementId = pElement->Id();
            avg_data.numberOfPoints = interface_weights_neg.size();
            avg_data.interfaceArea = 0.0;
            
            // Initialize average coordinates and normal to zero
            avg_data.averageCoordinates = ZeroVector(3);
            avg_data.averageNormal = ZeroVector(3);
            
            // Calculate the global coordinates of each interface Gauss point
            std::vector<array_1d<double,3>> interface_global_coords;
            interface_global_coords.resize(interface_weights_neg.size());
            
            // Initialize to zero
            for (unsigned int gp = 0; gp < interface_weights_neg.size(); ++gp) {
                interface_global_coords[gp] = ZeroVector(3);
            }
            
            // Calculate global coordinates using shape functions
            for (unsigned int gp = 0; gp < interface_weights_neg.size(); ++gp) {
                for (unsigned int i_node = 0; i_node < p_geom->size(); ++i_node) {
                    const array_1d<double, 3>& r_node_coords = (*p_geom)[i_node].Coordinates();
                    for (unsigned int d = 0; d < 3; ++d) {
                        interface_global_coords[gp][d] += interface_shape_function_neg(gp, i_node) * r_node_coords[d];
                    }
                }
            }
            
            // Compute total interface area
            for (unsigned int gp = 0; gp < interface_weights_neg.size(); ++gp) {
                avg_data.interfaceArea += interface_weights_neg[gp];
            }
            
            // Compute weighted average coordinates and normal in a simpler way
            for (unsigned int gp = 0; gp < interface_weights_neg.size(); ++gp) {
                avg_data.averageCoordinates += interface_weights_neg[gp] * interface_global_coords[gp];
                avg_data.averageNormal += interface_weights_neg[gp] * interface_normals_neg[gp];
            }
            
            // Finalize the averages by dividing by the total interface area
            if (avg_data.interfaceArea > 1e-10) {
                avg_data.averageCoordinates /= avg_data.interfaceArea;
                
                // Normalize the average normal
                const double avg_normal_norm = norm_2(avg_data.averageNormal);
                if (avg_normal_norm > 1e-10) {
                    avg_data.averageNormal /= avg_normal_norm;
                }
            }
            
            // Add the data to the container
            mInterfaceAverageContainer.push_back(avg_data);
            
            // AW 20.5: comment this debug output
            //std::cout << "Processed interface averages for element " << pElement->Id() 
            //          << " with " << interface_weights_neg.size() << " interface points" << std::endl;
        }
    }
}

void InterfaceAveragesUtility::ComputeModelPartInterfaceAverages(const ModelPart& rModelPart)
{
    // Clear previous data
    ClearInterfaceAverages();
    
    // Process each element in the model part
    for (ModelPart::ElementsContainerType::const_iterator it = rModelPart.ElementsBegin(); 
         it != rModelPart.ElementsEnd(); ++it) {
        Element::Pointer p_element = *(it.base());
        CollectElementInterfaceAverages(p_element);
    }
    
    std::cout << "Computed interface averages for " << mInterfaceAverageContainer.size() 
              << " elements in model part " << rModelPart.Name() << std::endl;
}


/**
 * @brief Find an element in the interface averages container by ID
 * @param rInterfaceAverages Container of interface average data
 * @param ElementId ID of the element to find
 * @return Pointer to the interface average data, or nullptr if not found
 */
const InterfaceAverageData* FindElementInInterfaceAverages(
    const std::vector<InterfaceAverageData>& rInterfaceAverages,
    int ElementId)
{
    for (const auto& avg : rInterfaceAverages) {
        if (avg.elementId == ElementId) {
            return &avg;
        }
    }
    return nullptr;
}

/**
 * @brief Get neighbors of an element through node connectivity
 * @param rModelPart The model part containing elements
 * @param ElementId ID of the element to get neighbors for
 * @return Vector of neighbor element IDs
 */
// std::vector<int> GetElementNeighbors(const ModelPart& rModelPart, int ElementId)
// {
//     std::vector<int> neighbors;
    
//     // Try to find the element
//     auto elem_it = rModelPart.Elements().find(ElementId);
//     if (elem_it == rModelPart.Elements().end()) {
//         return neighbors;
//     }
    
//     Element& r_element = *elem_it;
    
//     // Get element neighbors through nodes
//     for (auto& node : r_element.GetGeometry()) {
//         GlobalPointersVector<Element>& r_neighbor_elements = node.GetValue(NEIGHBOUR_ELEMENTS);
//         for (auto& neighbor_elem : r_neighbor_elements) {
//             if (neighbor_elem.Id() != ElementId && 
//                 std::find(neighbors.begin(), neighbors.end(), neighbor_elem.Id()) == neighbors.end()) {
//                 neighbors.push_back(neighbor_elem.Id());
//             }
//         }
//     }
    
//     return neighbors;
// }

/**
 * @brief Get neighbors of an element through elemental connectivity
 * @param rModelPart The model part containing elements
 * @param ElementId ID of the element to get neighbors for
 * @return Vector of neighbor element IDs
 */
std::vector<int> GetElementNeighbors(const ModelPart& rModelPart, int ElementId)
{
    std::vector<int> neighbors;
    
    // Find the element
    auto elem_it = rModelPart.Elements().find(ElementId);
    if (elem_it == rModelPart.Elements().end()) {
        return neighbors;
    }
    
    Element& r_element = *elem_it;
    
    // Check if the element has NEIGHBOUR_ELEMENTS variable
    if (r_element.Has(NEIGHBOUR_ELEMENTS)) {
        // Get direct element neighbors
        GlobalPointersVector<Element>& r_neighbor_elements = r_element.GetValue(NEIGHBOUR_ELEMENTS);
        
        // Loop through neighbors with null check
        for (unsigned int i = 0; i < r_neighbor_elements.size(); i++) {
            // Check for null pointers
            if (r_neighbor_elements(i).get() != nullptr) {
                neighbors.push_back(r_neighbor_elements(i)->Id());
            }
        }
    }
    
    return neighbors;
}

/**
 * @brief Get three points (target element and two neighbors) for normal fitting
 * @param rModelPart The model part containing elements
 * @param rInterfaceAverages Container of interface average data
 * @param ElementId ID of the target element
 * @param rPoints Output vector to hold the three points (target + 2 neighbors)
 * @return True if three points were found, false otherwise
 */
bool GetThreePointsForFitting(
    const ModelPart& rModelPart,
    const std::vector<InterfaceAverageData>& rInterfaceAverages,
    int ElementId,
    std::vector<const InterfaceAverageData*>& rPoints)
{
    // Clear the vector
    rPoints.clear();
    
    // Find the target element data
    const InterfaceAverageData* target_data = FindElementInInterfaceAverages(rInterfaceAverages, ElementId);
    if (!target_data) {
        std::cout << "Element " << ElementId << " not found in interface averages" << std::endl;
        return false;
    }
    
    // Add the target element to the points
    rPoints.push_back(target_data);
    
    // Get immediate neighbors with interface data
    std::vector<int> neighbors = GetElementNeighbors(rModelPart, ElementId);
    std::vector<const InterfaceAverageData*> interface_neighbors;
    
    for (int neighbor_id : neighbors) {
        const InterfaceAverageData* neighbor_data = FindElementInInterfaceAverages(rInterfaceAverages, neighbor_id);
        if (neighbor_data) {
            interface_neighbors.push_back(neighbor_data);
            if (interface_neighbors.size() >= 2) {
                // We found two neighbors with interface data, add them to points
                rPoints.push_back(interface_neighbors[0]);
                rPoints.push_back(interface_neighbors[1]);
                return true;
            }
        }
    }
    
    // If we don't have enough immediate neighbors, try neighbors of neighbors
    if (interface_neighbors.size() < 2) {
        std::set<int> processed_neighbors;
        for (int id : neighbors) {
            processed_neighbors.insert(id);
        }
        processed_neighbors.insert(ElementId);
        
        for (int neighbor_id : neighbors) {
            std::vector<int> second_level_neighbors = GetElementNeighbors(rModelPart, neighbor_id);
            
            for (int nn_id : second_level_neighbors) {
                // Skip already processed elements
                if (processed_neighbors.find(nn_id) != processed_neighbors.end()) {
                    continue;
                }
                
                const InterfaceAverageData* nn_data = FindElementInInterfaceAverages(rInterfaceAverages, nn_id);
                if (nn_data) {
                    interface_neighbors.push_back(nn_data);
                    processed_neighbors.insert(nn_id);
                    
                    if (interface_neighbors.size() >= 2) {
                        // We found two neighbors with interface data (including second level), add them to points
                        rPoints.push_back(interface_neighbors[0]);
                        rPoints.push_back(interface_neighbors[1]);
                        return true;
                    }
                }
            }
        }
    }
    
    // If we still don't have enough points, add what we have
   if (interface_neighbors.size() == 1) {
        rPoints.push_back(interface_neighbors[0]);
        std::cout << "Warning: Only two points available for element " << ElementId << std::endl;
        return false;
    }
    
    std::cout << "Warning: Only one point available for element " << ElementId << std::endl;
    return false;
}

/**
 * @brief Fit a normal vector using exactly three points with focus on stability
 * @param rModelPart The model part containing elements
 * @param rInterfaceAverages Container of interface average data
 * @param ElementId ID of the target element
 * @return The fitted normal vector, or the original normal if fitting fails
 */
array_1d<double, 3> FitLinearNormal(
    const ModelPart& rModelPart,
    const std::vector<InterfaceAverageData>& rInterfaceAverages,
    int ElementId,
    double& a0, double& a1, double& a2,   // Coefficients for nx = a0 + a1*x + a2*y
    double& b0, double& b1, double& b2)   // Coefficients for ny = b0 + b1*x + b2*y
{
    // Find target element data
    const InterfaceAverageData* target_data = nullptr;
    for (const auto& avg : rInterfaceAverages) {
        if (avg.elementId == ElementId) {
            target_data = &avg;
            break;
        }
    }
    
    // If target element not found, return a default normal
    if (!target_data) {
        std::cout << "Error: Element " << ElementId << " not found in interface averages" << std::endl;
        array_1d<double, 3> default_normal = ZeroVector(3);
        default_normal[0] = 1.0;  // Default to x-direction
        
        // Set coefficients to zero
        a0 = 1.0; a1 = 0.0; a2 = 0.0;
        b0 = 0.0; b1 = 0.0; b2 = 0.0;
        
        return default_normal;
    }
    
    // Store the target element coordinates and normal
    const double target_x = target_data->averageCoordinates[0];
    const double target_y = target_data->averageCoordinates[1];
    const double target_nx = target_data->averageNormal[0];
    const double target_ny = target_data->averageNormal[1];
    const double target_nz = target_data->averageNormal[2];
    
    // Get neighbors without using the GetThreePointsForFitting function
    std::vector<const InterfaceAverageData*> neighbors;
    
    // Get direct neighbors from ModelPart
    std::vector<int> neighbor_ids = GetElementNeighbors(rModelPart, ElementId);
    
    // Find matching neighbors in interface averages
    for (int id : neighbor_ids) {
        for (const auto& avg : rInterfaceAverages) {
            if (avg.elementId == id) {
                neighbors.push_back(&avg);
                break;
            }
        }
        
        // Break if we found enough neighbors
        if (neighbors.size() >= 2) {
            break;
        }
    }
    
    // If we don't have enough neighbors, search for neighbors of neighbors
    if (neighbors.size() < 2) {
        for (int id : neighbor_ids) {
            std::vector<int> second_level = GetElementNeighbors(rModelPart, id);
            
            for (int id2 : second_level) {
                // Skip if it's the target or already processed
                if (id2 == ElementId || std::find(neighbor_ids.begin(), neighbor_ids.end(), id2) != neighbor_ids.end()) {
                    continue;
                }
                
                for (const auto& avg : rInterfaceAverages) {
                    if (avg.elementId == id2) {
                        neighbors.push_back(&avg);
                        break;
                    }
                }
                
                // Break if we found enough neighbors
                if (neighbors.size() >= 2) {
                    break;
                }
            }
            
            if (neighbors.size() >= 2) {
                break;
            }
        }
    }
    
    // If we still don't have enough neighbors, just return the original normal
    if (neighbors.size() < 2) {
        std::cout << "Warning: Not enough neighbors for element " << ElementId << ". Using original normal." << std::endl;
        
        array_1d<double, 3> original_normal = ZeroVector(3);
        original_normal[0] = target_nx;
        original_normal[1] = target_ny;
        original_normal[2] = target_nz;
        
        // Set coefficients (constants only, no derivatives)
        a0 = target_nx; a1 = 0.0; a2 = 0.0;
        b0 = target_ny; b1 = 0.0; b2 = 0.0;
        
        return original_normal;
    }
    
    // We now have the target and two neighbors
    const double x1 = target_x;
    const double y1 = target_y;
    const double nx1 = target_nx;
    const double ny1 = target_ny;
    
    const double x2 = neighbors[0]->averageCoordinates[0];
    const double y2 = neighbors[0]->averageCoordinates[1];
    const double nx2 = neighbors[0]->averageNormal[0];
    const double ny2 = neighbors[0]->averageNormal[1];
    
    const double x3 = neighbors[1]->averageCoordinates[0];
    const double y3 = neighbors[1]->averageCoordinates[1];
    const double nx3 = neighbors[1]->averageNormal[0];
    const double ny3 = neighbors[1]->averageNormal[1];
    
    // Calculate coefficients directly using determinants
    // This avoids potential issues with Gaussian elimination
    
    // Compute the determinant of the coefficient matrix
    double det = (x2*y3 - x3*y2) - x1*(y3 - y2) + y1*(x3 - x2);
    
    if (std::abs(det) < 1.0e-10) {
        std::cout << "Warning: Singular matrix for element " << ElementId << ". Using original normal." << std::endl;
        
        array_1d<double, 3> original_normal = ZeroVector(3);
        original_normal[0] = target_nx;
        original_normal[1] = target_ny;
        original_normal[2] = target_nz;
        
        // Set coefficients (constants only, no derivatives)
        a0 = target_nx; a1 = 0.0; a2 = 0.0;
        b0 = target_ny; b1 = 0.0; b2 = 0.0;
        
        return original_normal;
    }
    
    // Compute coefficients for nx
    a0 = ((nx1*(x2*y3 - x3*y2)) + (nx2*(x3*y1 - x1*y3)) + (nx3*(x1*y2 - x2*y1))) / det;
    a1 = ((nx1*(y2 - y3)) + (nx2*(y3 - y1)) + (nx3*(y1 - y2))) / det;
    a2 = ((nx1*(x3 - x2)) + (nx2*(x1 - x3)) + (nx3*(x2 - x1))) / det;
    
    // Compute coefficients for ny
    b0 = ((ny1*(x2*y3 - x3*y2)) + (ny2*(x3*y1 - x1*y3)) + (ny3*(x1*y2 - x2*y1))) / det;
    b1 = ((ny1*(y2 - y3)) + (ny2*(y3 - y1)) + (ny3*(y1 - y2))) / det;
    b2 = ((ny1*(x3 - x2)) + (ny2*(x1 - x3)) + (ny3*(x2 - x1))) / det;
    
    // Evaluate the fitted normal at the target point (which should match the original normal)
    double fitted_nx = a0 + a1*x1 + a2*y1;
    double fitted_ny = b0 + b1*x1 + b2*y1;
    
    // Create the fitted normal vector
    array_1d<double, 3> fitted_normal = ZeroVector(3);
    fitted_normal[0] = fitted_nx;
    fitted_normal[1] = fitted_ny;
    fitted_normal[2] = target_nz; // Preserve the original z-component
    
    // Normalize the vector
    double norm = std::sqrt(fitted_normal[0]*fitted_normal[0] + 
                           fitted_normal[1]*fitted_normal[1] + 
                           fitted_normal[2]*fitted_normal[2]);
    
    if (norm > 1.0e-10) {
        fitted_normal[0] /= norm;
        fitted_normal[1] /= norm;
        fitted_normal[2] /= norm;
    }
    
    // Calculate curvature
    double curvature = a1 + b2;
    
    // Output information
    std::cout << "Normal fitting for element " << ElementId << ":" << std::endl;
    std::cout << "  nx = " << a0 << " + " << a1 << "*x + " << a2 << "*y" << std::endl;
    std::cout << "  ny = " << b0 << " + " << b1 << "*x + " << b2 << "*y" << std::endl;
    std::cout << "  nz = " << target_nz << " (preserved from original)" << std::endl;
    std::cout << "  Curvature (a1 + b2): " << curvature << std::endl;
    
    // Verify fit at target element
    double nx_error = std::abs(fitted_nx - target_nx);
    double ny_error = std::abs(fitted_ny - target_ny);
    
    if (nx_error > 1.0e-10 || ny_error > 1.0e-10) {
        std::cout << "Warning: Target element " << ElementId << " normal not exactly fitted:" << std::endl;
        std::cout << "  Original: [" << target_nx << ", " << target_ny << "]" << std::endl;
        std::cout << "  Fitted:   [" << fitted_nx << ", " << fitted_ny << "]" << std::endl;
        std::cout << "  Error:    [" << nx_error << ", " << ny_error << "]" << std::endl;
    }
    
    return fitted_normal;
}

/**
 * @brief Save fitted normals to a file with robust error handling
 * @param rModelPart The model part containing elements
 * @param rInterfaceAverages Container of interface average data
 * @param Filename Filename to save the results
 */
void SaveFittedNormalsToFile(
    const ModelPart& rModelPart,
    const std::vector<InterfaceAverageData>& rInterfaceAverages,
    const std::string& Filename)
{
    try {
        // Open output file
        std::ofstream outFile(Filename);
        
        if (!outFile.is_open()) {
            std::cerr << "Error: Could not open file " << Filename << " for writing." << std::endl;
            return;
        }
        
        // Write header
        outFile << std::fixed << std::setprecision(15);
        outFile << "Element_ID\tX\tY\tZ\tOrig_NX\tOrig_NY\tOrig_NZ\tFit_NX\tFit_NY\tFit_NZ\tDiff\t"
                << "a0\ta1\ta2\tb0\tb1\tb2\tCurvature" << std::endl;
        
        // Process each element one by one
        int successful_elements = 0;
        
        for (const auto& avg : rInterfaceAverages) {
            int element_id = avg.elementId;
            double a0 = 0.0, a1 = 0.0, a2 = 0.0;
            double b0 = 0.0, b1 = 0.0, b2 = 0.0;
            array_1d<double, 3> fitted_normal;
            double curvature = 0.0;
            double diff_magnitude = 0.0;
            
            try {
                // Calculate a linearly fitted normal
                fitted_normal = FitLinearNormal(rModelPart, rInterfaceAverages, element_id, 
                                             a0, a1, a2, b0, b1, b2);
                
                // Calculate difference between fitted and original normal
                array_1d<double, 3> normal_diff = fitted_normal - avg.averageNormal;
                diff_magnitude = std::sqrt(normal_diff[0]*normal_diff[0] + 
                                        normal_diff[1]*normal_diff[1] + 
                                        normal_diff[2]*normal_diff[2]);
                
                // Calculate curvature
                curvature = a1 + b2;
                successful_elements++;
            }
            catch (const std::exception& e) {
                std::cerr << "Error processing element " << element_id << ": " << e.what() << std::endl;
                
                // Use original normal as fallback
                fitted_normal = avg.averageNormal;
                a0 = avg.averageNormal[0];
                b0 = avg.averageNormal[1];
                curvature = 0.0;
                diff_magnitude = 0.0;
            }
            
            // Write to file
            outFile << element_id << "\t"
                    << avg.averageCoordinates[0] << "\t"
                    << avg.averageCoordinates[1] << "\t"
                    << avg.averageCoordinates[2] << "\t"
                    << avg.averageNormal[0] << "\t"
                    << avg.averageNormal[1] << "\t"
                    << avg.averageNormal[2] << "\t"
                    << fitted_normal[0] << "\t"
                    << fitted_normal[1] << "\t"
                    << fitted_normal[2] << "\t"
                    << diff_magnitude << "\t"
                    << a0 << "\t" << a1 << "\t" << a2 << "\t"
                    << b0 << "\t" << b1 << "\t" << b2 << "\t"
                    << curvature << std::endl;
        }
        
        outFile.close();
        std::cout << "Saved " << rInterfaceAverages.size() << " fitted normals with curvature to " << Filename 
                  << " (" << successful_elements << " successfully fitted)" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error in SaveFittedNormalsToFile: " << e.what() << std::endl;
    }
}

/**
 * @brief Set the ELEMENT_CUT_NORMAL variable for all elements that are cut by the interface
 * @param rModelPart The model part containing elements
 * @return Number of elements where normal was set
 */
int SetElementCutNormals(ModelPart& rModelPart)
{
    int count = 0;
    
    // First compute interface averages if not already done
    if (InterfaceAveragesUtility::GetInterfaceAverages().empty()) {
        InterfaceAveragesUtility::ComputeModelPartInterfaceAverages(rModelPart);
    }
    
    // Get interface averages
    const auto& interface_averages = InterfaceAveragesUtility::GetInterfaceAverages();
    
    // Set normal for each element in the interface averages
    for (const auto& avg : interface_averages) {
        auto elem_it = rModelPart.Elements().find(avg.elementId);
        if (elem_it != rModelPart.Elements().end()) {
            Element& r_element = *elem_it;
            
            // Set the normal vector as element variable
            r_element.SetValue(ELEMENT_CUT_NORMAL, avg.averageNormal);
            count++;
        }
    }
    
    return count;
}

/**
 * @brief Get X component of element cut normal
 * @param rElement Element to get the normal from
 * @return X component of the normal
 */
double GetElementCutNormalX(const Element& rElement)
{
    return rElement.GetValue(ELEMENT_CUT_NORMAL)[0];
}

/**
 * @brief Get Y component of element cut normal
 * @param rElement Element to get the normal from
 * @return Y component of the normal
 */
double GetElementCutNormalY(const Element& rElement)
{
    return rElement.GetValue(ELEMENT_CUT_NORMAL)[1];
}

/**
 * @brief Get Z component of element cut normal
 * @param rElement Element to get the normal from
 * @return Z component of the normal
 */
double GetElementCutNormalZ(const Element& rElement)
{
    return rElement.GetValue(ELEMENT_CUT_NORMAL)[2];
}

/**
 * @brief Clear the intersection data with normal container
 */
void ClearIntersectionDataWithNormal()
{
    g_IntersectionDataWithNormalContainer.clear();
}

/**
 * @brief Get the container of intersection data with normals
 * @return Reference to the container
 */
const std::vector<IntersectionDataWithNormal>& GetIntersectionDataWithNormal()
{
    return g_IntersectionDataWithNormalContainer;
}

/**
 * @brief Populate the intersection data with normal container
 * @param rModelPart The model part containing elements
 * @return Number of elements processed
 */
int CollectIntersectionDataWithNormal(ModelPart& rModelPart)
{
    // Clear previous data
    ClearIntersectionDataWithNormal();
    
    // Make sure we have length data
    int num_lengths = CalculateAndStoreElementIntersectionLengths(rModelPart);
    
    // Make sure we have interface averages
    if (InterfaceAveragesUtility::GetInterfaceAverages().empty()) {
        InterfaceAveragesUtility::ComputeModelPartInterfaceAverages(rModelPart);
    }
    
    // Get interface averages
    const auto& interface_averages = InterfaceAveragesUtility::GetInterfaceAverages();
    
    // Create a map for easy lookup of interface data by element ID
    std::map<int, const InterfaceAverageData*> avg_map;
    for (const auto& avg : interface_averages) {
        avg_map[avg.elementId] = &avg;
    }
    
    // Process each element with intersection length data
    int count = 0;
    for (ModelPart::ElementsContainerType::iterator it = rModelPart.ElementsBegin(); 
         it != rModelPart.ElementsEnd(); ++it) {
        Element& r_element = *it;
        double length = r_element.GetValue(INTERSECTION_LENGTH_2D);
        
        // Only include elements with non-zero intersection length
        if (length > 0.0) {
            int element_id = r_element.Id();
            
            // Create the data object
            IntersectionDataWithNormal data;
            data.elementId = element_id;
            data.intersectionLength = length;
            
            // Try to find normal data
            auto avg_it = avg_map.find(element_id);
            if (avg_it != avg_map.end()) {
                const InterfaceAverageData* avg = avg_it->second;
                
                // Copy normal and coordinates
                data.normal = avg->averageNormal;
                data.coordinates = avg->averageCoordinates;
            }
            
            // Add to the container
            g_IntersectionDataWithNormalContainer.push_back(data);
            count++;
        }
    }
    
    std::cout << "Collected " << count << " elements with intersection data and normals" << std::endl;
    return count;
}

/**
 * @brief Save the intersection data with normals to file
 * @param filename Filename to save the results
 */
void SaveIntersectionDataWithNormalToFile(const std::string& filename)
{
    std::ofstream outFile(filename);
    
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }
    
    outFile << std::fixed << std::setprecision(15);
    outFile << "Element_ID\tIntersection_Length\tNormal_X\tNormal_Y\tNormal_Z\tCoord_X\tCoord_Y\tCoord_Z" << std::endl;
    
    for (const auto& data : g_IntersectionDataWithNormalContainer) {
        outFile << data.elementId << "\t"
               << data.intersectionLength << "\t"
               << data.normal[0] << "\t"
               << data.normal[1] << "\t" 
               << data.normal[2] << "\t"
               << data.coordinates[0] << "\t"
               << data.coordinates[1] << "\t"
               << data.coordinates[2] << std::endl;
    }
    
    outFile.close();
    std::cout << "Successfully wrote " << g_IntersectionDataWithNormalContainer.size() 
              << " elements with intersection data and normals to " << filename << std::endl;
}
//////////////////////////////////////////////////////////////////////
/**
 * @brief Compute an averaged normal for an element considering neighboring cut elements
 * @param rModelPart The model part containing elements
 * @param ElementId ID of the target element
 * @param NeighborLevels Number of neighbor levels to consider (0 = no neighbors, 1 = direct neighbors only)
 * @return Weighted average normal vector
 */
array_1d<double, 3> ComputeAveragedElementNormal(
    const ModelPart& rModelPart,
    int ElementId,
    int NeighborLevels)
{
    // Initialize output normal
    array_1d<double, 3> averaged_normal = ZeroVector(3);
    double total_weight = 0.0;
    
    // Get the element
    auto elem_it = rModelPart.Elements().find(ElementId);
    if (elem_it == rModelPart.Elements().end()) {
        return averaged_normal; // Return zero vector if element not found
    }
    
    Element& r_element = *elem_it;
    
    // Add the current element to the calculation
    double elem_length = r_element.GetValue(INTERSECTION_LENGTH_2D);
    array_1d<double, 3> elem_normal = r_element.GetValue(ELEMENT_CUT_NORMAL);
    
    // Skip if element is not cut
    if (elem_length <= 0.0) {
        return elem_normal; // Return element's normal directly if not cut
    }
    
    // Special case: if neighbor hops is 0, return only the element's own normal
    if (NeighborLevels == 0) {
        return elem_normal;
    }
    
    // Add the element's contribution
    averaged_normal += elem_length * elem_normal;
    total_weight += elem_length;
    
    // Find and collect all neighbors up to specified levels
    std::set<int> processed_elements;
    std::set<int> current_level_ids;
    std::set<int> next_level_ids;
    
    // Start with current element
    processed_elements.insert(ElementId);
    current_level_ids.insert(ElementId);
    
    // Traverse neighbor levels
    for (int level = 0; level < NeighborLevels; level++) {
        next_level_ids.clear();

        ///////////////////
        // Calculate weight scaling factor for this level (2/(1+e^(level))
        double level_weight = 2/(1+exp(level+1));  // 0.537883, 0.238406, etc.
        ///////////////////
        // Process each element in current level
        for (int current_id : current_level_ids) {
            // Find direct neighbors
            std::vector<int> neighbors = GetElementNeighbors(rModelPart, current_id);
            //KRATOS_INFO("element Neighbors") <<"level= "<< level+1 <<", "<<"level_weight= "<< level_weight << ", "<< ElementId << ": " <<neighbors << std::endl;
            
            for (int neighbor_id : neighbors) {
                // Skip if already processed
                if (processed_elements.find(neighbor_id) != processed_elements.end()) {
                    continue;
                }
                
                // Get neighbor element
                auto neighbor_it = rModelPart.Elements().find(neighbor_id);
                if (neighbor_it == rModelPart.Elements().end()) {
                    continue;
                }
                
                Element& r_neighbor = *neighbor_it;
                
                // Get neighbor intersection length and normal
                double neighbor_length = r_neighbor.GetValue(INTERSECTION_LENGTH_2D);
                
                // Only include if it's a cut element
                if (neighbor_length > 0.0) {
                    array_1d<double, 3> neighbor_normal = r_neighbor.GetValue(ELEMENT_CUT_NORMAL);
                    
                    // Make sure the normal orientation is consistent
                    // (dot product > 0 means normals point in roughly same direction)
                    if (inner_prod(elem_normal, neighbor_normal) < 0) {
                        neighbor_normal = -neighbor_normal; // Flip the normal if pointing in opposite direction
                    }
                    
                    // // Add contribution
                    // averaged_normal += neighbor_length * neighbor_normal;
                    // total_weight += neighbor_length;
                    //////////////////////////////
                    // Add contribution with level-based weight scaling
                    averaged_normal += level_weight * neighbor_length * neighbor_normal;
                    total_weight += level_weight * neighbor_length;
                    /////////////////////////////
                }
                
                // Add to next level
                next_level_ids.insert(neighbor_id);
                processed_elements.insert(neighbor_id);
            }
        }
        
        // Update for next level
        current_level_ids = next_level_ids;
    }
    
    // Normalize the result if we have non-zero weight
    if (total_weight > 1e-12) {
        averaged_normal /= total_weight;
        
        // Normalize the vector
        double norm = norm_2(averaged_normal);
        if (norm > 1e-12) {
            averaged_normal /= norm;
        }
    } else {
        // If no weights, return original normal
        averaged_normal = elem_normal;
    }
    
    return averaged_normal;
}

/**
 * @brief Compute and store averaged normals for all cut elements in the model part
 * @param rModelPart The model part containing elements
 * @param NeighborLevels Number of neighbor levels to consider
 * @param VariableName The variable name to store the averaged normal (default: ELEMENT_CUT_NORMAL_AVERAGED)
 * @return Number of elements processed
 */
int ComputeAndStoreAveragedNormals(
    ModelPart& rModelPart,
    int NeighborLevels,
    const std::string& VariableName)
{
    ClearAveragedNormals(rModelPart, VariableName);
    
    // Make sure we have cut normals set
    if (InterfaceAveragesUtility::GetInterfaceAverages().empty()) {
        InterfaceAveragesUtility::ComputeModelPartInterfaceAverages(rModelPart);
    }
    
    int num_elements = SetElementCutNormals(rModelPart);
    std::cout << "Set cut normals for " << num_elements << " elements" << std::endl;
    
    // Count of processed elements
    int count = 0;
    
    // Process each element
    for (auto it = rModelPart.ElementsBegin(); it != rModelPart.ElementsEnd(); ++it) {
        Element& r_element = *it;
        double length = r_element.GetValue(INTERSECTION_LENGTH_2D);
        
        // Only process cut elements
        if (length > 0.0) {
            // Compute averaged normal
            array_1d<double, 3> averaged_normal = ComputeAveragedElementNormal(
                rModelPart, r_element.Id(), NeighborLevels);
            
            // Store it
            r_element.SetValue(ELEMENT_CUT_NORMAL_AVERAGED, averaged_normal);
            count++;
        }
    }
    
    std::cout << "Computed averaged normals for " << count << " elements using " 
              << NeighborLevels << " neighbor levels" << std::endl;
    
    return count;
}
/**
 * @brief Clear averaged normal values from all elements
 * @param rModelPart The model part containing elements
 * @param VariableName The variable name storing the averaged normal
 */
void ClearAveragedNormals(
    ModelPart& rModelPart,
    const std::string& VariableName)
{
    // Check if the variable exists
    if (!KratosComponents<Variable<array_1d<double, 3>>>::Has(VariableName)) {
        std::cout << "Warning: Variable " << VariableName << " not found. Nothing to clear." << std::endl;
        return;
    }
    
    // Zero vector to reset values
    array_1d<double, 3> zero_vector = ZeroVector(3);
    
    // Clear all element values
    for (auto it = rModelPart.ElementsBegin(); it != rModelPart.ElementsEnd(); ++it) {
        it->SetValue(ELEMENT_CUT_NORMAL_AVERAGED, zero_vector);
    }
    
    std::cout << "Cleared " << VariableName << " values for all elements" << std::endl;
}

/**
 * @brief Save the averaged normals to file
 * @param rModelPart The model part containing elements
 * @param Filename Filename to save the results
 * @param VariableName Name of the variable storing the averaged normals
 */
void SaveAveragedNormalsToFile(
    const ModelPart& rModelPart,
    const std::string& Filename,
    const std::string& VariableName)
{
    // Open output file
    std::ofstream outFile(Filename);
    
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << Filename << " for writing." << std::endl;
        return;
    }
    
    // Write header
    outFile << std::fixed << std::setprecision(15);
    outFile << "Element_ID\tNormal_X\tNormal_Y\tNormal_Z\tLength" << std::endl;
    
    // Count of elements with valid averaged normals
    int count = 0;
    
    // Process each element
    for (auto it = rModelPart.ElementsBegin(); it != rModelPart.ElementsEnd(); ++it) {
        Element& r_element = *it;
        
        // Check if element has an intersection length
        double intersection_length = r_element.GetValue(INTERSECTION_LENGTH_2D);
        
        // Skip elements that are not cut by the interface
        if (intersection_length <= 0.0) {
            continue;
        }
        
        // Get the averaged normal vector - directly use ELEMENT_CUT_NORMAL_AVERAGED
        array_1d<double, 3> averaged_normal = r_element.GetValue(ELEMENT_CUT_NORMAL_AVERAGED);
        
        // Skip elements with zero normal (not calculated or invalid)
        if (norm_2(averaged_normal) < 1e-10) {
            continue;
        }
        
        // Write to file
        outFile << std::fixed << std::setprecision(15); 
        outFile << r_element.Id() << "\t"
                << averaged_normal[0] << "\t"
                << averaged_normal[1] << "\t"
                << averaged_normal[2] << "\t"
                << intersection_length << std::endl;
        
        count++;
    }
    
    outFile.close();
    std::cout << "Successfully wrote " << count << " elements with averaged normals to " << Filename << std::endl;
}



// AW 14.5: end of added part
//////////////////////////////////////////////

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
    
    // AW 24.4: print statement removed
    // std::cout << "Starting circle fitting with " << points.size() << " intersection points." << std::endl;
    // std::cout << "Using all available points from 2-hop neighborhoods." << std::endl;
    
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
            // AW 24.4: remove debug message
            /* std::cout << "Element " << elemId 
                      << " has " << combined_points.size() 
                      << " points for circle fitting (" 
                      << original_point_count << " original + " 
                      << points_from_neighbors << " from neighbors)." << std::endl; */
            
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
            
            // AW 24.4: print statement removed
           /*  std::cout << "Element " << elemId 
                      << " fitted with " << combined_points.size() 
                      << " points: (x-" << fit.a 
                      << ")² + (y-" << fit.b << ")² = " << fit.c 
                      << " (radius = " << radius << ")" 
                      << std::endl;
            std::cout << "    Average fit error on original points: " << avg_error << std::endl; */
        } else {
            // AW 24.4: print statement removed
            /*             std::cout << "Element " << elemId 
                      << " has only " << combined_points.size() 
                      << " unique points (less than " << MIN_POINTS_FOR_CIRCLE_FIT 
                      << " required) - cannot perform circle fitting." << std::endl; */
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
    // AW 24.4: removed debug message
    // std::cout << "Starting quadratic curve fitting with " << points.size() << " intersection points." << std::endl;
    // std::cout << "Using all available points from 2-hop neighborhoods." << std::endl;
    
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
            // AW 24.4: print statement removed
            /* std::cout << "Element " << elemId 
                      << " has " << combined_points.size() 
                      << " points for quadratic fitting (" 
                      << original_point_count << " original + " 
                      << points_from_neighbors << " from neighbors)." << std::endl; */
            
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
            double tolerance = 5.0;
            // AW 21.4: changed to two-way tolerance
            bool rotate_axes = (y_range_initial / x_range_initial > tolerance) || 
                            (x_range_initial / y_range_initial > tolerance);

            if (rotate_axes) {
                // AW 24.4: print statement removed
                /* std::cout << "    AW 14.4: Rotating axes for element " << elemId 
                        << " due to steep gradient (y_range = " << y_range_initial 
                        << ", x_range = " << x_range_initial << ")." << std::endl; */
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
            
            // AW 24.4: print statement removed
            // std::cout << "    Average fit error on original points: " << avg_error << std::endl;

            // AW 14.4 new: Print all (x, y) pairs and their scaled x
            // AW 24.4: print statement removed
            // std::cout << "    Fit points (original and scaled x):" << std::endl;
        } else {
            // AW 24.4: print statement removed
            /* std::cout << "Element " << elemId 
                      << " has only " << combined_points.size() 
                      << " unique points (less than " << MIN_POINTS_FOR_CURVE_FIT 
                      << " required) - cannot perform quadratic curve fitting." << std::endl; */
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
