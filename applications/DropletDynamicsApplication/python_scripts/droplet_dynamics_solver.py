# Importing the Kratos Library
import KratosMultiphysics
from KratosMultiphysics import auxiliary_solver_utilities
from KratosMultiphysics.python_solver import PythonSolver
import KratosMultiphysics.python_linear_solver_factory as linear_solver_factory
# Import applications
import KratosMultiphysics.FluidDynamicsApplication as KratosCFD
import KratosMultiphysics.DropletDynamicsApplication as KratosDroplet
#import KratosMultiphysics.ConvectionDiffusionApplication as KratosConv
# Import base class file
#from KratosMultiphysics.ConvectionDiffusionApplication.convection_diffusion_solver import ConvectionDiffusionSolver
#from KratosMultiphysics.FluidDynamicsApplication.fluid_solver import FluidSolver
#from KratosMultiphysics.FluidDynamicsApplication.navier_stokes_two_fluids_solver import NavierStokesTwoFluidsSolver

################# AW-C 2.7: all of these libraries are newly added #################
# AW 8.5: Necessary libraries originating from the merge with conservative level set branch
import math
import sys
import time
import importlib
import numpy as np
import json
# AW: additional libraries conditionally used
# OS-level operations 
import os
# Python debugger 
import pdb
# AW 10.4: necessary for csv file writing
import csv
# AW 26.4: necessary libraries for Nurbs fitting
import pandas as pd
import matplotlib.pyplot as plt                        # plotting
from geomdl import fitting, NURBS                      # NURBS fitting + curve class
from geomdl.visualization import VisMPL as vis
from collections import defaultdict, deque
# AW 5.5: libraries needed for closest point minimization
from scipy.optimize import minimize_scalar
from matplotlib import cm
############### AW-C 2.7: end of newly added libraries #########################

from pathlib import Path

def CreateSolver(model, custom_settings):
    return DropletDynamicsSolver(model, custom_settings)

class DropletDynamicsSolver(PythonSolver):  # Before, it was derived from NavierStokesTwoFluidsSolver

    @classmethod
    def GetDefaultParameters(cls):
        ##settings string in json format
        # AW 21.4: quasistatic contact angle settings added
        # AW 5.5: smoothing coefficient increased to 500
        # AW 19.5: fitting settings added
        # AW 28.5: added ref point of initial center of droplet on solid surface to fitting settings (for correctly computing normal orientation)
        # AW 2.6: added settings for curvature smoothing + normal penalty, x_threshold for mixed wettability, normal penalty
        # AW 3.6: added parallel redistancing settings 
        # AW 25.6: added normal penalty settings
        default_settings = KratosMultiphysics.Parameters("""
        {
            "solver_type": "two_fluids",
            "model_part_name": "",
            "domain_size": -1,
            "model_import_settings": {
                "input_type": "mdpa",
                "input_filename": "unknown_name",
                "reorder": false
            },
            "material_import_settings": {
                "materials_filename": ""
            },
            "maximum_iterations": 7,
            "echo_level": 0,
            "time_order": 2,
            "time_scheme": "bdf2",
            "compute_reactions": false,
            "analysis_type": "non_linear",
            "reform_dofs_at_each_step": false,
            "consider_periodic_conditions": false,
            "relative_velocity_tolerance": 1e-3,
            "absolute_velocity_tolerance": 1e-5,
            "relative_pressure_tolerance": 1e-3,
            "absolute_pressure_tolerance": 1e-5,
            "linear_solver_settings"       : {
                "solver_type"         : "amgcl"
            },
            "volume_model_part_name" : "volume_model_part",
            "skin_parts": [""],
            "assign_neighbour_elements_to_conditions": true,
            "no_skin_parts":[""],
            "time_stepping"                : {
                "automatic_time_step" : true,
                "CFL_number"          : 1,
                "minimum_delta_time"  : 1e-2,
                "maximum_delta_time"  : 1.0,
                "time_step"           : 0.0
            },
            "periodic": "periodic",
            "move_mesh_flag": false,
            "acceleration_limitation": true,
            "formulation": {
                "dynamic_tau": 1.0
            },
            "levelset_convection_settings": {
                "max_CFL" : 1.0,
                "max_substeps" : 0,
                "eulerian_error_compensation" : false,
                "element_type" : "levelset_convection_supg",
                "element_settings" : {
                    "dynamic_tau" : 0.0,
                    "cross_wind_stabilization_factor" : 0.7
                }
            },
            "contact_angle_settings": {
                "theta_advancing" : 130,
                "theta_receding" : 130
            },                                               
            "distance_reinitialization": "none",
            "parallel_redistance_max_layers" : 25,
            "max_distance" : 1.0,
            "calculate_exact_distances_to_plane" : true,
            "preserve_interface" : false,    
            "distance_smoothing": false,
            "distance_smoothing_coefficient": 250.0,
            "distance_modification_settings": {
                "model_part_name": "",
                "distance_threshold": 1e-7,
                "continuous_distance": true,
                "check_at_each_time_step": true,
                "avoid_almost_empty_elements": false,
                "deactivate_full_negative_elements": false
            },
            "QuasiStatic_ContactAngle_Settings": {
                "QuasiStatic_ContactAngle" : true,                                        
                "Theta_equilibrium_hydrophilic" : 50,
                "Theta_equilibrium_hydrophobic" : 130,                                     
                "Penalty_coefficient" : 100,
                "X_threshold" : 0.005,
                "smooth_scaling" : false,
                "smooth_scaling_lower_threshold" : 1,
                "smooth_scaling_upper_threshold" : 3
            },
            "convection_diffusion_settings": {
                "Perform_conservative_law": false,
                "echo_level": 0,
                "parallel_type": "",
                "max_substeps": 100,
                "time_step": 0.001,
                "power_value_in_epsilon_Calculation": 0.9,
                "denominator_value_in_epsilon_Calculation": 2.0,
                "convergenge_tolerence": 0.0001                            
            },
            "fitting_settings": {
            "fitting_type": "nurbs",             
            "use_partial_fitting": false,          
            "fitting_element_ids": [49, 50, 51, 149, 151, 152, 249, 349], 
            "normal_evaluation_mode": 1,
            "reference_point_x": 0.015,
            "reference_point_y": 0                                                        
            },
            "curvature_normal_smoothing_settings": {
            "do_smoothing": false,             
            "method": "savgol",          
            "window_size": 3, 
            "polynomial_order": 2                                                     
            },
            "normal_penalty_settings": {
            "do_normal_penalty": false,
            "is_horizontal_wall": false,
            "x_left_wall" : 0,
            "x_right_wall": 1,
            "y_bottom_wall": 0,
            "tolerance_normal_penalty": 1e-8,
            "fix_static_equilibrium": false
            }                                                                                                                                              
        }""")

        default_settings.AddMissingParameters(super(DropletDynamicsSolver, cls).GetDefaultParameters())
        return default_settings


    def __init__(self, model, custom_settings):
        """Initializing the solver."""
        # TODO: DO SOMETHING IN HERE TO REMOVE THE "time_order" FROM THE DEFAULT SETTINGS BUT KEEPING THE BACKWARDS COMPATIBILITY

        # AW 8.5 Merge
         #Defining a pseudo time variable for extracting the convection diffusion results
        self.pseudo_time = 0

        if custom_settings.Has("levelset_convection_settings"):
            if custom_settings["levelset_convection_settings"].Has("levelset_splitting"):
                custom_settings["levelset_convection_settings"].RemoveValue("levelset_splitting")
                KratosMultiphysics.Logger.PrintWarning("NavierStokesTwoFluidsSolver", "\'levelset_splitting\' has been temporarily deactivated. Using the standard levelset convection with no splitting.")

        #TODO: Remove this after the retrocompatibility period
        if custom_settings.Has("bfecc_convection"):
            KratosMultiphysics.Logger.PrintWarning("NavierStokesTwoFluidsSolver", "the semi-Lagrangian \'bfecc_convection\' is no longer supported. Using the standard Eulerian levelset convection.")
            custom_settings.RemoveValue("bfecc_convection")
            if custom_settings.Has("bfecc_number_substeps"):
                custom_settings.RemoveValue("bfecc_number_substeps")

        # FluidSolver.__init__(self,model,custom_settings)
        super(DropletDynamicsSolver,self).__init__(model, custom_settings)

        # Either retrieve the model part from the model or create a new one
        model_part_name = self.settings["model_part_name"].GetString()

        if model_part_name == "":
            raise Exception('Please provide the model part name as the "model_part_name" (string) parameter!')

        if self.model.HasModelPart(model_part_name):
            self.main_model_part = self.model.GetModelPart(model_part_name)
        else:
            self.main_model_part = self.model.CreateModelPart(model_part_name)

        domain_size = self.settings["domain_size"].GetInt()
        if domain_size == -1:
            raise Exception('Please provide the domain size as the "domain_size" (int) parameter!')

        self.main_model_part.ProcessInfo.SetValue(KratosMultiphysics.DOMAIN_SIZE, domain_size)

        self.element_name = "DropletDynamics"
        self.condition_name = "TwoFluidNavierStokesWallCondition"
        self.element_integrates_in_time = True
        self.element_has_nodal_properties = True

        self.min_buffer_size = 3

        # Set the levelset characteristic variables and add them to the convection settings
        # These are required to be set as some of the auxiliary processes admit user-defined variables
        self._levelset_variable = KratosMultiphysics.DISTANCE
        self._levelset_gradient_variable = KratosMultiphysics.DISTANCE_GRADIENT
        self._levelset_convection_variable = KratosMultiphysics.VELOCITY
        self.settings["levelset_convection_settings"].AddEmptyValue("levelset_variable_name").SetString("DISTANCE")
        self.settings["levelset_convection_settings"].AddEmptyValue("levelset_gradient_variable_name").SetString("DISTANCE_GRADIENT")
        self.settings["levelset_convection_settings"].AddEmptyValue("levelset_convection_variable_name").SetString("VELOCITY")

        dynamic_tau = self.settings["formulation"]["dynamic_tau"].GetDouble()
        self.main_model_part.ProcessInfo.SetValue(KratosMultiphysics.DYNAMIC_TAU, dynamic_tau)

        surface_tension = True
        #if (self.settings["formulation"].Has("surface_tension")):
        #    surface_tension = self.settings["formulation"]["surface_tension"].GetBool()
        self.main_model_part.ProcessInfo.SetValue(KratosCFD.SURFACE_TENSION, surface_tension)

        self.momentum_correction = True
        #if self.settings["formulation"].Has("momentum_correction"):
        #    self.momentum_correction = self.settings["formulation"]["momentum_correction"].GetBool()
        self.main_model_part.ProcessInfo.SetValue(KratosCFD.MOMENTUM_CORRECTION, self.momentum_correction)

        self._reinitialization_type = self.settings["distance_reinitialization"].GetString()

        self._distance_smoothing = self.settings["distance_smoothing"].GetBool()
        smoothing_coefficient = self.settings["distance_smoothing_coefficient"].GetDouble()
        self.main_model_part.ProcessInfo.SetValue(KratosCFD.SMOOTHING_COEFFICIENT, smoothing_coefficient)

        self._apply_acceleration_limitation = self.settings["acceleration_limitation"].GetBool()

        ## Set the distance reading filename
        # TODO: remove the manual "distance_file_name" set as soon as the problem type one has been tested.
        #if (self.settings["distance_reading_settings"]["import_mode"].GetString() == "from_GiD_file"):
        #    self.settings["distance_reading_settings"]["distance_file_name"].SetString(self.settings["model_import_settings"]["input_filename"].GetString()+".post.res")

        KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, "Construction of NavierStokesTwoFluidsSolver finished.")

        # AW 10.4: create csv files
        # Define output CSV paths (could also be moved to settings if desired)
        self._curvature_csv_unfitted = "unfitted_curvature.csv"
        self._curvature_csv_fitted = "fitted_curvature.csv"
        self._normals_csv_unfitted = "unfitted_normals.csv"
        self._normals_csv_fitted = "fitted_normals.csv"

        # Create headers (overwrite on init)
        with open(self._curvature_csv_unfitted, "w") as f:
            f.write("Time,Element_ID,GaussPoint,UnfittedCurvature\n")

        # AW 2.7: header adapted 
        with open(self._curvature_csv_fitted, "w") as f:
            f.write("Time,Element_ID,GaussPoint,FittedCurvature\n")

        with open(self._normals_csv_unfitted, "w") as f:
            f.write("Time,Element_ID,GaussPoint,Nx,Ny,Nz\n")

        # AW 2.7: header adapted
        with open(self._normals_csv_fitted, "w") as f:
            f.write("Time,Element_ID,GaussPoint,Nx,Ny,Nz\n")

        # AW 21.4: Added user-defined quasistationary contact line settings
        qscl_settings = self.settings["QuasiStatic_ContactAngle_Settings"]
        QuasiStatic_ContactAngle = qscl_settings["QuasiStatic_ContactAngle"].GetBool()
        Theta_equilibrium_hydrophilic = qscl_settings["Theta_equilibrium_hydrophilic"].GetDouble()
        Theta_equilibrium_hydrophobic = qscl_settings["Theta_equilibrium_hydrophobic"].GetDouble()
        Penalty_coefficient = qscl_settings["Penalty_coefficient"].GetDouble()
        # AW 2.6: added user-defined setting for x-threshold in mixed wettability
        X_threshold = qscl_settings["X_threshold"].GetDouble()
        self.main_model_part.ProcessInfo.SetValue(KratosDroplet.quasi_static_contact_angle, QuasiStatic_ContactAngle)
        self.main_model_part.ProcessInfo.SetValue(KratosDroplet.theta_equilibrium_hydrophilic, Theta_equilibrium_hydrophilic)
        self.main_model_part.ProcessInfo.SetValue(KratosDroplet.theta_equilibrium_hydrophobic, Theta_equilibrium_hydrophobic)
        self.main_model_part.ProcessInfo.SetValue(KratosDroplet.penalty_coefficient, Penalty_coefficient)
        # AW 2.6: added user-defined setting for x-threshold in mixed wettability
        self.main_model_part.ProcessInfo.SetValue(KratosDroplet.X_threshold, X_threshold)
        # AW 25.6: corresponding settings for h_coeff smoothing
        smooth_scaling = qscl_settings["smooth_scaling"].GetBool()
        self.main_model_part.ProcessInfo.SetValue(KratosDroplet.smooth_scaling, smooth_scaling)
        smooth_scaling_lower_threshold = qscl_settings["smooth_scaling_lower_threshold"].GetDouble()
        self.main_model_part.ProcessInfo.SetValue(KratosDroplet.smooth_scaling_lower_threshold, smooth_scaling_lower_threshold)
        smooth_scaling_upper_threshold = qscl_settings["smooth_scaling_upper_threshold"].GetDouble()
        self.main_model_part.ProcessInfo.SetValue(KratosDroplet.smooth_scaling_upper_threshold, smooth_scaling_upper_threshold)


        # AW 19.5: Added user-defined fitting settings
        fitting_settings = self.settings["fitting_settings"]
        fitting_type = fitting_settings["fitting_type"].GetString()
        self.main_model_part.ProcessInfo.SetValue(KratosDroplet.FittingType, fitting_type)
        use_partial_fitting = fitting_settings["use_partial_fitting"].GetBool()
        self.main_model_part.ProcessInfo.SetValue(KratosDroplet.UsePartialFitting, use_partial_fitting)
        fitting_element_ids = fitting_settings["fitting_element_ids"].GetVector()
        self.main_model_part.ProcessInfo.SetValue(KratosDroplet.FittingElementIds, fitting_element_ids)
        normal_evaluation_mode = fitting_settings["normal_evaluation_mode"].GetInt()
        self.main_model_part.ProcessInfo.SetValue(KratosDroplet.NormalEvaluationMode, normal_evaluation_mode)
        # AW 28.5: Added reference point for normal orientation
        self.reference_point_x = fitting_settings["reference_point_x"].GetDouble()
        self.reference_point_y = fitting_settings["reference_point_y"].GetDouble()

        # AW 2.6: Added user-defined settings
        curvature_normal_smoothing_settings = self.settings["curvature_normal_smoothing_settings"]
        self.do_curvature_normal_smoothing = curvature_normal_smoothing_settings["do_smoothing"].GetBool()
        self.curvature_normal_smoothing_method = curvature_normal_smoothing_settings["method"].GetString()
        self.curvature_normal_smoothing_window_size = curvature_normal_smoothing_settings["window_size"].GetInt()
        self.curvature_normal_smoothing_poly_order = curvature_normal_smoothing_settings["polynomial_order"].GetInt()

        # AW 2.6: added normal penalty setting
        normal_penalty_settings = self.settings["normal_penalty_settings"]
        self.do_normal_penalty = normal_penalty_settings["do_normal_penalty"].GetBool()
        # AW 25.6: added the rest of the settings
        self.is_horizontal_wall = normal_penalty_settings["is_horizontal_wall"].GetBool()
        self.x_left_wall = normal_penalty_settings["x_left_wall"].GetDouble()
        self.x_right_wall = normal_penalty_settings["x_right_wall"].GetDouble()
        self.y_bottom_wall = normal_penalty_settings["y_bottom_wall"].GetDouble()
        self.tolerance_normal_penalty = normal_penalty_settings["tolerance_normal_penalty"].GetDouble()
        self.fix_static_equilibrium = normal_penalty_settings["fix_static_equilibrium"].GetBool()


    def AddDofs(self):
        dofs_and_reactions_to_add = []
        dofs_and_reactions_to_add.append(["VELOCITY_X", "REACTION_X"])
        dofs_and_reactions_to_add.append(["VELOCITY_Y", "REACTION_Y"])
        dofs_and_reactions_to_add.append(["VELOCITY_Z", "REACTION_Z"])
        dofs_and_reactions_to_add.append(["PRESSURE", "REACTION_WATER_PRESSURE"])
        KratosMultiphysics.VariableUtils.AddDofsList(dofs_and_reactions_to_add, self.main_model_part)

        KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, "Fluid solver DOFs added correctly.")


    def GetDofsList(self):
        """This function creates and returns a list with the DOFs defined in the conditions and elements specifications
        Note that this requires the main_model_part to be already set, that is to say to have already performed the element substitution (see PrepareModelPart).
        """
        return KratosMultiphysics.SpecificationsUtilities.GetDofsListFromSpecifications(self.main_model_part)


    def ImportModelPart(self):
        # we can use the default implementation in the base class
        self._ImportModelPart(self.main_model_part,self.settings["model_import_settings"])


    def AddVariables(self):
        self.main_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.DENSITY)
        self.main_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.DYNAMIC_VISCOSITY)
        self.main_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.PRESSURE)
        self.main_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.VELOCITY)
        self.main_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.ACCELERATION)
        self.main_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.MESH_VELOCITY)
        self.main_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.IS_STRUCTURE)
        self.main_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.BODY_FORCE)
        self.main_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.NODAL_H)
        # AW 8.5 Merge
        self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.NODAL_H_MAX)
        self.main_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.NODAL_AREA)
        self.main_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.REACTION)
        self.main_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.REACTION_WATER_PRESSURE)
        self.main_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.NORMAL)
        self.main_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.EXTERNAL_PRESSURE)
        self.main_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.FLAG_VARIABLE)
        self.main_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.DISTANCE)              # Distance function nodal values
        self.main_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.DISTANCE_GRADIENT)     # Distance gradient nodal values
        self.main_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.AUX_DISTANCE)
        self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.DISTANCE_AUX)                   # Auxiliary distance function nodal values
        self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.DISTANCE_AUX2)                  # Auxiliary distance function nodal values       
        self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.DISTANCE_GRADIENT_AUX)          # Auxiliary Distance gradient nodal values
        self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.CONVECTIVE_VELOCITY)            # Store conctive velocity for level-set process
        self.main_model_part.AddNodalSolutionStepVariable(KratosCFD.CURVATURE)                      # Store curvature as a nodal variable
        self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.AREA_VARIABLE_AUX)              # Auxiliary area_variable for parallel distance calculator
        self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.NORMAL_VECTOR)                  # Auxiliary normal vector at interface
        self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.TANGENT_VECTOR)                 # Auxiliary tangent vector at contact line
        self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.CONTACT_VECTOR)                 # Auxiliary contact vector
        self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.CONTACT_ANGLE)                  # Contact angle (may not be needed at nodes)
        self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.CONTACT_VECTOR_MICRO)           # Auxiliary contact vector at micro-scale
        self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.CONTACT_ANGLE_MICRO)            # Contact angle (micro-scale)
        self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.CONTACT_VELOCITY)               # Contact line tangential velocity (normal to the contact-line)
        #self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.VELOCITY_STAR)                  # Last known velocity
        #self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.PRESSURE_STAR)                  # Last known pressure
        # self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.PRESSURE_GRADIENT_AUX)          # Pressure gradient on positive and negative sides
        #self.main_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.NORMAL_VELOCITY)
        # AW 28.4: register fitted curvature and normals 
        self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.FITTED_CURVATURE)
        self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.FITTED_NORMAL)
        # AW 14.5: added new variables for normal averaging
        self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.VELOCITY_X_GRADIENT)
        self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.VELOCITY_Y_GRADIENT)
        self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.VELOCITY_Z_GRADIENT)      
        self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.DISTANCE_GRADIENT_DIVERGENCE)           
        self.main_model_part.AddNodalSolutionStepVariable(KratosDroplet.DISTANCE_GRADIENT_SIZE_GRADIENT)
        # AW 14.5: end of new variables

    def PrepareModelPart(self):
        # Restarting the simulation is OFF (needs a careful implementation)
        # Set fluid properties from materials json file
        materials_imported = self._SetPhysicalProperties()
        if not materials_imported:
            KratosMultiphysics.Logger.PrintWarning(self.__class__.__name__, "Material properties have not been imported. Check \'material_import_settings\' in your ProjectParameters.json.")
        # Replace default elements and conditions
        self._ReplaceElementsAndConditions()
        # Set and fill buffer
        self._SetAndFillBuffer()

        # Executes the check and prepare model process. Always executed as it also assigns neighbors which are not saved in a restart
        self._ExecuteCheckAndPrepare()

        KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, "Model reading finished.")


    def ExportModelPart(self):
        # Writing the model part
        name_out_file = self.settings["model_import_settings"]["input_filename"].GetString()+".out"
        KratosMultiphysics.ModelPartIO(name_out_file, KratosMultiphysics.IO.WRITE).WriteModelPart(self.main_model_part)
        KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, "Model export finished.")

   
    def GetMinimumBufferSize(self):
        return self.min_buffer_size

    
    # Temporary name: "fluid_computational_model_part"
    def GetComputingModelPart(self):
        if not self.main_model_part.HasSubModelPart("fluid_computational_model_part"):
            raise Exception("The ComputingModelPart was not created yet!")
        return self.main_model_part.GetSubModelPart("fluid_computational_model_part")


    def Initialize(self):
        computing_model_part = self.GetComputingModelPart()
        # Calculate boundary normals
        KratosMultiphysics.NormalCalculationUtils().CalculateOnSimplex(
            computing_model_part,
            computing_model_part.ProcessInfo[KratosMultiphysics.DOMAIN_SIZE])
        
        # AW 8.5 Merge
         # Initializing level-set function
        if ( self.settings["convection_diffusion_settings"]["Perform_conservative_law"].GetBool() == True ):
            self._Initialize_Conservative_LevelSet_Function()

        # Finding nodal and elemental neighbors
        data_communicator = computing_model_part.GetCommunicator().GetDataCommunicator()
        neighbour_search = KratosMultiphysics.FindGlobalNodalNeighboursProcess(
            data_communicator,
            computing_model_part)
        neighbour_search.Execute()

        elemental_neighbour_search = KratosMultiphysics.GenericFindElementalNeighboursProcess(
            computing_model_part)
        elemental_neighbour_search.Execute()

        # Set and initialize the solution strategy
        solution_strategy = self._GetSolutionStrategy()
        solution_strategy.SetEchoLevel(self.settings["echo_level"].GetInt())
        solution_strategy.Initialize()

        # Set nodal properties after setting distance(level-set).
        self._SetNodalProperties()

        # Initialize the distance correction process
        self._GetDistanceModificationProcess().ExecuteInitialize()
        self._GetDistanceModificationProcess().ExecuteInitializeSolutionStep()

        #Here the initial water volume of the system is calculated without considering inlet and outlet flow rate
        self.initial_system_volume=KratosCFD.FluidAuxiliaryUtilities.CalculateFluidNegativeVolume(self.GetComputingModelPart())

        # Instantiate the level set convection process
        # Note that is is required to do this in here in order to validate the defaults and set the corresponding distance gradient flag
        # Note that the nodal gradient of the distance is required either for the eulerian BFECC limiter or by the algebraic element antidiffusivity
        self._GetLevelSetConvectionProcess()

        # Just to be sure that no conflict would occur if the element is derived from the Navier Stokes Two Fluids.
        self.mass_source = False

        KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, "Solver initialization finished.")

        # The external interfacial force (per unit area) should be set in case of the presence of electromagnetic forces, etc.
        KratosMultiphysics.VariableUtils().SetNonHistoricalVariableToZero(KratosDroplet.EXT_INT_FORCE, self.main_model_part.Elements)


    def AdvanceInTime(self, current_time):
        dt = self.settings["time_stepping"]["time_step"].GetDouble() # The automatic dt calculation is deactivated.
        new_time = current_time + dt

        self.main_model_part.CloneTimeStep(new_time)
        self.main_model_part.ProcessInfo[KratosMultiphysics.STEP] += 1

        return new_time

    
    def InitializeSolutionStep(self):

        # AW 11.6-C: Initializes or resets the value of the non-historical variable DISTANCE_CORRECTION to 0.0 for all nodes
        # Momentum correction is on by default!
        KratosMultiphysics.VariableUtils().SetNonHistoricalVariable(KratosCFD.DISTANCE_CORRECTION, 0.0, self.main_model_part.Nodes)

        # AW 11.6-C: Recomputes the BDF2 time integration coefficients and stores them in ProcessInfo
        # Recompute the BDF2 coefficients
        (self.time_discretization).ComputeAndSaveBDFCoefficients(self.GetComputingModelPart().ProcessInfo)

        # AW 11.6-C: Moves (convects) the level-set distance field using the velocity from the previous time step.
        # Perform the level-set convection according to the previous step velocity
        self._PerformLevelSetConvection()

        KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, "Level-set convection is performed.")

        # AW 21.5: moved all of these lines before the intersection points utility is called

        # AW 11.6-C: Computes the gradient of the DISTANCE variable
        # filtering noises is necessary for curvature calculation
        # distance gradient is used as a boundary condition for smoothing process
        self._GetDistanceGradientProcess().Execute()

        # AW 11.6-C: If the _distance_smoothing flag is True, it runs a smoothing algorithm on the DISTANCE variable
        #AW 21.5: made smoothing conditional again
        if self._distance_smoothing:
            self._GetDistanceSmoothingProcess().Execute()
            KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, "Smoothing process is finished.")

            # AW 11.6-C: Recomputes the distance gradient after smoothing
            # distance gradient is called again to comply with the smoothed/modified DISTANCE
            self._GetDistanceGradientProcess().Execute()

        ########### AW 21.5: Function definitions ###################
        # AW-C 26.4: Defines a function to order unordered interface points into contours
        def order_interface_contour(cut_df, tol=1e-12):
            """
            Given a DataFrame of cut points with columns ['Element_ID','Point_ID','X','Y','Z'], and one interface point per row,
            returns a list of ordered (X,Y,Z) points describing the interface contour(s).
            """
            # 1) Represent each point by a tuple rounded to tolerance
            # AW-C 26.4: Helper to make a "key" from a point by rounding each coordinate to avoid floating point errors
            # when identifying equal points that differ due to numerical precision
            # simply speaking, rounds each coordinate to the nearest multiple of tol
            def key(pt):
                return (round(pt[0]/tol)*tol, round(pt[1]/tol)*tol, round(pt[2]/tol)*tol)
            
            # Build element → points and point_key → elements
            # AW-C 26.4: Maps each element ID to the list of points it owns; stores the two interface points for each element
            elem_points = defaultdict(list)
            # AW-C 26.4: Maps each rounded point (key) to all element IDs it belongs to; maps each (rounded) point to all elements that include it
            point_elems = defaultdict(list)
            
            # AW-C 26.4: Loops through all rows of the input DataFrame, filling both mappings above
            for _, row in cut_df.iterrows():
                e = row['Element_ID']
                pt = (row['X'], row['Y'], row['Z'])
                k = key(pt)
                elem_points[e].append(pt)
                point_elems[k].append(e)
            
            # 2) Find boundary points: keys present in only one element
            # loops through all point keys and selects only those where the point appears in exactly one element
            # (which uniquely identifies a boundary point by construction)
            boundary_keys = [k for k, elems in point_elems.items() if len(elems)==1]
            
            # 3) Helper to walk one contour starting from start_key
            # AW-C 26.4: starting from a key, follows the chain of connected points through the mesh, building up one ordered contour, and stops at a boundary or if it returns to the start (closed loop)
            def walk(start_key):
                # Initializes the list to store the ordered contour;
                # Each item will be a rounded point (3D tuple) visited during the walk
                contour = []
                # Set the starting key as the current point (the one that is the input)
                current_key = start_key
                # Remembers the last visited element ID to avoid going backward
                prev_elem = None
                
                # we'll break out manually when we reach a boundary or complete a loop
                while True:
                    # Add the current point to the contour we're building; ensures ordering according to visiting
                    contour.append(current_key)
                    # Get the list of element IDs that contain this point;
                    # Usually, len(elems) is 2 (internal point) or 1 (boundary)
                    elems = point_elems[current_key]
                    # If elems[0] is not the previous element, go to it
                    # Otherwise, try elems[1]
                    # If there is no other element (i.e., len(elems) == 1), this point is a boundary, and we stop
                    next_elem = elems[0] if elems[0]!=prev_elem else (elems[1] if len(elems)>1 else None)
                    # Stops the walk if there’s no other element to go to — this is the end of an open contour
                    if next_elem is None:
                        # This is an open contour, and we’re done
                        break  # reached boundary
                    # Retrieves the two interface points that belong to this next element (unordered)
                    pts = elem_points[next_elem]
                    # From the two points of this element, select the one that is not the current_key; This is the next point along the contour
                    other_pt = pts[0] if key(pts[0])!=current_key else pts[1]
                    # Round the next point to a key so it can be compared/stored consistently
                    other_key = key(other_pt)
                    
                    # Update prev_elem to the current one so we don’t backtrack in the next loop iteration
                    prev_elem = next_elem
                    # If we're back to where we started → it's a closed contour → exit
                    if other_key == start_key:
                        break  # closed loop
                    # Move forward to the next point in the contour
                    current_key = other_key
                
                # Convert the 3D keys back to 2D coordinates (X, Y only)
                return [ (kx, ky) for (kx, ky, kz) in contour ]#, kz
        
            # 4) Collect all contours
            # AW-C 26.4: Walks through all boundaries first (open curves), then through all remaining points (closed loops) to collect all contours
            # contours: will store the final list of ordered 2D point lists
            # idea is that this will also work for topological changes, e.g. droplet breakup and merging
            contours = []
            # visited: keeps track of points we've already used in a contour (using their rounded keys), to avoid duplicating them
            visited = set()
            # First walk from each boundary (open curve)
            for b in boundary_keys:
                if b in visited: continue
                # walk(b) → build a new contour starting from b
                c = walk(b)
                # Append it to contours
                contours.append(c)
                # Add all points from this contour to visited
                visited.update(c)
            # Then any closed loops
            # Now loop through every point key in point_elems
            for k in point_elems:
                if k in visited: continue
                c = walk(k)
                contours.append(c)
                visited.update(c)
            
            return contours
        

        def find_closest_u(curve, point, u_min=0.0, u_max=1.0):
            """
            Projects a Gauss point onto the NURBS curve by minimizing squared distance to the closest point on the curve
            
            Input:
            - curve: the NURBS curve object
            - Gauss point: np.array([x, y]) 
            - u_min, u_max: parameter range of the curve

            Output:
            - u_opt: optimal parameter value u where the curve is closest to the point
            """
            # function that computes the squared distance between a point on the curve and the input point
            def distance_squared(u):
                # get point on the curve at parameter u → curve.evaluate_single(u)
                pt = np.array(curve.evaluate_single(u)[:2]) # evaluates the 2D coordinates [x, y] of the curve at parameter u
                return np.sum((pt - point)**2) # np.sum(...) = (x1 - x0)² + (y1 - y0)² --> squared euclidean distance 

            # Uses scipy.optimize.minimize_scalar to minimize distance_squared(u) in the interval [umin,umax][umin​,umax​]
            # The method 'bounded' is used to ensure that the optimization is performed within the specified bounds
            # res.X contains the u value that minimizes the distance
            res = minimize_scalar(distance_squared, bounds=(u_min, u_max), method='bounded')
            return res.x  # optimal u



        ########### AW 21.5: End of function definitions #############

        

        # AW 8.5 Merge (Outcommented for now, purely for visualization)
        # Curvature correction (If needed)
        """ if ( self.settings["convection_diffusion_settings"]["Perform_conservative_law"].GetBool() == True ):
            self._Curvature_Correction() """
        

        # AW 2.6: user defined boolean to decide on normal penalty usage
        if self.do_normal_penalty:
            # debug print, outcomment if check needed
            # print("Normal Penalization is executed.")
            
            # set hydrophilic and hydrophobic contact angle to zero initially
            contact_angle_hydrophilic = 0.0
            contact_angle_hydrophobic = 0.0

            # retrieve the user-defined X_threshold, which separates hydrophilic and hydrophobic regimes
            X_threshold = self.main_model_part.ProcessInfo[KratosDroplet.X_threshold]

            # loop over all nodes and checks for an available contact angle micro (computed in C++, ST fct 2)
            for node in self.main_model_part.Nodes:
                # index 0 denotes current time step
                angle = node.GetSolutionStepValue(KratosDroplet.CONTACT_ANGLE_MICRO, 0)
                # AW 6.6: adapted to consider mixed wettability configurations (as well as generally, left and right contact angles)
                # retrieve the corresponding nodal X coordinate
                x = node.X
                # if the found angle is not zero, store it conditionally based on the X_threshold
                if angle != 0.0:
                    if x < X_threshold:
                        contact_angle_hydrophilic = angle
                        # debug print, outcomment if check needed
                        #print(f"Found contact angle on hydrophilic part of domain: {angle:.4f} at X = {x:.6f}")
                    elif x > X_threshold:
                        contact_angle_hydrophobic = angle
                        # debug print, outcomment if check needed
                        #print(f"Found contact angle on hydrophobic part of domain: {angle:.4f} at X = {x:.6f}")

            # AW 2.6: retrieve hydrophilic, hydrophobic contact angle and X_threshold
            theta_equilibrium_hydrophilic = self.main_model_part.ProcessInfo[KratosDroplet.theta_equilibrium_hydrophilic]
            theta_equilibrium_hydrophobic = self.main_model_part.ProcessInfo[KratosDroplet.theta_equilibrium_hydrophobic]

            # Define orientation and wall locations (for now; later, make this user-definable in PP.json)
            # AW 25.6: adaptations made to read everything from PP.json
            # main idea: normal computation differs based on if the wall is vertical or horizontal
            # AW 11.6
            is_horizontal_wall = self.is_horizontal_wall  # droplet sits on bottom/top wall
            # AW adapted 19.6
            x_left_wall = self.x_left_wall
            x_right_wall = self.x_right_wall
            y_bottom_wall = self.y_bottom_wall
            tol = self.tolerance_normal_penalty
            # AW 19.6: added to enforce to "fix" the static eq contact angle, in accordance with Gruending 2020
            fix_StaticEqContactAngle = self.fix_static_equilibrium

            # compute differenece of current contact angle to the equilibrium angle, considering hydrophilic and hydrophobic contact angle
            diff_hydrophilic = abs(contact_angle_hydrophilic - theta_equilibrium_hydrophilic)
            if diff_hydrophilic < 1:
                beta_hydrophilic = 1
            elif diff_hydrophilic > 9:
                beta_hydrophilic = 0
            else:
                # AW 6.6: 3.1416 replaced by math.pi
                beta_hydrophilic = 0.5*(1+math.cos(math.pi*(diff_hydrophilic-1.0)/8))
            print("beta_hydrophilic=",beta_hydrophilic)
            
            # AW 6.6: adapted to consider hydrophilic and hydrophobic contact angle
            diff_hydrophobic = abs(contact_angle_hydrophobic - theta_equilibrium_hydrophobic)
            if diff_hydrophobic < 1:
                beta_hydrophobic = 1
            elif diff_hydrophobic > 9:
                beta_hydrophobic = 0
            else:
                beta_hydrophobic = 0.5*(1+math.cos(math.pi*(diff_hydrophobic-1.0)/8))
            print("beta_hydrophobic=",beta_hydrophobic)

            # AW 19.6: added to enforce to "fix" the static eq contact angle, in accordance with Gruending 2020
            if fix_StaticEqContactAngle:
                beta_hydrophilic = 1
                beta_hydrophobic = 1
            
            # AW 4.6: adapted to also work for vertical walls
            # Loop over all nodes to penalize the distance gradient
            for node in self.main_model_part.Nodes:
                # retrieve distance gradient nodally
                gx = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X)
                gy = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Y)
                gz = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Z)
                # Normalize nodal distance gradients (if it is normalized, the normal and distance gradient are identical!)
                # AW 18.6: this added, as needed when cls is used
                g = (gx**2 + gy**2 + gz**2)**0.5
                if g!=0:
                    gx /= g
                    gy /= g
                    gz /= g
                    node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X, gx)
                    node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Y, gy)
                    node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Z, gz)
                else:
                    print("Zero distance gradients!")
                    node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X, 0)
                    node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Y, 0)
                    node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Z, 0)
                

                # AW 11.6: debug print, delete once works
                if not is_horizontal_wall:
                    # print(f"[DEBUG] node ID {node.Id}, — node.X = {node.X:.12f}, — node.Y = {node.Y:.12f}, |Y - 0.0| = {abs(node.Y - y_bottom_wall)}")
                    # AW 11.3: delete this print statement once it works
                    #  Get time info from ProcessInfo
                    """ time = self.main_model_part.ProcessInfo[KratosMultiphysics.TIME]
                    step = self.main_model_part.ProcessInfo[KratosMultiphysics.STEP]
                    print(f"[Step {step} | t = {time:.6f}] Unpenalized normal at X = {node.X:.6f}")
                    print("gx before: ", gx)
                    print("gy before: ", gy)
                    print("gz before: ", gz)
                                            """
                # Apply contact angle logic on horizontal bottom wall (Y ≈ 0)
                if is_horizontal_wall and abs(node.Y - y_bottom_wall) < tol:
                    # Penalize hydrophilic eq normal on the left wall (only if penalty coefficient is nonzero)
                    if node.X <= X_threshold and beta_hydrophilic > 0.0:
                        gy = math.cos(contact_angle_hydrophilic * math.pi / 180) + beta_hydrophilic * (
                            math.cos(theta_equilibrium_hydrophilic * math.pi / 180) - math.cos(contact_angle_hydrophilic * math.pi / 180)
                        )
                        gx = math.sin(contact_angle_hydrophilic * math.pi / 180) + beta_hydrophilic * (
                            math.sin(theta_equilibrium_hydrophilic * math.pi / 180) - math.sin(contact_angle_hydrophilic * math.pi / 180)
                        )
                        if node.X < self.reference_point_x:
                            gx *= -1
                    elif node.X > X_threshold and beta_hydrophobic > 0.0:
                        gy = math.cos(contact_angle_hydrophobic * math.pi / 180) + beta_hydrophobic * (
                            math.cos(theta_equilibrium_hydrophobic * math.pi / 180) - math.cos(contact_angle_hydrophobic * math.pi / 180)
                        )
                        gx = math.sin(contact_angle_hydrophobic * math.pi / 180) + beta_hydrophobic * (
                            math.sin(theta_equilibrium_hydrophobic * math.pi / 180) - math.sin(contact_angle_hydrophobic * math.pi / 180)
                        )
                        if node.X < self.reference_point_x:
                            gx *= -1

                # Apply contact angle logic on vertical side walls (X ≈ left or right)
                if not is_horizontal_wall:
                    # LEFT wall
                    if abs(node.X - x_left_wall) < tol:
                        # AW 6.6: corrected this part
                        if node.X <= X_threshold and beta_hydrophilic > 0.0:
                            gy = math.cos(contact_angle_hydrophilic * math.pi / 180) + beta_hydrophilic * (
                            math.cos(theta_equilibrium_hydrophilic * math.pi / 180) - math.cos(contact_angle_hydrophilic * math.pi / 180)
                            )
                            gx = math.sin(contact_angle_hydrophilic * math.pi / 180) + beta_hydrophilic * (
                                math.sin(theta_equilibrium_hydrophilic * math.pi / 180) - math.sin(contact_angle_hydrophilic * math.pi / 180)
                            )
                            # AW adapted 11.6: gx flipping always needed on left wall
                            # AW 17.6: readapted to be in accordance with latest commit
                            if node.X < self.reference_point_x:
                                gx *= -1
                         # AW 6.6: corrected this part
                        elif node.X > X_threshold and beta_hydrophobic > 0.0:
                            gy = math.cos(contact_angle_hydrophobic * math.pi / 180) + beta_hydrophobic * (
                            math.cos(theta_equilibrium_hydrophobic * math.pi / 180) - math.cos(contact_angle_hydrophobic * math.pi / 180)
                            )
                            gx = math.sin(contact_angle_hydrophobic * math.pi / 180) + beta_hydrophobic * (
                                math.sin(theta_equilibrium_hydrophobic * math.pi / 180) - math.sin(contact_angle_hydrophobic * math.pi / 180)
                            )
                            # AW 17.6: readapted to be in accordance with latest commit
                            if node.X < self.reference_point_x:
                                gx *= -1
                        # Clockwise 90 deg rotation for left wall
                        gx, gy = gy, -gx

                        # AW 19.6
                        print("Normal enforced at left wall; gx,gy components: ", gx, gy)


                        
 
                    # RIGHT wall
                    elif abs(node.X - x_right_wall) < tol:
                         # AW 6.6: corrected this part
                        if node.X <= X_threshold and beta_hydrophilic > 0.0:
                            gy = math.cos(contact_angle_hydrophilic * math.pi / 180) + beta_hydrophilic * (
                            math.cos(theta_equilibrium_hydrophilic * math.pi / 180) - math.cos(contact_angle_hydrophilic * math.pi / 180)
                            )
                            gx = math.sin(contact_angle_hydrophilic * math.pi / 180) + beta_hydrophilic * (
                                math.sin(theta_equilibrium_hydrophilic * math.pi / 180) - math.sin(contact_angle_hydrophilic * math.pi / 180)
                            )
                            # AW 17.6: readapted to be in accordance with latest commit
                            if node.X < self.reference_point_x:
                                gx *= -1
                         # AW 6.6: corrected this part
                        elif node.X > X_threshold and beta_hydrophobic > 0.0:
                            gy = math.cos(contact_angle_hydrophobic * math.pi / 180) + beta_hydrophobic * (
                            math.cos(theta_equilibrium_hydrophobic * math.pi / 180) - math.cos(contact_angle_hydrophobic * math.pi / 180)
                            )
                            gx = math.sin(contact_angle_hydrophobic * math.pi / 180) + beta_hydrophobic * (
                                math.sin(theta_equilibrium_hydrophobic * math.pi / 180) - math.sin(contact_angle_hydrophobic * math.pi / 180)
                            )
                           # AW 17.6: readapted to be in accordance with latest commit
                            if node.X < self.reference_point_x:
                                gx *= -1

                        # rotate by 90deg counter-clockwise for rightvertical wall
                        gx, gy = -gy, gx

                        # AW 11.3: delete this print statement once it works
                        """  print(f"[Step {step} | t = {time:.6f}] Penalized normal at X = {node.X:.6f}")
                        print("  gx after:", gx)
                        print("  gy after:", gy)
                        print("  gz after:", gz) """

                        # AW 19.6
                        print("Normal enforced at right wall; gx,gy components: ", gx, gy)
               
                # Normalize again and set
                g = (gx**2 + gy**2 + gz**2)**0.5
                # AW 18.6: this added for the cls method to work
                if g!=0:
                    gx /= g
                    gy /= g
                    gz /= g
                    node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X, gx)
                    node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Y, gy)
                    node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Z, gz)
                else:
                    print("Zero distance gradients!")
                    node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X, 0)
                    node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Y, 0)
                    node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Z, 0)
               

            ####################### end of normal penalization ###################
        
          # AW 6.6: updated this to reset contact velocity and contact angle micro at the beginning of every time step
        # (this is done for improved output debugging; notably: has to be done after normal penalization)
        # AW-C 2.7: idea here is to reset contact angle micro and contact velocity, such that every time step we only store nodally where
        # we actually compute both of these variables in the C++ implementation
        for node in self.main_model_part.Nodes:
            node.SetSolutionStepValue(KratosDroplet.CONTACT_ANGLE_MICRO, 0.0)
            node.SetSolutionStepValue(KratosDroplet.CONTACT_VELOCITY, 0.0)
        
        # curvature is calculated using nodal distance gradient
        self._GetDistanceCurvatureProcess().Execute()

        ##########
        # Contact angle calculation
        # self._GetContactAngleEvaluatorProcess().Execute()
        # Store current level-set to check for wetting/dewetting used in contact_angle_evaluator
        for node in self.main_model_part.Nodes:
            old_distance = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE)
            node.SetValue(KratosDroplet.DISTANCE_AUX, old_distance)
        # print("Contact Angle Evaluator: Finished")
        ##########
        
         # it is needed to store level-set consistent nodal PRESSURE_GRADIENT for stabilization purpose
        self._GetConsistentNodalPressureGradientProcess().Execute()

        # TODO: Performing mass conservation check and correction process

        # Perform distance correction to prevent ill-conditioned cuts
        self._GetDistanceModificationProcess().ExecuteInitializeSolutionStep()

        # Update the DENSITY and DYNAMIC_VISCOSITY values according to the new level-set
        self._SetNodalProperties()

        # AW 15.5: everything related to inters points, fitting, normal averaging AFTER the distance modification process as of now!!!

        # AW 19.5: Reload fitting settings from ProcessInfo
        fitting_type = self.main_model_part.ProcessInfo[KratosDroplet.FittingType]
        normal_evaluation_mode = self.main_model_part.ProcessInfo[KratosDroplet.NormalEvaluationMode]

        # AW 19.5: Only run Intersection Points utility if fitting_type is explicitly set to "nurbs" (needed for nurbs fitting) or normal_evaluation_mode==3 (needed for normal averaging) or curvature+normal smoothing is activated
        if fitting_type == "nurbs" or normal_evaluation_mode == 2 or normal_evaluation_mode == 3 or self.do_curvature_normal_smoothing:
            # Debug print: delete once it works
            print(f"IntersectionPointsUtility is executed because fitting_type = '{fitting_type}' or normal_evaluation_mode = {normal_evaluation_mode} or curvature-smoothing = '{self.do_curvature_normal_smoothing}'.")
  
            # Clear any existing intersection points from previous steps
            KratosDroplet.IntersectionPointsUtility.ClearIntersectionPoints()
        
            # Collect intersection points from all elements
            for element in self.main_model_part.Elements:
                KratosDroplet.IntersectionPointsUtility.CollectElementIntersectionPoints(element)
            
            # AW 5.5: Run diagnostic to check how many elements are split by the level-set; Delete once not needed anymore
            KratosDroplet.IntersectionPointsUtility.DiagnosticOutput(self.main_model_part)
        
            # Get all intersection points
            points = KratosDroplet.IntersectionPointsUtility.GetIntersectionPoints()
            KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, f"Collected {len(points)} intersection points.")
        
            # Save intersection points to file
            KratosDroplet.IntersectionPointsUtility.SaveIntersectionPointsToFile("intersection_points.txt")

        # AW-C: Nurbs fitting executed either if fitting_type was explicitly set to "nurbs" or if normal_evaluation_mode is 2
        if fitting_type == "nurbs" or normal_evaluation_mode == 2:
            # Debug statement: delete once it works
            print("NURBS fitting is executed because fitting_type is set to 'nurbs' or normal_evaluation_mode set to 2.")

            ##################### Beginning of Nurbs Fitting ########################
           
            ##### Part 1: Read intersection points and order them into a continuous interface contour ######
            # AW_C 26.4: Reads the file intersection_points.txt as a table, using tab characters as separators
            data = pd.read_csv('intersection_points.txt', sep="\t")
            # AW-C 26.4: function that takes raw intersection points and orders them into continuous interface lines (open or closed contours)
            contours = order_interface_contour(data)
            # AW-C 26.4: picks the first contour (in case there are several disconnected ones
            # --> we consequently assume one interface currently
            cut_points = contours[0]
            # AW-C 26.4: debug print, to check basic functionality
            # print(cut_points)

            ##### Part 2: Globally fit a NURBS curve of specified order to the ordered contour #####
            # AW-C 26.4: currently, we fit a quadratic NURBS curve (parabolic segments) 
            degree = 2  # quadratic NURBS
            # AW-C 26.4: sets the number of control points for the NURBS curve (this controls the flexibility of the fit)
            ctrlpts_size= 13
            # AW-C 4.7: before trying Least squares fit, try an exact fit through data points conditionally
            try_exact_fit = False
            if try_exact_fit:
                try:
                    print("Trying exact interpolation first...")
                    curve = fitting.interpolate_curve(cut_points, degree=degree)
                    print("Interpolation successful.")
                    success = True
                except ZeroDivisionError:
                    print("[WARNING] Interpolation failed due to ZeroDivisionError. Proceeding to approximation...")
                    success = False
            else:
                success = False


            # AW 4.7: commented this line for now, as success variable already set based on the prior code part trying exact interpolation
            # success = False

            # AC-C 4.7: minimum number of control points set to 6
            max_attempts = ctrlpts_size - 5  # number of fallback attempts allowed
            if not success:
                 # AW-C 26.4: if exact fitting fails, use an approximate fit (least squares) with the given control point number and degree, which is more robust for noisy or incomplete data
                print("Entering into Least Squares based curve fitting.")
                for offset in range(max_attempts + 1):  # try with ctrlpts_size, ctrlpts_size-1, ...
                    try:
                        current_ctrlpts_size = ctrlpts_size - offset
                        if current_ctrlpts_size < 6:
                            print("[ERROR] Cannot fit curve: control point count dropped below 6.")
                            break
                        print(f"Trying with ctrlpts_size = {current_ctrlpts_size}")
                        curve = fitting.approximate_curve(cut_points, degree=degree, ctrlpts_size=current_ctrlpts_size)
                        print("Least Squares based Curve fitting done.")
                        success = True
                        break
                    except ZeroDivisionError:
                        print(f"[WARNING] Curve fitting failed with ctrlpts_size = {current_ctrlpts_size}, trying fewer points...")

            if not success:
                raise RuntimeError("Curve fitting failed for all attempted control point sizes.")


            ##### Part 3: (Optional) Visualization ######

            plot_nurbs = True

            if plot_nurbs:
                # AW-C 26.4: curve.delta sets the sampling resolution when evaluating the curve (smaller = more points, higher detail)
                # --> solely for postprocessing
                curve.delta = 0.0001  # evaluation resolution
        
                # Convert order interface points contour to NumPy array
                cut_points_np = np.array(cut_points)

                # AW 2.5: this part added for debug visualization
                curve.evaluate()

                # Fitted B-Spline curve points
                fitted_points = np.array(curve.evalpts)  # evaluated using delta = 0.01

                # Control points
                control_points = np.array(curve.ctrlpts)  # shape (ctrlpts_size, 2)

                # Plot all in one figure
                plt.figure(figsize=(6, 6))
                plt.axis("equal")

                # 2. Fitted NURBS curve
                plt.plot(fitted_points[:, 0], fitted_points[:, 1], 'r-', linewidth=2, label='Fitted B-Spline Curve')

                # 1. Original ordered cut points (interface contour)
                plt.plot(cut_points_np[:, 0], cut_points_np[:, 1], 'bo-', markersize=3, label='Ordered Cut Points')

                # 3. Control points
                plt.plot(control_points[:, 0], control_points[:, 1], 'ko--', label='Control Points', markersize=4)

                # Labels and legend
                plt.title("B-Spline Fitting of Interface Contour")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()

                # Save and show
                plt.savefig("interface_fitting_nurbs_debug.png", dpi=300)
                plt.show()



            ##### Part 5: Compute curvature and normals at Gauss points using nearest neighbour search
            
            # this creates a defaultdict from the collections module
            # Idea: If it tries to access a key that doesn't exist, it will automatically create an empty list as the default value
            # dictionary will map: element ID → list of intersection points (tuples like (x, y))
            # --> generally, cleaner and safer than checking for existence, and appending conditionally
            element_intersections = defaultdict(list)

            # Iterate over each row of the order interface points
            # data.iterrows() returns a (row_index, row_data) tuple, and _ is used to ignore the row index
            for _, row in data.iterrows():
                # Extracts the element ID from the row and casts it to an integer
                # ID will be used as the key in the element_intersections dictionary
                eid = int(row['Element_ID'])
                # Extracts the x and y coordinates from the row and stores them as a tuple
                point = (row['X'], row['Y'])
                # Appends this (x, y) point to the list associated with eid in element_intersections
                # If this is the first time eid appears, the defaultdict automatically creates a new list
                element_intersections[eid].append(point)

            # Initializes a dictionary to store curvature and normal values per element
            # Keys will be element IDs, and values will be dictionaries like:
            # { "curvatures": [k1, k2], "normals": [(nx1, ny1), (nx2, ny2)] }
            element_gp_data = {}


            # Opens the file "element_gp_curvatures_normals.csv" in write mode; f is the file handle (file will be either created or overwritten)
            with open("element_gp_curvatures_normals.csv", "w") as f:
                # Writes the header row for the CSV file
                f.write("ElementID,kappa1,kappa2,nx1,ny1,nx2,ny2,gp1_x,gp1_y,gp2_x,gp2_y\n")

                # Loops through each element ID (eid) and its associated list of intersection points (pts)
                for eid, pts in element_intersections.items():
                    # Only processes elements that have exactly 2 intersection points (actually not needed, as every element has two intersection points anyways for our current configuration)
                    if len(pts) != 2:
                        continue  # skip degenerate elements
                    
                    # Converts the two intersection points to NumPy arrays for vector math;
                    # p0 and p1 are the endpoints of the interface segment in this element
                    p0, p1 = np.array(pts[0]), np.array(pts[1])

                    # Gauss points in natural coords: ξ = ±1/√3
                    # standard Gauss-Legendre quadrature points for a linear element
                    gauss_weights = [-1/np.sqrt(3), 1/np.sqrt(3)]
                    # Maps each natural coordinate ξ to a physical Gauss point on the interface segment [p0, p1] using linear interpolation
                    # uses the isoparametric mapping from reference element to physical space.
                    gauss_points = [(0.5 * ((1 - xi) * p0 + (1 + xi) * p1)) for xi in gauss_weights]

                    # Initializes lists to store the computed curvature and normal vector for each Gauss point
                    curvatures = []
                    normals = []

                    # Store the actual coordinates of each Gauss point
                    gp_coords = []

                    # Loops over the two Gauss points just computed
                    for gp in gauss_points:

                        # AW 4.7: Note, this next block was adapted to adjust for correct endpoints

                        # we skip evaluating at the exact endpoints by using the inner range:
                        # u_min = curve.knotvector[p+1] → the first non-clamped parameter
                        # u_max = curve.knotvector[-(p+2)] → the last non-clamped parameter
                        u_min = curve.knotvector[ curve.degree + 1 ]
                        u_max = curve.knotvector[ -(curve.degree + 2) ]
                        # Returns the parametric coordinate u_gp of the closest point on the curve
                        u_gp = find_closest_u(curve, gp, u_min=u_min, u_max=u_max)

                        # Computes the 0th, 1st, and 2nd derivatives of the NURBS curve at parameter u_gp
                        # d1 is the tangent vector
                        # d2 is the second derivative (used for curvature)
                        d0, d1, d2 = curve.derivatives(u_gp, order=2)
                        # Extracts the x and y components of the first and second derivatives
                        x1, y1 = d1[0], d1[1]
                        x2, y2 = d2[0], d2[1]
                        # Computes 2D curvature at that point using the standard formula
                        num = abs(x1 * y2 - y1 * x2)
                        den = (x1 ** 2 + y1 ** 2) ** 1.5
                        # Avoids division by zero if the denominator is zero
                        curvature = num / den if den != 0 else 0.0
                        # Computes the normal vector as perpendicular to the tangent
                        # computes the Euclidean norm (or length) of the 2D vector
                        tnorm = np.hypot(x1, y1)
                        nx, ny = (y1 / tnorm, -x1 / tnorm) if tnorm != 0 else (0.0, 0.0) # outward unit normal

                        ########## AW 28.5: this part added to enforce normal vector to point outwards from domain
                        normal_vec = np.array([nx, ny])
                        # Reference vector from center to current Gauss point
                        ref_vec = np.array([gp[0] - self.reference_point_x, gp[1] - self.reference_point_y])

                        # uses the dot product to check if normal_vec is pointing toward the reference point (i.e., inward)
                        if np.dot(ref_vec, normal_vec) < 0:
                            # Flip normal if it's pointing inward
                            normal_vec *= -1
                        nx, ny = normal_vec
                        normal = (nx, ny)
                        ######## AW 28.5: end of this additional impl  

                        curvatures.append(curvature)
                        normals.append(normal)
                        gp_coords.append(gp)  # store actual gauss point (x, y)

                    # Save to dictionary for elemental assignment
                    element_gp_data[eid] = {
                        "curvatures": curvatures,
                        "normals": normals
                    }

                    # Write to file
                    k1, k2 = curvatures
                    (nx1, ny1), (nx2, ny2) = normals
                    (gp1_x, gp1_y), (gp2_x, gp2_y) = gp_coords

                    f.write(f"{eid},{k1},{k2},{nx1},{ny1},{nx2},{ny2},{gp1_x},{gp1_y},{gp2_x},{gp2_y}\n")

            print("Saved curvature and normals at Gauss points to element_gp_curvatures_normals.csv")

            ##### Part 6: Store curvatures and normals per GaussPoint for accessibility in c++ implementation

            # Loops through all elements in the main_model_part (the computational mesh in Kratos)
            for element in self.main_model_part.Elements:
                # Retrieves the element ID for the current element. This will match the keys in element_gp_data
                eid = element.Id
                # If this element doesn't have any stored Gauss point data (e.g., it wasn't cut by the interface), skip it
                if eid not in element_gp_data:
                    continue
                
                # Retrieves the list of two curvature values and the list of two normal vectors for this element (one for each Gauss point)
                curvatures = element_gp_data[eid]["curvatures"]
                normals = element_gp_data[eid]["normals"]

                # Stores the curvature at Gauss point 1 and 2 as element-level scalar values using Kratos’ SetValue() function 
                element.SetValue(KratosDroplet.CURVATURE_FITTED_GAUSS1, curvatures[0])
                element.SetValue(KratosDroplet.CURVATURE_FITTED_GAUSS2, curvatures[1])
                # does the same for normals; [*normals[0], 0.0] expands the 2D tuple (nx, ny) into a 3D vector by adding a 0 in the z-direction (since Kratos expects 3D vectors even in 2D problems)
                element.SetValue(KratosDroplet.NORMAL_FITTED_GAUSS1, [*normals[0], 0.0])
                element.SetValue(KratosDroplet.NORMAL_FITTED_GAUSS2, [*normals[1], 0.0])

                print(f"Element {eid} - kappa1 = {element.GetValue(KratosDroplet.CURVATURE_FITTED_GAUSS1)}, kappa2 = {element.GetValue(KratosDroplet.CURVATURE_FITTED_GAUSS2)}")

        ##################### End of Nurbs Fitting ########################

        ################### AW 22.5: Start of Unfitted Curvature Smoothing ###################
        
        # user-defined flag to control whether curvature smoothing is performed
        if self.do_curvature_normal_smoothing:
           
            from scipy.signal import savgol_filter
            # Step 1: Load intersection points and recover a ordered contour of points along the interface
            # Reads the file intersection_points.txt as a table, using tab characters as separators.
            data = pd.read_csv('intersection_points.txt', sep="\t")
            # function that takes raw intersection points and orders them into continuous interface lines (contours)
            contours = order_interface_contour(data)
            # picks the first contour (in case there are several disconnected ones); assumes one interface currently
            cut_points = contours[0]
            # now a list of [x, y] coordinates tracing the interface, in order
            # print(cut_points)

            # Defines a helper function to map each (x, y) point in cut_points to its corresponding Element_ID from the intersection data
            def match_cut_points_to_elements(cut_points, intersection_data, tol=1e-12):
                """
                Given ordered cut_points and the full intersection_points DataFrame,
                returns a list of Element_IDs in the same order.
                """
                ordered_element_ids = []
                used_elements = set()

                for cut_x, cut_y in cut_points:
                    # Search for matching point in intersection data
                    match = intersection_data[
                        (np.abs(intersection_data['X'] - cut_x) < tol) &
                        (np.abs(intersection_data['Y'] - cut_y) < tol)
                    ]

                    if not match.empty:
                        # Pick the first matching element that's not yet used
                        for _, row in match.iterrows():
                            elem_id = int(row['Element_ID'])
                            if elem_id not in used_elements:
                                ordered_element_ids.append(elem_id)
                                used_elements.add(elem_id)
                                break  # Go to next cut point
                    else:
                        print(f"[WARNING] No match found for cut point ({cut_x}, {cut_y})")

                return ordered_element_ids



            # Step 3: Map cut points to ordered Element_IDs
            ordered_elem_ids = match_cut_points_to_elements(cut_points, data)



            def smooth_interface_curvature_and_gradient(model_part, ordered_elem_ids, curvature_var, gradient_var,
                                                        window_size=5, polyorder=2, method='savgol'):
                """
                Smooths nodal curvature and distance gradient using Savitzky–Golay or moving average.
                
                Parameters:
                    model_part       : Kratos ModelPart
                    ordered_elem_ids : list of interface element IDs
                    curvature_var    : Kratos Variable for curvature (scalar)
                    gradient_var     : Kratos Variable for distance gradient (vector)
                    window_size      : smoothing window size (odd int)
                    polyorder        : used for Savitzky–Golay (ignored if method='average')
                    method           : 'savgol' or 'average'
                """

                # AW 2.6: print statement added
                KratosMultiphysics.Logger.PrintInfo(
                    "smooth_interface_curvature_and_gradient",
                    f"Curvature smoothing is executed with method='{method}', window_size={window_size}, polyorder={polyorder}."
                )

                def moving_average(data, window_size):
                    pad = window_size // 2
                    padded = np.pad(data, pad_width=pad, mode='edge')
                    return np.convolve(padded, np.ones(window_size) / window_size, mode='valid')

                def apply_filter(data, window_size, polyorder):
                    if len(data) < window_size:
                        return data
                    if method == 'savgol':
                        return savgol_filter(data, window_length=window_size, polyorder=polyorder)
                    else:
                        return moving_average(data, window_size)

                # Step 1: ordered unique nodes
                ordered_nodes = []
                visited_node_ids = set()
                prev_nodes = None

                for elem_id in ordered_elem_ids:
                    elem = model_part.GetElement(elem_id)
                    current_nodes = list(elem.GetGeometry())

                    if prev_nodes:
                        prev_ids = {n.Id for n in prev_nodes}
                        shared = [n for n in current_nodes if n.Id in prev_ids]
                        if shared:
                            shared_id = shared[0].Id
                            idx = next(i for i, n in enumerate(current_nodes) if n.Id == shared_id)
                            current_nodes = current_nodes[idx:] + current_nodes[:idx]

                    for node in current_nodes:
                        if node.Id not in visited_node_ids:
                            ordered_nodes.append(node)
                            visited_node_ids.add(node.Id)

                    prev_nodes = current_nodes

                # --- CURVATURE ---
                kappa_before = {node.Id: node.GetValue(curvature_var) for node in ordered_nodes}
                kappa_vals = np.array(list(kappa_before.values()))
                kappa_smoothed = apply_filter(kappa_vals, window_size, polyorder)
                kappa_after = {}

                for node, val in zip(ordered_nodes, kappa_smoothed):
                    node.SetValue(curvature_var, val)
                    kappa_after[node.Id] = val

                with open("curvature_smoothing_log.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Element_ID", "Node_ID", "Curvature_Before", "Curvature_After"])
                    for elem_id in ordered_elem_ids:
                        elem = model_part.GetElement(elem_id)
                        for node in elem.GetGeometry():
                            nid = node.Id
                            if nid in kappa_before:
                                writer.writerow([elem_id, nid, kappa_before[nid], kappa_after[nid]])

                # --- DISTANCE GRADIENT ---
                grad_before = {node.Id: list(node.GetSolutionStepValue(gradient_var)) for node in ordered_nodes}

                grad_components = [[], [], []]
                for node in ordered_nodes:
                    vec = grad_before[node.Id]
                    for i in range(3):
                        grad_components[i].append(vec[i])

                grad_smoothed = [apply_filter(np.array(comp), window_size, polyorder) for comp in grad_components]

                grad_after = {}
                for i, node in enumerate(ordered_nodes):
                    smoothed_vec = np.array([grad_smoothed[0][i], grad_smoothed[1][i], grad_smoothed[2][i]])
                    node.SetSolutionStepValue(gradient_var, smoothed_vec)
                    grad_after[node.Id] = smoothed_vec

                with open("distance_gradient_smoothing_log.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Element_ID", "Node_ID",
                                    "GradX_Before", "GradY_Before", "GradZ_Before",
                                    "GradX_After", "GradY_After", "GradZ_After"])
                    for elem_id in ordered_elem_ids:
                        elem = model_part.GetElement(elem_id)
                        for node in elem.GetGeometry():
                            nid = node.Id
                            if nid in grad_before:
                                g_b = grad_before[nid]
                                g_a = grad_after[nid]
                                writer.writerow([elem_id, nid, g_b[0], g_b[1], g_b[2], g_a[0], g_a[1], g_a[2]])

            # AW 2.6: updated to use user-defined variables from PP.json

            smooth_interface_curvature_and_gradient(
                self.main_model_part,
                ordered_elem_ids,
                KratosCFD.CURVATURE,
                KratosMultiphysics.DISTANCE_GRADIENT,
                window_size=self.curvature_normal_smoothing_window_size,
                polyorder=self.curvature_normal_smoothing_poly_order,
                method=self.curvature_normal_smoothing_method
            )

        ################### AW 22.5: End of Unfitted Curvature Smoothing ###################

   

        # AW 19.5: execute normal averaging only when explicitly set
        if normal_evaluation_mode == 3:
            # debug print, delete once it works
            print("Normal averaging is executed because normal_evaluation_mode is set to 3.")
            ####################### Normal averaging added ##########################
            # AW 14.5: normal averaging added

            # INTERSECTION LENGTH CALCULATION - CORRECT VERSION
            timestamp = self.main_model_part.ProcessInfo[KratosMultiphysics.TIME]
            points_filename = f"intersection_points_{timestamp:.6f}.txt"
            combined_filename = f"intersection_data_{timestamp:.6f}.txt"
            averaged_normals_filename = f"averaged_normals_{timestamp:.6f}.txt"

            # Step 1: Clear and collect intersection points
            KratosDroplet.IntersectionPointsUtility.ClearIntersectionPoints()
            for element in self.main_model_part.Elements:
                KratosDroplet.IntersectionPointsUtility.CollectElementIntersectionPoints(element)

            # Step 2: Save intersection points to file
            #KratosDroplet.IntersectionPointsUtility.SaveIntersectionPointsToFile(points_filename)
            # KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, f"Saved intersection points to {points_filename}")

            # Step 3: Calculate and store element intersection lengths
            num_elements = KratosDroplet.CalculateAndStoreElementIntersectionLengths(self.main_model_part)
            #KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, 
            #                                    f"Calculated and stored intersection lengths for {num_elements} elements")

            # Step 4: Calculate interface averages
            KratosDroplet.InterfaceAveragesUtility.ClearInterfaceAverages()
            KratosDroplet.InterfaceAveragesUtility.ComputeModelPartInterfaceAverages(self.main_model_part)

            # Step 5: Collect intersection data with normals
            num_elements = KratosDroplet.CollectIntersectionDataWithNormal(self.main_model_part)
            #KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, 
            #                                    f"Collected {num_elements} elements with intersection data and normals")

            # Step 6: Save the collected data to file
            #KratosDroplet.SaveIntersectionDataWithNormalToFile(combined_filename)
            #KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, 
            #                                    f"Saved intersection data with normals to {combined_filename}")

            # Step 7: Set cut normals on elements
            num_elements_with_normals = KratosDroplet.SetElementCutNormals(self.main_model_part)
            #KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__,
            #                                f"Set cut normals for {num_elements_with_normals} elements")
            
            # Step 8: Compute averaged normals with 1 neighbor level
            num_elements_averaged = KratosDroplet.ComputeAndStoreAveragedNormals(
                self.main_model_part, 2, "ELEMENT_CUT_NORMAL_AVERAGED")
            #KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__,
            #    f"Computed averaged normals for {num_elements_averaged} elements")
        
            # Step 9: Save the averaged normals to file with timestamped filename
            #KratosDroplet.SaveAveragedNormalsToFile(self.main_model_part, averaged_normals_filename)
            #KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__,
            #    f"Saved averaged normals to {averaged_normals_filename}")
            ###############################
            # Step 4: Read the intersection lengths from elements and save to file
            # Create a map to pass to SaveIntersectionLengthsToFile
            intersection_lengths = {}
            for element in self.main_model_part.Elements:
                length = KratosDroplet.GetElementIntersectionLength(element)
                if length > 0.0:  # Only save elements that have a valid intersection length
                    intersection_lengths[element.Id] = length
        
            # # Save to file
            #KratosDroplet.SaveIntersectionLengthsToFile(intersection_lengths, lengths_filename)
            #KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, 
            #                                    f"Saved {len(intersection_lengths)} intersection lengths to {lengths_filename}")
            

            # # Save element average normals to file
            #normals_filename = f"element_normals_{timestamp:.6f}.txt"
            #KratosDroplet.SaveElementAverageNormalsToFile(self.main_model_part, normals_filename)
            #KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, 
            #                                    f"Saved element average normals to {normals_filename}")
            
            # # Clear any existing intersection points from previous steps
            # KratosDroplet.IntersectionPointsUtility.ClearIntersectionPoints()
        
            # # Collect intersection points from all elements
            # for element in self.main_model_part.Elements:
            #     KratosDroplet.IntersectionPointsUtility.CollectElementIntersectionPoints(element)
            
            # # # Run diagnostic to check how many elements are split by the level-set
            # # KratosDroplet.IntersectionPointsUtility.DiagnosticOutput(self.main_model_part)
        
            # # Get all intersection points
            # points = KratosDroplet.IntersectionPointsUtility.GetIntersectionPoints()
            # KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, f"Collected {len(points)} intersection points.")
        
            # # Save intersection points to file
            # KratosDroplet.IntersectionPointsUtility.SaveIntersectionPointsToFile("intersection_points.txt")
            # AW 14.5: end of the additional normal averaging
            ####################### End of normal averaging #########################

        

        
        


      

        # Initialize the solver current step
        self._GetSolutionStrategy().InitializeSolutionStep()

        # We set this value at every time step as other processes/solvers also use them
        dynamic_tau = self.settings["formulation"]["dynamic_tau"].GetDouble()
        self.main_model_part.ProcessInfo.SetValue(KratosMultiphysics.DYNAMIC_TAU, dynamic_tau)


    def Predict(self):
        self._GetSolutionStrategy().Predict()


    # AW 10.6: outcommented this for leveque test
    def SolveSolutionStep(self):
        is_converged = self._GetSolutionStrategy().SolveSolutionStep()
        if not is_converged:
            msg  = "Droplet dynamics solver did not converge for step " + str(self.main_model_part.ProcessInfo[KratosMultiphysics.STEP]) + "\n"
            msg += "corresponding to time " + str(self.main_model_part.ProcessInfo[KratosMultiphysics.TIME]) + "\n"
            KratosMultiphysics.Logger.PrintWarning(self.__class__.__name__, msg)
        return is_converged 


    def FinalizeSolutionStep(self):
        KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, "Mass and momentum conservation equations are solved.")
        
        # AW 8.5 Merge
        # Performing The Conservative Law
        if ( self.settings["convection_diffusion_settings"]["Perform_conservative_law"].GetBool() == True ):
            step = self.main_model_part.ProcessInfo[KratosMultiphysics.STEP]
            self._PerformConservativeLaw()
            self._SetNodalProperties()
            self._GetDistanceGradientProcess().Execute()

        # Recompute the distance field according to the new level-set position
        # AW 3.6: adapted for redistancing

        # AW 3.6: old code
        """  if self._reinitialization_type != "none":
            step = self.main_model_part.ProcessInfo[KratosMultiphysics.STEP]
            if step == 2 or step%12==0:
                self._GetDistanceReinitializationProcess().Execute()
                KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, "Redistancing process is finished.") """
        
        # AW 3.6: new code
        if self._reinitialization_type != "none" and self.main_model_part.ProcessInfo[KratosMultiphysics.STEP] % 50 == 0:
            self._GetDistanceReinitializationProcess().Execute()
            KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, "Redistancing process is finished.")

        

        # Prepare distance correction for next step
        self._GetDistanceModificationProcess().ExecuteFinalizeSolutionStep()

        # Finalize the solver current step
        self._GetSolutionStrategy().FinalizeSolutionStep()
        # Limit the obtained acceleration for the next step
        # This limitation should be called on the second solution step onwards (e.g. STEP=3 for BDF2)
        # We intentionally avoid correcting the acceleration in the first resolution step as this might cause problems with zero initial conditions
        if self._apply_acceleration_limitation and self.main_model_part.ProcessInfo[KratosMultiphysics.STEP] >= self.min_buffer_size:
            self._GetAccelerationLimitationUtility().Execute()

        


    def Check(self):
        self._GetSolutionStrategy().Check()


    def Clear(self):
        self._GetSolutionStrategy().Clear()


    def _SetPhysicalProperties(self):   # Might be removed from the NavierStokesTwoFluidsSolver!!! 
                                        # It is just a duplicate of the same routine defined in NavierStokesTwoFluidsSolver
        warn_msg  = '\nThe materials import mechanism used in the two fluids solver is DEPRECATED!\n'
        warn_msg += 'It will be removed to use the base fluid_solver.py one as soon as the subproperties are available.\n'
        KratosMultiphysics.Logger.PrintWarning('\n\x1b[1;31mDEPRECATION-WARNING\x1b[0m', warn_msg)

        # Check if the fluid properties are provided using a .json file
        materials_filename = self.settings["material_import_settings"]["materials_filename"].GetString()
        if (materials_filename != ""):
            data_comm = KratosMultiphysics.ParallelEnvironment.GetDefaultDataCommunicator()

            def GetAuxMaterialsFileName(mat_file_name, prop_id):
                p_mat_file_name = Path(mat_file_name)
                new_stem = "{}_p{}".format(p_mat_file_name.stem, prop_id)
                return str(p_mat_file_name.with_name(new_stem).with_suffix(p_mat_file_name.suffix))

            with open(materials_filename,'r') as materials_file:
                materials = KratosMultiphysics.Parameters(materials_file.read())

            if data_comm.Rank() == 0:
                # Create and read an auxiliary materials file for each one of the fields (only on one rank)
                for i_material in materials["properties"]:
                    aux_materials = KratosMultiphysics.Parameters()
                    aux_materials.AddEmptyArray("properties")
                    aux_materials["properties"].Append(i_material)

                    aux_materials_filename = GetAuxMaterialsFileName(materials_filename, i_material["properties_id"].GetInt())
                    with open(aux_materials_filename,'w') as aux_materials_file:
                        aux_materials_file.write(aux_materials.WriteJsonString())

            data_comm.Barrier()

            # read the files on all ranks
            for i_material in materials["properties"]:
                aux_materials_filename = GetAuxMaterialsFileName(materials_filename, i_material["properties_id"].GetInt())
                aux_material_settings = KratosMultiphysics.Parameters("""{"Parameters": {"materials_filename": ""}} """)
                aux_material_settings["Parameters"]["materials_filename"].SetString(aux_materials_filename)
                KratosMultiphysics.ReadMaterialsUtility(aux_material_settings, self.model)

            data_comm.Barrier()

            materials_imported = True
        else:
            materials_imported = False

         # If the element uses nodal material properties, transfer them to the nodes
        if self.element_has_nodal_properties:
            self._SetNodalProperties()

        return materials_imported


    def _SetNodalProperties(self):
        # Keep it for now, this function might need more parameters for EHD
        # If the element uses nodal material properties, transfer them to the nodes
        if self.element_has_nodal_properties:
            # Get fluid 1 and 2 properties
            properties_1 = self.main_model_part.Properties[1]
            properties_2 = self.main_model_part.Properties[2]

            rho_1 = properties_1.GetValue(KratosMultiphysics.DENSITY)
            rho_2 = properties_2.GetValue(KratosMultiphysics.DENSITY)
            mu_1 = properties_1.GetValue(KratosMultiphysics.DYNAMIC_VISCOSITY)
            mu_2 = properties_2.GetValue(KratosMultiphysics.DYNAMIC_VISCOSITY)

            # Check fluid 1 and 2 properties
            if rho_1 <= 0.0:
                raise Exception("DENSITY set to {0} in Properties {1}, positive number expected.".format(rho_1, properties_1.Id))
            if rho_2 <= 0.0:
                raise Exception("DENSITY set to {0} in Properties {1}, positive number expected.".format(rho_2, properties_2.Id))
            if mu_1 <= 0.0:
                raise Exception("DYNAMIC_VISCOSITY set to {0} in Properties {1}, positive number expected.".format(mu_1, properties_1.Id))
            if mu_2 <= 0.0:
                raise Exception("DYNAMIC_VISCOSITY set to {0} in Properties {1}, positive number expected.".format(mu_2, properties_2.Id))

            # Transfer density and (dynamic) viscostity to the nodes
            for node in self.main_model_part.Nodes:
                if node.GetSolutionStepValue(self._levelset_variable) <= 0.0:
                    node.SetSolutionStepValue(KratosMultiphysics.DENSITY, rho_1)
                    node.SetSolutionStepValue(KratosMultiphysics.DYNAMIC_VISCOSITY, mu_1)
                else:
                    node.SetSolutionStepValue(KratosMultiphysics.DENSITY, rho_2)
                    node.SetSolutionStepValue(KratosMultiphysics.DYNAMIC_VISCOSITY, mu_2)

    
    # This routine is the duplicate of the same one defined in FluidSolver 
    def _ReplaceElementsAndConditions(self):
        ## Get number of nodes and domain size
        elem_num_nodes = self._GetElementNumNodes()
        cond_num_nodes = self._GetConditionNumNodes()
        domain_size = self.main_model_part.ProcessInfo[KratosMultiphysics.DOMAIN_SIZE]

        ## If there are no elements and/or conditions, default to triangles/tetra meshes to avoid breaking the ReplaceElementsAndConditionsProcess
        ## This only affects the input name (if there are no elements or conditions to replace, nothing is replaced).
        if elem_num_nodes == 0:
            elem_num_nodes = domain_size + 1
        if cond_num_nodes == 0:
            cond_num_nodes = domain_size

        ## Complete the element name
        if (self.element_name is not None):
            new_elem_name = self.element_name + str(int(domain_size)) + "D" + str(int(elem_num_nodes)) + "N"
        else:
            raise Exception("There is no element name. Define the self.element_name string variable in your derived solver.")

        ## Complete the condition name
        if (self.condition_name is not None):
            new_cond_name = self.condition_name + str(int(domain_size)) + "D" + str(int(cond_num_nodes)) + "N"
        else:
            raise Exception("There is no condition name. Define the self.condition_name string variable in your derived solver.")

        ## Set the element and condition names in the Json parameters
        #self.settings["element_replace_settings"] = KratosMultiphysics.Parameters("""{}""")
        self.settings.AddValue("element_replace_settings", KratosMultiphysics.Parameters("""{}"""))
        self.settings["element_replace_settings"].AddEmptyValue("element_name").SetString(new_elem_name)
        self.settings["element_replace_settings"].AddEmptyValue("condition_name").SetString(new_cond_name)

        ## Call the replace elements and conditions process
        KratosMultiphysics.ReplaceElementsAndConditionsProcess(self.main_model_part, self.settings["element_replace_settings"]).Execute()


    # This routine is the duplicate of the same one defined in FluidSolver
    def _SetAndFillBuffer(self):
        init_dt = self.settings["time_stepping"]["time_step"].GetDouble() # The automatic dt calculation is deactivated.
        auxiliary_solver_utilities.SetAndFillBuffer(self.main_model_part, self.min_buffer_size, init_dt)


    # This routine is the duplicate of the same one defined in FluidSolver
    def _ExecuteCheckAndPrepare(self):
        ## Check that the input read has the shape we like
        prepare_model_part_settings = KratosMultiphysics.Parameters("{}")
        prepare_model_part_settings.AddValue("volume_model_part_name",self.settings["volume_model_part_name"])
        prepare_model_part_settings.AddValue("skin_parts",self.settings["skin_parts"])
        prepare_model_part_settings.AddValue("assign_neighbour_elements_to_conditions",self.settings["assign_neighbour_elements_to_conditions"])

        # CheckAndPrepareModelProcess(self.main_model_part, prepare_model_part_settings).Execute()
        if prepare_model_part_settings["volume_model_part_name"].GetString() == "":
            raise Exception("Please define the \"volume_model_part_name\" (string) argument.")

        volume_model_part_name = prepare_model_part_settings["volume_model_part_name"].GetString()
        skin_name_list = prepare_model_part_settings["skin_parts"]

        if self.main_model_part.Name == volume_model_part_name:
            self.volume_model_part = self.main_model_part
        else:
            self.volume_model_part = self.main_model_part.GetSubModelPart(volume_model_part_name)

        skin_parts = []
        for i in range(skin_name_list.size()):
            skin_parts.append(self.main_model_part.GetSubModelPart(skin_name_list[i].GetString()))

        # Temporary name: "fluid_computational_model_part"
        if self.main_model_part.HasSubModelPart("fluid_computational_model_part"):
            fluid_computational_model_part = self.main_model_part.GetSubModelPart("fluid_computational_model_part")
        else:
            fluid_computational_model_part = self.main_model_part.CreateSubModelPart("fluid_computational_model_part")
            fluid_computational_model_part.ProcessInfo = self.main_model_part.ProcessInfo

            for node in self.volume_model_part.Nodes:
                fluid_computational_model_part.AddNode(node,0)
            for elem in self.volume_model_part.Elements:
                fluid_computational_model_part.AddElement(elem,0)

            list_of_ids = set()
            for part in skin_parts:
                for cond in part.Conditions:
                    list_of_ids.add(cond.Id)

            fluid_computational_model_part.AddConditions(list(list_of_ids))

        # Orientation of the elements: only for trangles and tetrahedrons (simplex elements)
        geometry = self.main_model_part.Elements.__iter__().__next__().GetGeometry()
        is_simplex = geometry.LocalSpaceDimension() + 1 == geometry.PointsNumber()
        if not is_simplex:
            msg = "Geoemetry is not simplex. Orientation check is only available"
            msg += " for simplex geometries and hence it will be skipped."
            KratosMultiphysics.Logger.PrintWarning(type(self).__name__, msg)
            return 0

        tmoc = KratosMultiphysics.TetrahedralMeshOrientationCheck
        throw_errors = False
        flags = (tmoc.COMPUTE_NODAL_NORMALS).AsFalse() | (tmoc.COMPUTE_CONDITION_NORMALS).AsFalse()
        # By default the neighboring elements are assigned
        flags |= tmoc.ASSIGN_NEIGHBOUR_ELEMENTS_TO_CONDITIONS

        KratosMultiphysics.TetrahedralMeshOrientationCheck(fluid_computational_model_part,throw_errors, flags).Execute()

    
    def _GetElementNumNodes(self):
        if self.main_model_part.NumberOfElements() != 0:
            element_num_nodes = len(self.main_model_part.Elements.__iter__().__next__().GetNodes())
        else:
            element_num_nodes = 0

        element_num_nodes = self.main_model_part.GetCommunicator().GetDataCommunicator().MaxAll(element_num_nodes)
        return element_num_nodes


    def _GetConditionNumNodes(self):
        if self.main_model_part.NumberOfConditions() != 0:
            condition_num_nodes = len(self.main_model_part.Conditions.__iter__().__next__().GetNodes())
        else:
            condition_num_nodes = 0

        condition_num_nodes = self.main_model_part.GetCommunicator().GetDataCommunicator().MaxAll(condition_num_nodes)
        return condition_num_nodes
    
    
    def _GetScheme(self):
        if not hasattr(self, '_scheme'):
            self._scheme = self._CreateScheme()
        return self._scheme
    
    def _CreateScheme(self):
        domain_size = self.GetComputingModelPart().ProcessInfo[KratosMultiphysics.DOMAIN_SIZE]
        # Here, the element incorporates the time integration scheme
        # It is required to perform the nodal update once the current time step is solved
        scheme = KratosMultiphysics.ResidualBasedIncrementalUpdateStaticSchemeSlip(
            domain_size,
            domain_size + 1)
        # Tthe BDF time discretization utility is required to update the BDF coefficients
        if (self.settings["time_scheme"].GetString() == "bdf2"):
            time_order = 2
            self.time_discretization = KratosMultiphysics.TimeDiscretization.BDF(time_order)
        else:
            err_msg = "Requested time integration scheme \"" + self.settings["time_scheme"].GetString()+ "\" is not available.\n"
            raise Exception(err_msg)
        return scheme


    def _GetConvergenceCriterion(self):
        if not hasattr(self, '_convergence_criterion'):
            self._convergence_criterion = self._CreateConvergenceCriterion()
        return self._convergence_criterion

    def _CreateConvergenceCriterion(self):
        convergence_criterion = KratosMultiphysics.MixedGenericCriteria(
                [(KratosMultiphysics.VELOCITY, self.settings["relative_velocity_tolerance"].GetDouble(), self.settings["absolute_velocity_tolerance"].GetDouble()),
                (KratosMultiphysics.PRESSURE, self.settings["relative_pressure_tolerance"].GetDouble(), self.settings["absolute_pressure_tolerance"].GetDouble())])
        convergence_criterion.SetEchoLevel(self.settings["echo_level"].GetInt())
        return convergence_criterion


    def _GetLinearSolver(self):
        if not hasattr(self, '_linear_solver'):
            self._linear_solver = self._CreateLinearSolver()
        return self._linear_solver

    def _CreateLinearSolver(self):
        linear_solver_configuration = self.settings["linear_solver_settings"]
        return linear_solver_factory.ConstructSolver(linear_solver_configuration)


    def _GetBuilderAndSolver(self):
        if not hasattr(self, '_builder_and_solver'):
            self._builder_and_solver = self._CreateBuilderAndSolver()
        return self._builder_and_solver

    def _CreateBuilderAndSolver(self):
        linear_solver = self._GetLinearSolver()
        if self.settings["consider_periodic_conditions"].GetBool():
            builder_and_solver = KratosCFD.ResidualBasedBlockBuilderAndSolverPeriodic(
                linear_solver,
                KratosCFD.PATCH_INDEX)
        else:
            builder_and_solver = KratosMultiphysics.ResidualBasedBlockBuilderAndSolver(linear_solver)
        return builder_and_solver


    def _GetSolutionStrategy(self):
        if not hasattr(self, '_solution_strategy'):
            self._solution_strategy = self._CreateSolutionStrategy()
        return self._solution_strategy

    def _CreateSolutionStrategy(self):
        # Only the nonlinear (Newton-Raphson) strategy is available.
        computing_model_part = self.GetComputingModelPart()
        time_scheme = self._GetScheme()
        convergence_criterion = self._GetConvergenceCriterion()
        builder_and_solver = self._GetBuilderAndSolver()
        return KratosMultiphysics.ResidualBasedNewtonRaphsonStrategy(
            computing_model_part,
            time_scheme,
            convergence_criterion,
            builder_and_solver,
            self.settings["maximum_iterations"].GetInt(),
            self.settings["compute_reactions"].GetBool(),
            self.settings["reform_dofs_at_each_step"].GetBool(),
            self.settings["move_mesh_flag"].GetBool())


    def _GetDistanceModificationProcess(self):
        if not hasattr(self, '_distance_modification_process'):
            self._distance_modification_process = self.__CreateDistanceModificationProcess()
        return self._distance_modification_process

    def __CreateDistanceModificationProcess(self):
        # Set suitable distance correction settings for free-surface problems
        # Note that the distance modification process is applied to the computing model part
        distance_modification_settings = self.settings["distance_modification_settings"]
        distance_modification_settings.ValidateAndAssignDefaults(self.GetDefaultParameters()["distance_modification_settings"])
        distance_modification_settings["model_part_name"].SetString(self.GetComputingModelPart().FullName())

        # Check user provided settings
        if not distance_modification_settings["continuous_distance"].GetBool():
            distance_modification_settings["continuous_distance"].SetBool(True)
            KratosMultiphysics.Logger.PrintWarning("Provided distance correction \'continuous_distance\' is \'False\'. Setting to \'True\'.")
        if not distance_modification_settings["check_at_each_time_step"].GetBool():
            distance_modification_settings["check_at_each_time_step"].SetBool(True)
            KratosMultiphysics.Logger.PrintWarning("Provided distance correction \'check_at_each_time_step\' is \'False\'. Setting to \'True\'.")
        if distance_modification_settings["avoid_almost_empty_elements"].GetBool():
            distance_modification_settings["avoid_almost_empty_elements"].SetBool(False)
            KratosMultiphysics.Logger.PrintWarning("Provided distance correction \'avoid_almost_empty_elements\' is \'True\'. Setting to \'False\' to avoid modifying the distance sign.")
        if distance_modification_settings["deactivate_full_negative_elements"].GetBool():
            distance_modification_settings["deactivate_full_negative_elements"].SetBool(False)
            KratosMultiphysics.Logger.PrintWarning("Provided distance correction \'deactivate_full_negative_elements\' is \'True\'. Setting to \'False\' to avoid deactivating the negative volume (e.g. water).")

        # Create and return the distance correction process
        return KratosCFD.DistanceModificationProcess(
            self.model,
            distance_modification_settings)


    def _GetAccelerationLimitationUtility(self):
        if not hasattr(self, '_acceleration_limitation_utility'):
            self._acceleration_limitation_utility = self.__CreateAccelerationLimitationUtility()
        return self._acceleration_limitation_utility
    
    def __CreateAccelerationLimitationUtility(self):
        maximum_multiple_of_g_acceleration_allowed = 5.0
        acceleration_limitation_utility = KratosCFD.AccelerationLimitationUtilities(
            self.GetComputingModelPart(),
            maximum_multiple_of_g_acceleration_allowed)

        return acceleration_limitation_utility


    def _PerformLevelSetConvection(self):
        # Solve the levelset convection problem
        self._GetLevelSetConvectionProcess().Execute()
    
    def _GetLevelSetConvectionProcess(self):
        if not hasattr(self, '_level_set_convection_process'):
            self._level_set_convection_process = self._CreateLevelSetConvectionProcess()
        return self._level_set_convection_process

    def _CreateLevelSetConvectionProcess(self):
        # Construct the level set convection process
        domain_size = self.main_model_part.ProcessInfo[KratosMultiphysics.DOMAIN_SIZE]
        computing_model_part = self.GetComputingModelPart()
        linear_solver = self._GetLevelsetLinearSolver()
        levelset_convection_settings = self.settings["levelset_convection_settings"]
        if domain_size == 2:
            level_set_convection_process = KratosMultiphysics.LevelSetConvectionProcess2D(
                computing_model_part,
                linear_solver,
                levelset_convection_settings)
        else:
            level_set_convection_process = KratosMultiphysics.LevelSetConvectionProcess3D(
                computing_model_part,
                linear_solver,
                levelset_convection_settings)

        return level_set_convection_process

    def _GetLevelsetLinearSolver(self):
        # A linear solver configured specifically for the level-set convection process
        if not hasattr(self, '_levelset_linear_solver'):
            self._levelset_linear_solver = self._CreateLinearSolver() # TODO: add customized configuration
        return self._levelset_linear_solver


    def _GetDistanceReinitializationProcess(self):
        if not hasattr(self, '_distance_reinitialization_process'):
            self._distance_reinitialization_process = self._CreateDistanceReinitializationProcess()
        return self._distance_reinitialization_process

    def _CreateDistanceReinitializationProcess(self):
        # Construct the variational distance calculation process
        if (self._reinitialization_type == "variational"):
            maximum_iterations = 2 #TODO: Make this user-definable
            linear_solver = self._GetRedistancingLinearSolver()
            computing_model_part = self.GetComputingModelPart()
            if self.main_model_part.ProcessInfo[KratosMultiphysics.DOMAIN_SIZE] == 2:
                distance_reinitialization_process = KratosMultiphysics.VariationalDistanceCalculationProcess2D(
                    computing_model_part,
                    linear_solver,
                    maximum_iterations,
                    KratosMultiphysics.VariationalDistanceCalculationProcess2D.CALCULATE_EXACT_DISTANCES_TO_PLANE)
            else:
                distance_reinitialization_process = KratosMultiphysics.VariationalDistanceCalculationProcess3D(
                    computing_model_part,
                    linear_solver,
                    maximum_iterations,
                    KratosMultiphysics.VariationalDistanceCalculationProcess3D.CALCULATE_EXACT_DISTANCES_TO_PLANE)

        elif (self._reinitialization_type == "parallel"):
            #TODO: move all this to solver settings

            # AW 3.6: old code, commented
            
            #layers = self.settings["parallel_redistance_max_layers"].GetInt()
            #parallel_distance_settings = KratosMultiphysics.Parameters("""{
            #    "max_levels" : 25,
            #    "max_distance" : 1.0,
            #    "calculate_exact_distances_to_plane" : true
            #}""")
            #parallel_distance_settings["max_levels"].SetInt(layers)


            # AW 3.6: new code
            layers = self.settings["parallel_redistance_max_layers"].GetInt()
            max_distance = self.settings["max_distance"].GetDouble()
            calculate_exact_distance_to_plane = self.settings["calculate_exact_distances_to_plane"].GetBool()
            preserve_interface = self.settings["preserve_interface"].GetBool()
            parallel_distance_settings = KratosMultiphysics.Parameters("""{
                "max_levels" : 25,
                "max_distance" : 1.0,
                "calculate_exact_distances_to_plane" : true,
                "preserve_interface" : false
            }""")
            parallel_distance_settings["max_levels"].SetInt(layers)
            parallel_distance_settings["max_distance"].SetDouble(max_distance)
            parallel_distance_settings["calculate_exact_distances_to_plane"].SetBool(calculate_exact_distance_to_plane)
            parallel_distance_settings["preserve_interface"].SetBool(preserve_interface)

            # Print after setting values
            print("DEBUG: Parallel Distance Settings that are active:")
            print(parallel_distance_settings)

            if self.main_model_part.ProcessInfo[KratosMultiphysics.DOMAIN_SIZE] == 2:
                distance_reinitialization_process = KratosMultiphysics.ParallelDistanceCalculationProcess2D(
                    self.main_model_part,
                    parallel_distance_settings)
            else:
                distance_reinitialization_process = KratosMultiphysics.ParallelDistanceCalculationProcess3D(
                    self.main_model_part,
                    parallel_distance_settings)
        elif (self._reinitialization_type == "none"):
                KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, "Redistancing is turned off.")
        else:
            raise Exception("Please use a valid distance reinitialization type or set it as \'none\'. Valid types are: \'variational\' and \'parallel\'.")

        return distance_reinitialization_process

    def _GetRedistancingLinearSolver(self):
        # A linear solver configured specifically for distance re-initialization process
        if not hasattr(self, '_redistancing_linear_solver'):
            self._redistancing_linear_solver = self._CreateLinearSolver() # TODO: add customized configuration
        return self._redistancing_linear_solver


    def _GetDistanceSmoothingProcess(self):
        if not hasattr(self, '_distance_smoothing_process'):
            self._distance_smoothing_process = self._CreateDistanceSmoothingProcess()
        return self._distance_smoothing_process

    def _CreateDistanceSmoothingProcess(self):
        # construct the distance smoothing process
        linear_solver = self._GetSmoothingLinearSolver()
        if self.main_model_part.ProcessInfo[KratosMultiphysics.DOMAIN_SIZE] == 2:
            distance_smoothing_process = KratosCFD.DistanceSmoothingProcess2D(
            self.main_model_part,
            linear_solver)
        else:
            distance_smoothing_process = KratosCFD.DistanceSmoothingProcess3D(
            self.main_model_part,
            linear_solver)

        return distance_smoothing_process

    def _GetSmoothingLinearSolver(self):
        # A linear solver configured specifically for the distance smoothing process
        if not hasattr(self, '_smoothing_linear_solver'):
            self._smoothing_linear_solver = self._CreateLinearSolver() # TODO: add customized configuration
        return self._smoothing_linear_solver


    def _GetDistanceGradientProcess(self):
        if not hasattr(self, '_distance_gradient_process'):
            self._distance_gradient_process = self._CreateDistanceGradientProcess()
        return self._distance_gradient_process

    def _CreateDistanceGradientProcess(self):
        distance_gradient_process = KratosMultiphysics.ComputeNodalGradientProcess(
                self.main_model_part,
                self._levelset_variable,
                self._levelset_gradient_variable,
                KratosMultiphysics.NODAL_AREA)

        return distance_gradient_process


    def _GetDistanceCurvatureProcess(self):
        if not hasattr(self, '_distance_curvature_process'):
            self._distance_curvature_process = self._CreateDistanceCurvatureProcess()
        return self._distance_curvature_process

    def _CreateDistanceCurvatureProcess(self):
        distance_curvature_process = KratosMultiphysics.ComputeNonHistoricalNodalNormalDivergenceProcess(
                self.main_model_part,
                self._levelset_gradient_variable,
                KratosCFD.CURVATURE,
                KratosMultiphysics.NODAL_AREA)

        return distance_curvature_process

    
    def _GetConsistentNodalPressureGradientProcess(self):
        if not hasattr(self, '_consistent_nodal_pressure_gradient_process'):
            self._consistent_nodal_pressure_gradient_process = self._CreateConsistentNodalPressureGradientProcess()
        return self._consistent_nodal_pressure_gradient_process
    
    def _CreateConsistentNodalPressureGradientProcess(self):
        consistent_nodal_pressure_gradient_process = KratosCFD.CalulateLevelsetConsistentNodalGradientProcess(
                self.main_model_part)

        return consistent_nodal_pressure_gradient_process
    

    
    def _GetContactAngleEvaluatorProcess(self):
        if not hasattr(self, '_distance_curvature_process'):
            self._distance_curvature_process = self._CreateContactAngleEvaluatorProcess()
        return self._distance_curvature_process

    def _CreateContactAngleEvaluatorProcess(self):
        contact_angle_settings = self.settings["contact_angle_settings"]
        contact_angle_settings.ValidateAndAssignDefaults(self.GetDefaultParameters()["contact_angle_settings"])
        contact_angle_evaluator = KratosDroplet.ContactAngleEvaluatorProcess(self.main_model_part, contact_angle_settings)

        return contact_angle_evaluator
    



    def FitPlaneAndExtrapolate(self, neighbor_nodes, variable):
        """
        Fit a linear plane (z = ax + by + c) using least squares and return the coefficients.

        :param neighbor_nodes: List of neighbor nodes.
        :param variable: The Kratos variable to be extrapolated.
        :return: (a, b, c) coefficients of the plane equation.
        """
        X, Y, Z = [], [], []

        for node in neighbor_nodes:
            X.append(node.X)
            Y.append(node.Y)
            Z.append(node.GetSolutionStepValue(variable))  # Get the value of the variable

        # Solve the least squares problem: [X Y 1] * [a b c] = Z
        A = np.vstack([X, Y, np.ones(len(X))]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)  # Solve for [a, b, c]

        return coeffs  # Returns (a, b, c)
    
    def ExtrapolateBoundaryValues(self, variable):
        """
        Extrapolate values for a given variable on boundary nodes using polynomial fitting.
    
        :param model_part: The Kratos ModelPart containing nodes and elements.
        :param variable: The Kratos variable to extrapolate (e.g., KM.DISTANCE, KM.TEMPERATURE).
        """
        for node in self.main_model_part.Nodes:
            if node.Is(KratosMultiphysics.BOUNDARY):  # Only process boundary nodes
                neighbor_nodes = set()

                # Find elements that contain this node
                for elem in self.main_model_part.Elements:
                    if node.Id in [n.Id for n in elem.GetNodes()]:  # Check if the node is in the element
                        for neighbor in elem.GetNodes():
                            if not neighbor.Is(KratosMultiphysics.BOUNDARY):  # Ensure we get non-boundary neighbors
                                neighbor_nodes.add(neighbor)

                # Ensure at least 3 neighbors for polynomial fitting
                if len(neighbor_nodes) >= 3:
                    a, b, c = self.FitPlaneAndExtrapolate(neighbor_nodes, variable)
                    estimated_value = a * node.X + b * node.Y + c
                    node.SetSolutionStepValue(variable, estimated_value)

    # AW 8.5 Merge
    #####################################################################################################################
    
    # Initialization of distance function distribution
    def _Initialize_Conservative_LevelSet_Function(self):

        find_nodal_h_max = KratosDroplet.FindNodalHProcessMax(self.main_model_part)
        find_nodal_h_max.Execute()

        # Find Maximum Value of Nodal_h inside the Model_Part   
        self.nodal_h_max = 0.0
        for node in self.main_model_part.Nodes:
            nodal_h = node.GetSolutionStepValue(KratosDroplet.NODAL_H_MAX)
            if (nodal_h > self.nodal_h_max):
                self.nodal_h_max = nodal_h
        print ("Maximum Value of Nodal_H = ",self.nodal_h_max)
        power = self.settings["convection_diffusion_settings"]["power_value_in_epsilon_Calculation"].GetDouble()
        denominator = self.settings["convection_diffusion_settings"]["denominator_value_in_epsilon_Calculation"].GetDouble()
        self.epsilon = self.nodal_h_max**(power)/denominator
        print ("Epsilon = ",self.epsilon)
      
        # Initialization of distance function
        for node in self.main_model_part.Nodes:
            distance = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE)
            conservative_distance = (1 / ( 1 + (math.e)**( -distance/self.epsilon) )) - 0.5
            node.SetSolutionStepValue(KratosMultiphysics.DISTANCE, conservative_distance)

    # Curvature correction
    def _Curvature_Correction(self):
        for node in self.main_model_part.Nodes:
            distance = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE)
            if (distance >= 0.49999 or distance <= -0.49999):
                node.SetValue(KratosCFD.CURVATURE, 0.0)

#####################################################################################################################

    def _PerformConservativeLaw(self):

        if (self.settings["domain_size"].GetInt() == 2):
            KratosDroplet.FindConservativeElementsProcess(self.main_model_part).Execute()

        self.convergenge_tolerence = self.settings["convection_diffusion_settings"]["convergenge_tolerence"].GetDouble()
        nodes1 = self.main_model_part.Nodes
        ndes1_num = len(nodes1)

        for node in nodes1:
            dist = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE, 0)
            if (dist > 0.499995):
                dist = 0.5
            elif (dist < -0.499995):
                dist = -0.5
            node.SetSolutionStepValue(KratosMultiphysics.DISTANCE, 0, dist)
        
        def CreateAnalysisStageWithFlushInstance(cls, global_model, parameters, epsilon, convergenge_tolerence):
            class AnalysisStageWithFlush(cls):

                def __init__(self, model,project_parameters, epsilon, convergenge_tolerence, flush_frequency=10.0):
                    super().__init__(model,project_parameters)
                    self.flush_frequency = flush_frequency
                    self.domain_size = project_parameters["solver_settings"]["domain_size"].GetInt()
                    if (self.domain_size == 3):
                        KratosMultiphysics.ModelPartIO(project_parameters["solver_settings"]["model_import_settings"]["input_filename"].GetString()).ReadModelPart(self._GetSolver().GetComputingModelPart()) 
                        project_parameters["solver_settings"]["model_import_settings"]["input_type"].SetString("use_input_model_part")
                    self.last_flush = time.time()
                    self.epsilon = epsilon
                    self.convergenge_tolerence = convergenge_tolerence
                    self.pseudo_steps_to_convergence = -1
                    sys.stdout.flush()

                def Initialize(self):
                    super().Initialize()
                    self.tolerence = 1.0
                    
                    nodes2 = self._GetSolver().GetComputingModelPart().Nodes
                    nodes2_num = len(nodes2)
                    print(ndes1_num, nodes2_num) 
                    
                    for node in nodes2:
                        node.SetSolutionStepValue(KratosMultiphysics.FLAG_VARIABLE, 1)
                        node.SetSolutionStepValue(KratosMultiphysics.DENSITY, 1.0)
                        node.SetSolutionStepValue(KratosMultiphysics.SPECIFIC_HEAT, 1.0)
                        node.SetSolutionStepValue(KratosMultiphysics.CONDUCTIVITY, self.epsilon)

                    # Synchronize nodes based on ID
                    nodes_mapping = {}
                    for node in nodes2:
                        node_id = node.Id
                        corresponding_node1 = nodes1[node_id]
                        nodes_mapping[node] = corresponding_node1 

                    for node in nodes2:
                        dist = nodes_mapping[node].GetSolutionStepValue(KratosMultiphysics.DISTANCE, 0)
                        dist += 0.5
                        node.SetSolutionStepValue(KratosMultiphysics.TEMPERATURE, 0, dist)
                    
                    for elem in self._GetSolver().GetComputingModelPart().Elements:
                        flag = 1
                        for node in elem.GetNodes():
                            temp = node.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE, 0)
                            if ( 0.00001 <= temp <= 0.99999 ):
                                flag = 0
                        if flag == 0:
                            for node in elem.GetNodes():
                                node.SetSolutionStepValue(KratosMultiphysics.FLAG_VARIABLE, 0)
                    
                    sys.stdout.flush()


                def InitializeSolutionStep(self): 
                    super().InitializeSolutionStep()
                    nodes2 = self._GetSolver().GetComputingModelPart().Nodes

                    step = self._GetSolver().GetComputingModelPart().ProcessInfo[KratosMultiphysics.STEP]
                    if (step == 1):
                        for node in nodes2:
                            normalx = node.GetSolutionStepValue(KratosMultiphysics.NORMAL_X, 0)
                            normaly = node.GetSolutionStepValue(KratosMultiphysics.NORMAL_Y, 0)
                            if (self.domain_size == 3):
                                normalz = node.GetSolutionStepValue(KratosMultiphysics.NORMAL_Z, 0)
                            flag = node.GetSolutionStepValue(KratosMultiphysics.FLAG_VARIABLE)
                            if (flag > 0.5):
                                normalx = normaly = normalz = 0.0
                            node.SetSolutionStepValue(KratosMultiphysics.NORMAL_X, 0, normalx)
                            node.SetSolutionStepValue(KratosMultiphysics.NORMAL_Y, 0, normaly)
                            if (self.domain_size == 3):
                                node.SetSolutionStepValue(KratosMultiphysics.NORMAL_Z, 0, normalz)

                    for node in nodes2:
                        gradx = node.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE_GRADIENT_X, 0)
                        grady = node.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE_GRADIENT_Y, 0)
                        node.SetSolutionStepValue(KratosMultiphysics.TEMPERATURE_GRADIENT_X, 1, gradx)
                        node.SetSolutionStepValue(KratosMultiphysics.TEMPERATURE_GRADIENT_Y, 1, grady)
                        if (self.domain_size == 3):
                            gradz = node.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE_GRADIENT_Z, 0)
                            node.SetSolutionStepValue(KratosMultiphysics.TEMPERATURE_GRADIENT_Z, 1, gradz)

                def FinalizeSolutionStep(self):
                    super().FinalizeSolutionStep()

                    if (self.domain_size == 3):
                        local_gradient_TEMPERATURE = KratosMultiphysics.ComputeNodalGradientProcess(self._GetSolver().GetComputingModelPart(), KratosMultiphysics.TEMPERATURE, KratosMultiphysics.TEMPERATURE_GRADIENT)
                    else:
                        local_gradient_TEMPERATURE = KratosMultiphysics.ComputeNodalGradientProcess2D(self._GetSolver().GetComputingModelPart(), KratosMultiphysics.TEMPERATURE, KratosMultiphysics.TEMPERATURE_GRADIENT)
                    local_gradient_TEMPERATURE.Execute()

                    nodes2 = self._GetSolver().GetComputingModelPart().Nodes

                    max_norm_of_temp_grad = 0.0
                    max_norm_of_temp_grad_prev = 0.0
                    max_diff = 0.0
                    for node in nodes2:
                        gradx = node.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE_GRADIENT_X, 1)
                        grady = node.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE_GRADIENT_Y, 1)
                        gradz = 0.0
                        if (self.domain_size == 3):
                            gradz = node.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE_GRADIENT_Z, 1)
                        norm_of_temp_grad_prev = math.sqrt(gradx**2 + grady**2 + gradz**2)
                        if ( norm_of_temp_grad_prev > max_norm_of_temp_grad_prev ):
                           max_norm_of_temp_grad_prev = norm_of_temp_grad_prev
                        gradx = node.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE_GRADIENT_X, 0)
                        grady = node.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE_GRADIENT_Y, 0)
                        if (self.domain_size == 3):
                            gradz = node.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE_GRADIENT_Z, 0)
                        norm_of_temp_grad = math.sqrt(gradx**2 + grady**2 + gradz**2)
                        if ( abs(norm_of_temp_grad - norm_of_temp_grad_prev) > max_diff ):
                           max_diff = abs(norm_of_temp_grad - norm_of_temp_grad_prev)
                        if ( norm_of_temp_grad > max_norm_of_temp_grad ):
                           max_norm_of_temp_grad = norm_of_temp_grad
                    self.tolerence = max_diff/max_norm_of_temp_grad_prev
                    print("max_norm_of_temp_grad = ", max_norm_of_temp_grad)
                    print("tolerence = ", self.tolerence)
                    
                    # Synchronize nodes based on ID
                    nodes_mapping = {}
                    for node in nodes2:
                        node_id = node.Id
                        corresponding_node1 = nodes1[node_id]
                        nodes_mapping[node] = corresponding_node1
                    
                    for node in nodes2:
                        temp = node.GetSolutionStepValue(KratosMultiphysics.TEMPERATURE, 0)
                        temp -= 0.5
                        nodes_mapping[node].SetSolutionStepValue(KratosMultiphysics.DISTANCE, 0, temp)
                    
                    if self.parallel_type == "OpenMP":
                        now = time.time()
                        if now - self.last_flush > self.flush_frequency:
                            sys.stdout.flush()
                            self.last_flush = now

                def KeepAdvancingSolutionLoop(self):
                    super().KeepAdvancingSolutionLoop()
                    self.pseudo_steps_to_convergence += 1
                    if (self.time < self.end_time):
                        return  self.tolerence > self.convergenge_tolerence
                    else:
                        step = self._GetSolver().GetComputingModelPart().ProcessInfo[KratosMultiphysics.STEP]
                        KratosMultiphysics.Logger.PrintWarning("Convergence was not achieved after", step,"pseudo time steps.")
                        return self.time < self.end_time

            return AnalysisStageWithFlush(global_model, parameters, epsilon, convergenge_tolerence)

        parameters = KratosMultiphysics.Parameters("""{ "problem_data": {     
        "start_time": 0.0
        },    
        "solver_settings": {
            "solver_type": "transient",
            "analysis_type": "non_linear",
            "model_part_name": "ThermalModelPart",
            "model_import_settings": {
                "input_type": "mdpa",
                "input_filename": "ConsModelPart"
            },
            "element_replace_settings": {
                "element_name": "ConservativeLevelsetElement",
                "condition_name": "ThermalFace"
            },
            "compute_reactions": false,
            "convergence_criterion": "residual_criterion",
            "solution_relative_tolerance": 1e-05,
            "solution_absolute_tolerance": 1e-07,
            "residual_relative_tolerance": 1e-05,
            "residual_absolute_tolerance": 1e-07,
            "time_stepping": {
            },
            "auxiliary_variables_list": [
                "FLAG_VARIABLE"
            ]
        },
        "output_processes": {
            "gid_output": [
                {
                    "python_module": "gid_output_process",
                    "kratos_module": "KratosMultiphysics",
                    "process_name": "GiDOutputProcess",
                "Parameters": { }
                }
            ]
        }
        }""")

        """pre-pending the absolut path of the files in the Project Parameters"""
        output_name = f'gid_output/Conv_{self.pseudo_time}'
        output_params = KratosMultiphysics.Parameters("""{
            "model_part_name": "ThermalModelPart",
            "postprocess_parameters": {
                "result_file_configuration": {
                    "gidpost_flags": {
                        "GiDPostMode": "GiD_PostBinary",
                        "WriteDeformedMeshFlag": "WriteDeformed",
                        "WriteConditionsFlag": "WriteConditions",
                        "MultiFileFlag": "SingleFile"
                    },
                    "file_label": "time",
                    "output_control_type": "time",
                    "output_interval": 1e-05,
                    "body_output": true,
                    "node_output": false,
                    "skin_output": false,
                    "plane_output": [],
                    "nodal_results": [
                        "TEMPERATURE",
                        "NORMAL",
                        "TEMPERATURE_GRADIENT",
                        "FLAG_VARIABLE"
                    ],
                    "gauss_point_results": [],
                    "nodal_nonhistorical_results": []
                },
                "point_data_configuration": []
            },
            "output_name": ""
        }""")
        output_params["output_name"].SetString(output_name)
        parameters["output_processes"]["gid_output"][0]["Parameters"] = output_params       
        self.pseudo_time +=1

        if (self.settings["domain_size"].GetInt() == 3):
            parameters["solver_settings"]["model_import_settings"]["input_filename"].SetString(self.settings["model_import_settings"]["input_filename"].GetString())

        parameters["problem_data"].AddEmptyValue("echo_level").SetInt(self.settings["convection_diffusion_settings"]["echo_level"].GetInt())
        parameters["problem_data"].AddEmptyValue("parallel_type").SetString(self.settings["convection_diffusion_settings"]["parallel_type"].GetString())
        parameters["solver_settings"].AddEmptyValue("domain_size").SetInt(self.settings["domain_size"].GetInt())
        parameters["solver_settings"]["time_stepping"].AddEmptyValue("time_step").SetDouble(self.settings["convection_diffusion_settings"]["time_step"].GetDouble())
        parameters["problem_data"].AddEmptyValue("end_time").SetDouble((self.settings["convection_diffusion_settings"]["max_substeps"].GetInt())*(self.settings["convection_diffusion_settings"]["time_step"].GetDouble()))
        
        analysis_stage_module_name = "KratosMultiphysics.ConvectionDiffusionApplication.convection_diffusion_analysis"
        analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
        analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

        analysis_stage_module = importlib.import_module(analysis_stage_module_name)
        analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

        global_model = KratosMultiphysics.Model()
        simulation = CreateAnalysisStageWithFlushInstance(analysis_stage_class, global_model, parameters, self.epsilon, self.convergenge_tolerence)
        simulation.Run()
        
        # Write the number of pseudo time steps to a text file
        with open('pseudo_steps_to_convergence.txt', 'a') as file:
             file.write(str(simulation.pseudo_steps_to_convergence) + '\n')


