# AW 8.5 Merge
import math
import sys
import time
import importlib
import numpy as np
# Importing the Kratos Library
import KratosMultiphysics
from KratosMultiphysics import auxiliary_solver_utilities
from KratosMultiphysics.python_solver import PythonSolver
import KratosMultiphysics.python_linear_solver_factory as linear_solver_factory
# AW 10.4: necessary for csv file writing
import os
import pdb
import csv
# AW 26.4: Nurbs fitting
import pandas as pd
import matplotlib.pyplot as plt                        # plotting
from geomdl import fitting, NURBS                      # NURBS fitting + curve class
from geomdl.visualization import VisMPL as vis
from collections import defaultdict, deque
# AW 5.5: needed for closest point minimization
from scipy.optimize import minimize_scalar
from matplotlib import cm

# Import applications
import KratosMultiphysics.FluidDynamicsApplication as KratosCFD
import KratosMultiphysics.DropletDynamicsApplication as KratosDroplet
#import KratosMultiphysics.ConvectionDiffusionApplication as KratosConv

# Import base class file
#from KratosMultiphysics.ConvectionDiffusionApplication.convection_diffusion_solver import ConvectionDiffusionSolver
#from KratosMultiphysics.FluidDynamicsApplication.fluid_solver import FluidSolver
#from KratosMultiphysics.FluidDynamicsApplication.navier_stokes_two_fluids_solver import NavierStokesTwoFluidsSolver

from pathlib import Path
# AW 8.5 Merge
import json

def CreateSolver(model, custom_settings):
    return DropletDynamicsSolver(model, custom_settings)


class DropletDynamicsSolver(PythonSolver):  # Before, it was derived from NavierStokesTwoFluidsSolver

    @classmethod
    def GetDefaultParameters(cls):
        ##settings string in json format
        # AW 21.4: quasistatic contact angle settings added
        # AW 5.5: smoothing coefficient increased to 500
        # AW 19.5: fitting settings added
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
            "distance_reinitialization": "variational",
            "parallel_redistance_max_layers" : 25,
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
                "Penalty_coefficient" : 100
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
            "normal_evaluation_mode": 1           
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

        # Create headers (overwrite on init)yy
        with open(self._curvature_csv_unfitted, "w") as f:
            f.write("Time,Element_ID,GaussPoint,UnfittedCurvature\n")

        with open(self._curvature_csv_fitted, "w") as f:
            f.write("Time,Element_ID,FittedCurvature\n")

        with open(self._normals_csv_unfitted, "w") as f:
            f.write("Time,Element_ID,GaussPoint,Nx,Ny,Nz\n")

        with open(self._normals_csv_fitted, "w") as f:
            f.write("Time,Element_ID,Nx,Ny,Nz\n")

        # AW 21.4: Added user-defined quasistationary contact line settings
        qscl_settings = self.settings["QuasiStatic_ContactAngle_Settings"]
        QuasiStatic_ContactAngle = qscl_settings["QuasiStatic_ContactAngle"].GetBool()
        Theta_equilibrium_hydrophilic = qscl_settings["Theta_equilibrium_hydrophilic"].GetDouble()
        Theta_equilibrium_hydrophobic = qscl_settings["Theta_equilibrium_hydrophobic"].GetDouble()
        Penalty_coefficient = qscl_settings["Penalty_coefficient"].GetDouble()
        self.main_model_part.ProcessInfo.SetValue(KratosDroplet.quasi_static_contact_angle, QuasiStatic_ContactAngle)
        self.main_model_part.ProcessInfo.SetValue(KratosDroplet.theta_equilibrium_hydrophilic, Theta_equilibrium_hydrophilic)
        self.main_model_part.ProcessInfo.SetValue(KratosDroplet.theta_equilibrium_hydrophobic, Theta_equilibrium_hydrophobic)
        self.main_model_part.ProcessInfo.SetValue(KratosDroplet.penalty_coefficient, Penalty_coefficient)

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

        # Momentum correction is on by default!
        KratosMultiphysics.VariableUtils().SetNonHistoricalVariable(KratosCFD.DISTANCE_CORRECTION, 0.0, self.main_model_part.Nodes)

        # Recompute the BDF2 coefficients
        (self.time_discretization).ComputeAndSaveBDFCoefficients(self.GetComputingModelPart().ProcessInfo)

        # Perform the level-set convection according to the previous step velocity
        self._PerformLevelSetConvection()

        KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, "Level-set convection is performed.")

        # AW 21.5: moved all of these lines before the intersection points utility is called

        # filtering noises is necessary for curvature calculation
        # distance gradient is used as a boundary condition for smoothing process
        self._GetDistanceGradientProcess().Execute()

        #AW 21.5: made smoothing conditional again
        if self._distance_smoothing:
            self._GetDistanceSmoothingProcess().Execute()
            KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, "Smoothing process is finished.")

            # distance gradient is called again to comply with the smoothed/modified DISTANCE
            self._GetDistanceGradientProcess().Execute()

        ########### AW 21.5: Function definitions ###################
        # AW-C 26.4: Defines a function to order unordered interface points into contours
        def order_interface_contour(cut_df, tol=1e-12):
            """
            Given a DataFrame of cut points with columns ['Element_ID','Point_ID','X','Y','Z'],
            returns a list of ordered (X,Y,Z) points describing the interface contour(s).
            """
            # 1) Represent each point by a tuple rounded to tolerance
            # AW-C 26.4: Helper to make a "key" from a point by rounding each coordinate to avoid floating point errors
            def key(pt):
                return (round(pt[0]/tol)*tol, round(pt[1]/tol)*tol, round(pt[2]/tol)*tol)
            
            # Build element → points and point_key → elements
            # AW-C 26.4: Maps each element ID to the list of points it owns
            elem_points = defaultdict(list)
            # AW-C 26.4: Maps each rounded point (key) to all element IDs it belongs to
            point_elems = defaultdict(list)
            
            # AW-C 26.4: Loops through all rows of the input DataFrame, filling both mappings above
            for _, row in cut_df.iterrows():
                e = row['Element_ID']
                pt = (row['X'], row['Y'], row['Z'])
                k = key(pt)
                elem_points[e].append(pt)
                point_elems[k].append(e)
            
            # 2) Find boundary points: keys present in only one element
            boundary_keys = [k for k, elems in point_elems.items() if len(elems)==1]
            
            # 3) Helper to walk one contour starting from start_key
            # AW-C 26.4: starting from a key, follows the chain of connected points through the mesh, building up one ordered contour, and stops at a boundary or if it returns to the start (closed loop)
            def walk(start_key):
                contour = []
                current_key = start_key
                prev_elem = None
                
                while True:
                    contour.append(current_key)
                    elems = point_elems[current_key]
                    # choose next element that isn't the one we came from
                    next_elem = elems[0] if elems[0]!=prev_elem else (elems[1] if len(elems)>1 else None)
                    if next_elem is None:
                        break  # reached boundary
                    # find the other point of that element
                    pts = elem_points[next_elem]
                    other_pt = pts[0] if key(pts[0])!=current_key else pts[1]
                    other_key = key(other_pt)
                    
                    prev_elem = next_elem
                    if other_key == start_key:
                        break  # closed loop
                    current_key = other_key
                
                # convert keys back to raw coords
                return [ (kx, ky) for (kx, ky, kz) in contour ]#, kz
        
            # 4) Collect all contours
            # AW-C 26.4: Walks through all boundaries first (open curves), then through all remaining points (closed loops) to collect all contours
            contours = []
            visited = set()
            # First walk from each boundary (open curve)
            for b in boundary_keys:
                if b in visited: continue
                c = walk(b)
                contours.append(c)
                visited.update(c)
            # Then any closed loops
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
        
         # it is needed to store level-set consistent nodal PRESSURE_GRADIENT for stabilization purpose
        self._GetConsistentNodalPressureGradientProcess().Execute()

        # TODO: Performing mass conservation check and correction process

        # Perform distance correction to prevent ill-conditioned cuts
        self._GetDistanceModificationProcess().ExecuteInitializeSolutionStep()



        # AW 15.5: everything related to inters points, fitting, normal averaging AFTER the distance modification process as of now!!!
        # AW 19.5: Added user-defined fitting settings

        # AW 19.5: Reload fitting settings from ProcessInfo
        fitting_type = self.main_model_part.ProcessInfo[KratosDroplet.FittingType]
        normal_evaluation_mode = self.main_model_part.ProcessInfo[KratosDroplet.NormalEvaluationMode]

        # AW 19.5: Only run Intersection Points utility if fitting_type is explicitly set to "nurbs" (needed for nurbs fitting) or normal_evaluation_mode==3 (needed for normal averaging)
        # AW 22.5: also add this boolean, as we need intersection points also for curvature smoothing
        do_curvature_smoothing = False
        if fitting_type == "nurbs" or normal_evaluation_mode == 2 or normal_evaluation_mode == 3 or do_curvature_smoothing:
            # Debug print: delete once it works
            print(f"IntersectionPointsUtility is executed because fitting_type = '{fitting_type}' or normal_evaluation_mode = {normal_evaluation_mode}.")
  
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

        # AW 19.5: only run nurbs fitting block if fitting method was explicitly set to nurbs
        # AW 27.5: adapted such that it is also run for normal evaluation mode 2
        if fitting_type == "nurbs" or normal_evaluation_mode == 2:
            # Debug statement: delete once it works
            print("NURBS fitting is executed because fitting_type is set to 'nurbs'.")


            ##################### Beginning of Nurbs Fitting ########################
            # AW 27.4: outcomment legacy (non Nurbs based) curve fitting + curvature/normal computation approaches
            """ # Fit curves to the points on an element-by-element basis
            KratosDroplet.IntersectionPointsUtility.ProcessIntersectionPointsAndFitCurves("element_curves.txt")
            KratosDroplet.IntersectionPointsUtility.ProcessIntersectionPointsAndFitCurvesparabola("element_curves_parabola.txt")

            # AW 9.4: Ensure curvature file is cleared before writing
            open("element_curvatures_simplified.csv", "w").close()

            KratosDroplet.CurvatureFittingUtility.ComputeFittedCurvatures(
                "element_curves_parabola.txt",
                "element_curves.txt",
                "intersection_points.txt",
                # AW 15.4: new input files
                "element_points_original.txt",
                "element_points_rotated.txt",
                "element_curvatures_simplified.csv"
            )

            KratosDroplet.CurvatureFittingUtility.LoadCurvatureCSV("element_curvatures_simplified.csv")

            # AW 10.4: Ensure averaged normals file is cleared before writing
            open("averaged_normals.csv", "w").close()

            # AW 10.4: Compute fitted normals and write to file
            from KratosMultiphysics.DropletDynamicsApplication import NormalComputationUtility
            NormalComputationUtility.ComputeAveragedNormals(
                "element_curves_parabola.txt",
                "intersection_points.txt",
                "element_points_rotated.txt",
                "averaged_normals.csv"
            )
            KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, "Averaged normals computed and saved.")

            # AW 11.4: Load fitted normals once per time step
            NormalComputationUtility.LoadNormalCSV("averaged_normals.csv")
            KratosMultiphysics.Logger.PrintInfo(self.__class__.__name__, "Averaged normals loaded.") """
            


            # AW_C 26.4: Reads the file intersection_points.txt as a table, using tab characters as separators.
            data = pd.read_csv('intersection_points.txt', sep="\t")
            # AW-C 26.4: would sort data by the X column
            # data.sort_values(by='X', inplace=True)
            # would print basic statistics about the data
            # data.describe()
            # would extract the X and Y columns as a list of coordinate pairs
            # pts = (data[['X', 'Y']].values).tolist()
            # would remove duplicate points from that list
            # cut_points = np.unique(pts, axis=0).tolist()
            # AW-C 26.4: function that takes raw intersection points and orders them into continuous interface lines (contours)
            contours = order_interface_contour(data)
            # AW-C 26.4: picks the first contour (in case there are several disconnected ones); assumes one interface currently
            cut_points = contours[0]
            # AW-C 26.4: now a list of [x, y] coordinates tracing the interface, in order
            # print(cut_points)


            # ─────────────────────────────────────────────────────────────────────────────
            # 2.2. Global NURBS interpolation
            # AW-C 26.4:means we fit a quadratic NURBS curve (parabolic segments) 
            degree = 2  # cubic NURBS
            # AW-C 26.4: sets the number of control points for the NURBS curve (this controls the flexibility of the fit)
            # AW-I 26.4: we could use some optimization to check what number of control points gives best fit (could be repeated e.g. every 50 time steps, similar to whats done in parallel redistancing)
            ctrlpts_size=27# len(data)//2
            # try:
            # AW-C 26.4: Attempts to do an exact interpolating NURBS fit (with fallback on error)  
            # curve = fitting.interpolate_curve(cut_points, degree=degree) #, centripetal=True global fit 
            # except ZeroDivisionError:
            #     # fallback to approximation
            #     LeastSquares fit
            # AW-C 26.4: actual code uses an approximate fit (least squares) with the given control point number and degree, which is more robust for noisy or incomplete data
            print("Entering into curve fitting.")
            print("Entering into curve fitting.")
            # AW 13.5: updated to use less control points in case least squares fitting values due to ill-conditioning
            success = False
            max_attempts = ctrlpts_size - 5  # number of fallback attempts allowed
            for offset in range(max_attempts + 1):  # try with ctrlpts_size, ctrlpts_size-1, ...
                try:
                    current_ctrlpts_size = ctrlpts_size - offset
                    if current_ctrlpts_size < 6:
                        print("[ERROR] Cannot fit curve: control point count dropped below 6.")
                        break
                    print(f"Trying with ctrlpts_size = {current_ctrlpts_size}")
                    curve = fitting.approximate_curve(cut_points, degree=degree, ctrlpts_size=current_ctrlpts_size)
                    print("Curve fitting done.")
                    success = True
                    break
                except ZeroDivisionError:
                    print(f"[WARNING] Curve fitting failed with ctrlpts_size = {current_ctrlpts_size}, trying fewer points...")

            if not success:
                raise RuntimeError("Curve fitting failed for all attempted control point sizes.")

            print("Curve fitting done.")

            # AW-C 26.4: curve.delta sets the sampling resolution when evaluating the curve (smaller = more points, higher detail).
            curve.delta = 0.0001  # evaluation resolution
            # AW-C 26.4: curve.vis attaches a 2D visualization object for plotting
            # curve.vis = vis.VisCurve2D()
            # Refine knot vector
            # AW-C 26.4: (Commented out): One could refine the NURBS knot vector for better control point influence, but it’s unused here
            # operations.refine_knotvector(curve, [1])
            # AW-C 26.4. plot the curve
            # fig = curve.render()

            # Convert contour to NumPy array
            cut_points_np = np.array(cut_points)

            """ # AW 2.5: this part added for debug visualization
            # Ensure curve is evaluated at high resolution
            curve.evaluate()

            # Fitted NURBS curve points
            fitted_points = np.array(curve.evalpts)  # evaluated using delta = 0.01

            # Control points
            control_points = np.array(curve.ctrlpts)  # shape (ctrlpts_size, 2)

            # Plot all in one figure
            plt.figure(figsize=(6, 6))
            plt.axis("equal")

            # 2. Fitted NURBS curve
            plt.plot(fitted_points[:, 0], fitted_points[:, 1], 'r-', linewidth=2, label='Fitted NURBS Curve')

            # 1. Original ordered cut points (interface contour)
            plt.plot(cut_points_np[:, 0], cut_points_np[:, 1], 'bo-', markersize=3, label='Ordered Cut Points')

            # 3. Control points
            plt.plot(control_points[:, 0], control_points[:, 1], 'ko--', label='Control Points', markersize=4)

            # Labels and legend
            plt.title("NURBS Fitting of Interface Contour")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            # Save and show
            plt.savefig("interface_fitting_nurbs_debug.png", dpi=300)
            # plt.show() """


            # AW 5.5: outcommented the part where curvature+normal where stored nodally as a rough approximation
            """   # sample densely along the curve for closest-point search
            # AW 29.4: outcommented, as of now, no more dense sampling and closest point search (expensive and inaccurate)
            u_start, u_end = curve.knotvector[degree], curve.knotvector[-degree-1]
            print("Starting sampling: ")
            us_dense = np.linspace(u_start, u_end, 1000)
            print("Evaluating pointwise curvature: ")
            curve_points_dense = np.array([curve.evaluate_single(u)[:2] for u in us_dense]) 
            # Compute chord-length-based parametric values for the original points
            #  selects the first interface line; cut_points is now a list of 2D coordinates 
            cut_points = contours[0]
            # Converts the list to a NumPy array of shape (N, 2), where N is the number of points
            cut_points_np = np.array(cut_points)
            # If cut_points_np = [[x0,y0], [x1,y1], [x2,y2]], then:
            # diffs=[[x1−x0,y1−y0],[x2−x1,y2−y1],...]
            # Shape becomes (N-1, 2)
            diffs = np.diff(cut_points_np, axis=0)
            # Computes Euclidean distance between each consecutive pair
            # gives the segment length between every two neighboring points
            dists = np.linalg.norm(diffs, axis=1)
            # Computes the cumulative sum of the segment lengths
            # E.g., [d0, d0+d1, d0+d1+d2, ...]
            # np.insert(..., 0, 0.0) prepends a zero at the start, meaning the first point has zero arc length
            arc_lengths = np.insert(np.cumsum(dists), 0, 0.0)
            # total_length is the full length of the interface curve.
            total_length = arc_lengths[-1]
            # param_values is the normalized arc-length parameter for each point
            # ui​=total arc length up to i​∈[0,1]
            # used as parameter values for evaluating the NURBS curve at the original points
            param_values = arc_lengths / total_length  # normalized u values in [0, 1]


            # 4. Prepare output lists
            out_element_id = []
            out_point_id = []
            out_x = []
            out_y = []
            out_u = []
            out_curv = []
            out_nx = []
            out_ny = []
            out_theta = []

            # AW 29.4: adapted, now looping directly through the ordered cut points
            # Build lookup to get Element_ID and Point_ID by matching (X, Y) from DataFrame (intersection_points.txt)
            meta_lookup = {
                (round(row['X'], 12), round(row['Y'], 12)): (row['Element_ID'], row['Point_ID'])
                for _, row in data.iterrows()
            }
            # output dictionary looking something like this:
            #{
            #(0.001234567890, 0.002345678901): (42, 0),
            # ...
            #}

            # Loops over each point along the ordered interface contour (previously produced by order_interface_contour())
            # i is the index, used to retrieve the corresponding parameter value from param_values
            for i, (x0, y0) in enumerate(cut_points):
                # Grabs the normalized arc length (∈ [0, 1]) corresponding to this interface point
                # used to evaluate the NURBS curve at the right location
                u_closest = param_values[i]

                # Evaluate NURBS derivatives
                d0, d1, d2 = curve.derivatives(u_closest, order=2)
                x1, y1 = d1[0], d1[1]
                x2, y2 = d2[0], d2[1]
                num = abs(x1*y2 - y1*x2)
                den = (x1**2 + y1**2)**1.5
                curvature = num / den if den != 0 else 0.0
                tnorm = np.hypot(x1, y1)
                nx, ny = y1 / tnorm, -x1 / tnorm if tnorm != 0 else (0.0, 0.0)
                normal = (-nx, -ny)
                theta = np.rad2deg(np.arctan2(normal[1], normal[0]))

                # Recover element and point ID from lookup
                # Reconstructs the lookup key to find the associated Element_ID and Point_ID
                # Try exact match first
                # AW 30.4: adapted to include nearest neighbour search for points where exact coordinates wasnt found
                key = (round(x0, 12), round(y0, 12))
                if key in meta_lookup:
                    elem_id, pt_id = meta_lookup[key]
                else:
                    # Fallback to nearest match by Euclidean distance
                    coords_array = np.array(list(meta_lookup.keys()))
                    distances = np.linalg.norm(coords_array - np.array([x0, y0]), axis=1)
                    min_idx = np.argmin(distances)
                    if distances[min_idx] < 1e-10:  # choose a suitable tolerance
                        elem_id, pt_id = list(meta_lookup.values())[min_idx]
                    else:
                        raise ValueError(f"Could not find metadata for point ({x0}, {y0}) — even with fallback search.")

                # Store everything
                out_element_id.append(elem_id)
                out_point_id.append(pt_id)
                out_x.append(x0)
                out_y.append(y0)
                out_u.append(u_closest)
                out_curv.append(curvature)
                out_nx.append(normal[0])
                out_ny.append(normal[1])
                out_theta.append(theta)

            # 5. Loop over the DataFrame rows
            for _, row in data.iterrows():
                # AW 29.4: outcommented, as of now, no more dense sampling and closest point search (expensive and inaccurate)
                x0, y0 = row['X'], row['Y']
                dists = np.linalg.norm(curve_points_dense - np.array([x0, y0]), axis=1)
                idx_closest = np.argmin(dists)
                u_closest = us_dense[idx_closest] 


                # Calculate NURBS properties at this location
                d0, d1, d2 = curve.derivatives(u_closest, order=2)
                x1, y1 = d1[0], d1[1]
                x2, y2 = d2[0], d2[1]
                num = abs(x1*y2 - y1*x2)
                den = (x1**2 + y1**2)**1.5
                curvature = num / den if den != 0 else 0.0
                tnorm = np.hypot(x1, y1)
                nx, ny = y1 / tnorm, -x1 / tnorm if tnorm != 0 else (0.0, 0.0)
                normal = (-nx, -ny)
                theta = np.rad2deg(np.arctan2(normal[1], normal[0]))

                # Store original identifiers and results
                out_element_id.append(row['Element_ID'])
                out_point_id.append(row['Point_ID'])
                out_x.append(x0)
                out_y.append(y0)
                out_u.append(u_closest)
                out_curv.append(curvature)
                out_nx.append(normal[0])
                out_ny.append(normal[1])
                out_theta.append(theta) 

            # 6. Build the output DataFrame
            output_df = pd.DataFrame({
                "Element_ID": out_element_id,
                "Point_ID": out_point_id,
                "X": out_x,
                "Y": out_y,
                "u": out_u,
                "curvature": out_curv,
                "normal_x": out_nx,
                "normal_y": out_ny,
                "theta_deg": out_theta
            })

            output_df.to_csv("curve_normals_curvature_at_intersection_points.csv", index=False)
            print("Saved curvature and normals at intersection points to curve_normals_curvature_at_intersection_points.csv") """



            ###### AW 2.5: added this part to compute and store curvature at Gauss Points elementally
            # ─────────────────────────────────────────────────────────────────────────────


            # Load intersection points from file (already done above as `data`)
            # Group intersection points per element
            # creates a defaultdict from the collections module
            # If you try to access a key that doesn't exist, it will automatically create an empty list as the default value
            # dictionary will map: element ID → list of intersection points (tuples like (x, y))
            element_intersections = defaultdict(list)
            # Iterates over each row of the data DataFrame
            # data.iterrows() returns a (row_index, row_data) tuple
            # underscore _ means we’re ignoring the row index — only row (a Pandas Series) matters
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

            # Opens the file "element_gp_curvatures_normals.csv" in write mode; f is the file handle
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
                        # Project gp to NURBS parametric space by arc-length (approximate)
                        # AW 5.5: old code, outcommented
                        """ dists = np.linalg.norm(cut_points_np - gp, axis=1)
                        idx_closest = np.argmin(dists)
                        u_gp = param_values[idx_closest]  # uses precomputed arc-length params """
                        # we skip evaluating at the exact endpoints by using the inner range:
                        # u_min = curve.knotvector[p] → the first non-clamped parameter
                        # u_max = curve.knotvector[-p - 1] → the last non-clamped parameter
                        u_min = curve.knotvector[curve.degree]
                        u_max = curve.knotvector[-curve.degree - 1]
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
                        # AW 27.5: change this to see if normal direction changes accordingly
                        # nx, ny = (-y1 / tnorm, x1 / tnorm) if tnorm != 0 else (0.0, 0.0) 
                        # old line
                        nx, ny = (y1 / tnorm, -x1 / tnorm) if tnorm != 0 else (0.0, 0.0) # outward unit normal
                        normal = (nx, ny)  

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

            # AW 2.5: Debug, delete!
            if 9581 in element_gp_data:
                print(f"Element 9581 found in element_gp_data:")
                print("Curvatures:", element_gp_data[9581]["curvatures"])
                print("Normals:", element_gp_data[9581]["normals"])
            else:
                print("Element 9581 NOT found in element_gp_data!")


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


            # AW 5.5.: commented, nodal storage is legacy
            """ # AW 26.4: now store curvatures and normals nodally 
            df = pd.read_csv('curve_normals_curvature_at_intersection_points.csv')
            element_curvature = dict(zip(df['Element_ID'], df['curvature']))
            element_normal_x = dict(zip(df['Element_ID'], df['normal_x']))
            element_normal_y = dict(zip(df['Element_ID'], df['normal_y']))

            for element in self.main_model_part.Elements:
                element_id = element.Id
                if element_id in element_curvature:
                    curvature = float(element_curvature[element_id])
                    normal_x = float(element_normal_x[element_id])
                    normal_y = float(element_normal_y[element_id])
                    for node in element.GetNodes():
                        node.SetValue(KratosDroplet.FITTED_CURVATURE, curvature)
                        node.SetValue(KratosDroplet.FITTED_NORMAL, [normal_x, normal_y, 0.0]) """
            
            """ # AW 5.5: debug plotting part for code validation

            ### Plotting Part 1: global

            # Collect all Gauss points and projected points from the NURBS fitting
            # Initialize two lists to store:
            all_gauss_points = []  # Gauss points on interface segments
            all_projected_points = [] # Their corresponding (closest point) "projections" on the NURBS

            # Extract the valid parameter range of the NURBS curve (excluding clamped ends) so we stay within a well-defined region when evaluating or projecting points
            u_min = curve.knotvector[curve.degree]
            u_max = curve.knotvector[-curve.degree - 1]

            # Iterate over each element that has exactly two intersection points (i.e., properly cut by the level set)
            for eid, pts in element_intersections.items():
                if len(pts) != 2:
                    continue
                
                # Convert the two intersection points of the element to NumPy arrays (endpoints of the interface segment)
                p0, p1 = np.array(pts[0]), np.array(pts[1])
                gauss_weights = [-1/np.sqrt(3), 1/np.sqrt(3)]
                # Compute the physical positions of two Gauss points on the segment using isoparametric mapping
                gauss_points = [(0.5 * ((1 - xi) * p0 + (1 + xi) * p1)) for xi in gauss_weights]

                # For each Gauss point:
                for gp in gauss_points:
                    u_proj = find_closest_u(curve, gp, u_min, u_max) # Project it onto the NURBS curve (via closest-point search)
                    proj_pt = np.array(curve.evaluate_single(u_proj)[:2]) # Evaluate the curve at the optimal u
                    all_gauss_points.append(gp) # Store both original (...)
                    all_projected_points.append(proj_pt) # (...) and projected point

            # Convert to NumPy arrays for easier plotting
            all_gauss_points = np.array(all_gauss_points)
            all_projected_points = np.array(all_projected_points)

            # Set up a square plotting window with equal scaling on both axes
            plt.figure(figsize=(8, 8))
            plt.axis("equal")

            # Plot 1: original ordered intersection points
            plt.plot(cut_points_np[:, 0], cut_points_np[:, 1], 'bo-', label='Ordered Cut Points', markersize=3)

            # Plot 2: control points
            control_points = np.array(curve.ctrlpts)
            plt.plot(control_points[:, 0], control_points[:, 1], 'ko--', label='Control Points', markersize=5)

            # Plot 3: fitted NURBS curve
            fitted_points = np.array(curve.evalpts)
            plt.plot(fitted_points[:, 0], fitted_points[:, 1], 'r-', linewidth=2, label='Fitted NURBS Curve')

            # Plot 4: Gauss points and their projections (color-matched)
            # Create a list of distinct colors (from the plasma colormap), one for each Gauss point
            colors = cm.plasma(np.linspace(0, 1, len(all_gauss_points)))
            # For each Gauss point and its projection:
            for i, (gp, pp) in enumerate(zip(all_gauss_points, all_projected_points)):
                plt.plot(gp[0], gp[1], 'o', color=colors[i], label='Gauss Point' if i == 0 else "", markersize=5) # Plot the Gauss point as a colored dot 
                plt.plot(pp[0], pp[1], 'x', color=colors[i], label='Projection on NURBS' if i == 0 else "", markersize=5) # Plot its projection as a colored "x"
                plt.plot([gp[0], pp[0]], [gp[1], pp[1]], '--', color=colors[i], linewidth=1) # Draw a dashed line connecting the two

            # Final plot settings
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Gauss Points and Their Projection onto NURBS")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig("nurbs_gauss_projection_debug.png", dpi=300)
            plt.show()  # Uncomment for interactive use

            ### Plotting Part 2: local
            zoom_eid = 9846  # Change this ID to inspect a different element

            # Only continue if that element exists in the intersection data
            if zoom_eid in element_intersections:
                # Grab the two interface points for the zoomed-in element
                p0, p1 = np.array(element_intersections[zoom_eid][0]), np.array(element_intersections[zoom_eid][1])
                gauss_weights = [-1/np.sqrt(3), 1/np.sqrt(3)]
                # Compute the two Gauss points for this segment
                gauss_points = [(0.5 * ((1 - xi) * p0 + (1 + xi) * p1)) for xi in gauss_weights]

                projected_points = []
                u_min = curve.knotvector[curve.degree]
                u_max = curve.knotvector[-curve.degree - 1]
                for gp in gauss_points:
                    u_proj = find_closest_u(curve, gp, u_min, u_max) # Find the closest projections on the curve for the two Gauss points
                    pt_proj = np.array(curve.evaluate_single(u_proj)[:2])
                    projected_points.append(pt_proj)

                gauss_points = np.array(gauss_points)
                projected_points = np.array(projected_points)

                # Begin zoomed plot
                plt.figure(figsize=(7, 7))
                plt.axis("equal")

                # Plot interface segment (cut line) of the element under investigation
                plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'b-', label='Interface Segment')

                # Plot control points
                control_points = np.array(curve.ctrlpts)
                plt.plot(control_points[:, 0], control_points[:, 1], 'ko--', markersize=5, label='Control Points')

                # Plot Gauss points and projections
                colors = ['tab:orange', 'tab:green']
                for i in range(2):
                    gp = gauss_points[i]
                    proj = projected_points[i]
                    plt.plot(gp[0], gp[1], 'o', color=colors[i], label=f'Gauss Pt {i+1}')
                    plt.plot(proj[0], proj[1], 'x', color=colors[i], label=f'Projection {i+1}')
                    plt.plot([gp[0], proj[0]], [gp[1], proj[1]], '--', color=colors[i], linewidth=1)

                # Optional: plot local portion of NURBS
                plt.plot(fitted_points[:, 0], fitted_points[:, 1], 'r-', linewidth=2, label='Fitted NURBS Curve')

                # Highlight segment ends
                plt.plot(p0[0], p0[1], 'bo')
                plt.plot(p1[0], p1[1], 'bo')

                # Set axis limits to zoom tightly around the Gauss/projection segment, adding a 5% margin
                all_x = np.concatenate([gauss_points[:, 0], projected_points[:, 0], [p0[0], p1[0]]])
                all_y = np.concatenate([gauss_points[:, 1], projected_points[:, 1], [p0[1], p1[1]]])
                
                x_min, x_max = np.min(all_x), np.max(all_x)
                y_min, y_max = np.min(all_y), np.max(all_y)

                # Apply margin (5% of range)
                x_margin = 0.05 * (x_max - x_min)
                y_margin = 0.05 * (y_max - y_min)

                plt.xlim(x_min - x_margin, x_max + x_margin)
                plt.ylim(y_min - y_margin, y_max + y_margin)


                plt.title(f"Zoom on Element {zoom_eid}")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"zoom_element_{zoom_eid}.png", dpi=300)
                plt.show()  # Uncomment for interactive session
            else:
                print(f"Element {zoom_eid} not found in intersection data.") """

        ##################### End of Nurbs Fitting ########################

        ################### AW 22.5: Start of Unfitted Curvature Smoothing ###################
        # flag to control whether curvature smoothing is performed
        if do_curvature_smoothing:
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



            smooth_interface_curvature_and_gradient(
                self.main_model_part,
                ordered_elem_ids,
                KratosCFD.CURVATURE,
                KratosMultiphysics.DISTANCE_GRADIENT,
                window_size=2,
                polyorder=1,
                method='avg'
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

        

        
        
        # AW 27.5: added this boolean to allow using only penalty term
        normal_penalty = True
        if normal_penalty:
             # debug print, delete once it works
            print("2nd Part of Normal averaging is executed because normal_evaluation_mode is set to 3.")
            ####################### second part of normal averaging #######################
            # AW 14.5: additional normal averaging
            # for node in self.main_model_part.Nodes:
            #     gx = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X)
            #     gy = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Y)
            #     gz = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Z)
            #     g = (gx**2+gy**2+gz**2)**0.5
            #     gx /= g
            #     gy /= g
            #     gz /= g
            #     node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X,gx)
            #     node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Y,gy)
            #     node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Z,gz)
            #     if node.Y == 0.0:
            #         node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Y,-0.17365)
            #         if node.X < 0.015:
            #             node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X, -0.9848)
            #         elif node.X > 0.015:
            #             node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X,0.9848)
            contact_angle = 0
            for node in self.main_model_part.Nodes:
                if node.GetSolutionStepValue(KratosDroplet.CONTACT_ANGLE_MICRO,0) != 0.0:
                    contact_angle = node.GetSolutionStepValue(KratosDroplet.CONTACT_ANGLE_MICRO,0)

            # AW 22.5: updated to consider equilibrium contact angle (100 in this case) here
            diff = abs(contact_angle - 100)
            if diff < 1:
                beta = 1
            elif diff > 9:
                beta = 0
            else:
                beta = 0.5*(1+math.cos(3.1416*(diff-1.0)/8))
            print("beta=",beta)

            for node in self.main_model_part.Nodes:
                gx = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X)
                gy = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Y)
                gz = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Z)
                g = (gx**2+gy**2+gz**2)**0.5
                gx /= g
                gy /= g
                gz /= g
                node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X,gx)
                node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Y,gy)
                node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Z,gz)
                if node.Y == 0.0:
                    if beta > 0.0:
                        gy = math.cos(contact_angle*3.1416/180) + beta*(math.cos(100.0*3.1416/180)- math.cos(contact_angle*3.1416/180))
                        if node.X > 0.015:
                            gx = math.sin(contact_angle*3.1416/180) + beta*(math.sin(100.0*3.1416/180)- math.sin(contact_angle*3.1416/180))
                        elif node.X < 0.015:
                            gx = -(math.sin(contact_angle*3.1416/180) + beta*(math.sin(100.0*3.1416/180)- math.sin(contact_angle*3.1416/180)))

                        g = (gx**2+gy**2+gz**2)**0.5

                        node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X,gx/g)
                        node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Y,gy/g)
                        node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Z,gz/g) 


                # if node.Y == 0.0:
                #     node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Y,0.50754)
                #     node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Z,0.0)
                #     if node.X < 0.015:
                #         node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X,-0.86163)
                #     elif node.X > 0.015:
                #         node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X,0.86163)

            # nx1=-1
            # nx2=1
            # ny1=ny2=0
            # for node in self.main_model_part.Nodes:
            #     nx=ny=0
            #     if node.Is(KratosMultiphysics.BOUNDARY):
            #         nx = node.GetSolutionStepValue(KratosDroplet.NORMAL_VECTOR_X)
            #         ny = node.GetSolutionStepValue(KratosDroplet.NORMAL_VECTOR_Y)
            #         diss = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE)
            #         if -0.001<diss <0.001 and 0.5 < (nx**2 + ny**2)**0.5:
            #             gx = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X)
            #             gy = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Y)
            #             if node.X < 0.015:
            #                 nx1 = nx
            #                 ny1 = ny
            #                 if node.Is(KratosMultiphysics.BOUNDARY):
            #                     node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X,0)
            #                     node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Y,0)
            #                 else:
            #                     nx1 = (3*nx1-gx)/2
            #                     ny1 = (3*ny1-gy)/2
            #             elif node.X > 0.015:
            #                 nx2 = nx
            #                 ny2 = ny
            #                 if node.Is(KratosMultiphysics.BOUNDARY):
            #                     node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X,0)
            #                     node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Y,0)
            #                 else:
            #                     nx2 = (3*nx2-gx)/2
            #                     ny2 = (3*ny2-gy)/2


            
            # for node in self.main_model_part.Nodes:
            #     if node.Is(KratosMultiphysics.BOUNDARY):
            #         if node.Y == 0.0:
            #             if node.X < 0.015:
            #                 node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X,nx1)
            #                 node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Y,ny1)
            #             elif node.X > 0.015:
            #                 node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X,nx2)
            #                 node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Y,ny2)
            #################################################################################################


            # for node in self.main_model_part.Nodes:
            #     if node.Is(KratosMultiphysics.BOUNDARY):
            #         gx = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X)
            #         gy = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Y)
            #         if gx > 0.0:
            #             nx = 0.9848
            #         elif gx < 0.0:
            #             nx = -0.9848
            #         if gy > 0.0:
            #             ny = 0.1736
            #         elif gy < 0.0:
            #             ny = -0.1736
            #         if gx != 0.0 or gy != 0.0:
            #             g = (gx**2+gy**2)**(0.5)
            #             gx = nx * g
            #             gy = ny *g
            #             node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X,gx)
            #             node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Y,gy)

            # nx1=-1
            # nx2=1
            # ny1=ny2=0
            # for node in self.main_model_part.Nodes:
            #     nx=ny=0
            #     if node.Is(KratosMultiphysics.BOUNDARY):
            #         nx = node.GetSolutionStepValue(KratosDroplet.NORMAL_VECTOR_X)
            #         ny = node.GetSolutionStepValue(KratosDroplet.NORMAL_VECTOR_Y)
            #         diss = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE)
            #         if -0.001<diss <0.001 and 0.5<=(nx**2 + ny**2)**0.5:
            #             if node.X < 0.015:
            #                 nx1 = nx
            #                 ny1 = ny
            #             elif node.X > 0.015:
            #                 nx2 = nx
            #                 ny2 = ny

            # for node in self.main_model_part.Nodes:
            #     if node.Is(KratosMultiphysics.BOUNDARY):
            #         if node.Y == 0.0:
            #             gx = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X)
            #             gy = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Y)
            #             g = (gx**2+gy**2)**(0.5)
            #             diss = node.GetSolutionStepValue(KratosMultiphysics.DISTANCE)
            #             if node.X < 0.015 and -0.001 < diss < 0.001:
            #                 gx = nx1 * g
            #                 gy = ny1 *g
            #                 node.SetSolutionStepValue(KratosDroplet.NORMAL_VECTOR_X,nx1)
            #                 node.SetSolutionStepValue(KratosDroplet.NORMAL_VECTOR_Y,ny1)
            #             elif node.X > 0.015 and -0.001 < diss < 0.001:
            #                 gx = nx2 * g
            #                 gy = ny2 *g
            #                 node.SetSolutionStepValue(KratosDroplet.NORMAL_VECTOR_X,nx2)
            #                 node.SetSolutionStepValue(KratosDroplet.NORMAL_VECTOR_Y,ny2)
            #             node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_X,gx)
            #             node.SetSolutionStepValue(KratosMultiphysics.DISTANCE_GRADIENT_Y,gy)
            # AW 14.5: end of additional normal averaging
            ####################### end of second part of normal averaging ###################

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

        # Initialize the solver current step
        self._GetSolutionStrategy().InitializeSolutionStep()

        # We set this value at every time step as other processes/solvers also use them
        dynamic_tau = self.settings["formulation"]["dynamic_tau"].GetDouble()
        self.main_model_part.ProcessInfo.SetValue(KratosMultiphysics.DYNAMIC_TAU, dynamic_tau)


    def Predict(self):
        self._GetSolutionStrategy().Predict()


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
        if self._reinitialization_type != "none":
            step = self.main_model_part.ProcessInfo[KratosMultiphysics.STEP]
            if step == 2 or step%12==0:
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
            layers = self.settings["parallel_redistance_max_layers"].GetInt()
            parallel_distance_settings = KratosMultiphysics.Parameters("""{
                "max_levels" : 25,
                "max_distance" : 1.0,
                "calculate_exact_distances_to_plane" : true
            }""")
            parallel_distance_settings["max_levels"].SetInt(layers)
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


