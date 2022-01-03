"""
Custom compartment models with contact networks
"""
# Standard Libraries
import copy
import json

# External Libraries
import networkx as networkx
import numpy as np
import scipy as scipy
import scipy.integrate

# Internal Libraries
from seirsplus.models.compartment_model_builder import CompartmentModelBuilder
from seirsplus import utils

import itertools
import time


class CompartmentNetworkModel():

    def __init__(self, 
                    compartments, 
                    networks,
                    mixedness=0.0, 
                    openness=0.0,
                    isolation_period=None,
                    transition_mode='exponential_rates', 
                    local_trans_denom_mode='all_contacts',
                    prevalence_flags=['active_infection'],
                    track_case_info=False,
                    store_Xseries=False,
                    node_groups=None,
                    seed=None):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update model execution options:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(seed is not None):
            np.random.seed(seed)
            self.seed = seed

        self.transition_mode        = transition_mode
        self.transition_timer_wt    = 1e5
        self.local_trans_denom_mode = local_trans_denom_mode
        self.track_case_info        = track_case_info

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the contact networks specifications:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.pop_size = None # will be updated in update_networks()
        self.networks = {}
        self.update_networks(copy.copy(networks))

        self.mixedness = mixedness
        self.openness  = openness

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize compartment metadata:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.stateID               = {}
        self.default_state         = None #  list(self.compartments.keys())[0] # default to first compartment specified
        self.excludeFromEffPopSize = []
        self.node_flags            = [[] for i in range(self.pop_size)]
        self.allNodeFlags          = set() 
        self.allCompartmentFlags   = set()
        self.counts                = {}
        self.flag_counts           = {} 
        self.track_flag_counts     = True
        self.store_Xseries         = store_Xseries

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the compartment model configuration and parameterizations:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.compartments    = {}
        self.infectivity_mat = {} 
        self.update_compartments(compartments)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize timekeeping:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.t       = 0 # current sim time
        self.tmax    = 0 # max sim time (will be set when run() is called)
        self.tidx    = 0 # current index in list of timesteps
        self.tseries = np.zeros(self.pop_size*min(len(self.compartments), 10))

        # Vectors holding the time that each node has their current state:
        self.state_timer        = np.zeros((self.pop_size,1))

        # Vectors holding the isolation status and isolation time for each node:
        self.isolation          = np.zeros(self.pop_size).astype(np.int)
        self.isolation_period   = isolation_period
        self.isolation_timer    = np.zeros(self.pop_size)
        self.totalIsolationTime = np.zeros(self.pop_size)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize testing, vaccination, etc:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.test_params    = {}
        self.test_types     = set()
        self.vaccine_series = {}

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize data series for tracking node subgroups:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.nodeGroupData = None
        if(node_groups):
            self.nodeGroupData = {}
            for groupName, nodeList in node_groups.items():
                self.nodeGroupData[groupName] = {'nodes':   np.array(nodeList),
                                                 'mask':    np.in1d(range(self.pop_size), nodeList).reshape((self.pop_size,1))}
                for compartment in self.compartments:
                    self.nodeGroupData[groupName][compartment]    = np.zeros(self.pop_size*min(len(self.compartments), 10))
                    self.nodeGroupData[groupName][compartment][0] = np.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.S)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize other metadata:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.caseLogs = []
        self.node_attributes = {}
        self.prevalence_flags = prevalence_flags

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize counts/prevalences and the states of individuals:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.process_initial_states()
            
    
    ########################################################
    ########################################################


    def update_networks(self, new_networks):
        if(not isinstance(new_networks, dict)):
            raise BaseException("Specify networks with a dictionary of adjacency matrices or networkx objects.")
        else:
            # Store both a networkx object and a np adjacency matrix representation of each network:
            for netID, network in new_networks.items():
                if type(network)==np.ndarray:
                    new_networks[netID] = {"networkx":  networkx.from_numpy_matrix(network), 
                                         "adj_matrix":  scipy.sparse.csr_matrix(network)}
                elif type(network)==networkx.classes.graph.Graph:
                    new_networks[netID] = {"networkx":  network, 
                                         "adj_matrix":  networkx.adj_matrix(network)}
                else:
                    raise BaseException("Network", netID, "should be specified by an adjacency matrix or networkx object.")
                # Store the number of nodes and node degrees for each network:
                new_networks[netID]["num_nodes"] = int(new_networks[netID]["adj_matrix"].shape[1])
                new_networks[netID]["degree"]    = new_networks[netID]["adj_matrix"].sum(axis=0).reshape(new_networks[netID]["num_nodes"],1)
                # Set all individuals to be active participants in this network by default:
                new_networks[netID]['active']            = np.ones(new_networks[netID]["num_nodes"])
                # Set all individuals to be inactive in this network when in isolation by default:
                new_networks[netID]["active_isolation"]  = np.zeros(new_networks[netID]["num_nodes"])
            
            self.networks.update(new_networks)

            # Ensure all networks have the same number of nodes:
            for key, network in self.networks.items():
                if(self.pop_size is None):
                    self.pop_size = network["num_nodes"]
                if(network["num_nodes"] !=  self.pop_size):
                    raise BaseException("All networks must have the same number of nodes.")


    ########################################################


    def update_compartments(self, new_compartments):
        if(isinstance(new_compartments, str) and '.json' in new_compartments):
            with open(new_compartments) as compartments_file:
                new_compartments = json.load(compartments_file)
        elif(isinstance(new_compartments, dict)):
            pass
        elif(isinstance(new_compartments, CompartmentModelBuilider)):
            new_compartments = new_compartments.compartments
        else:
            raise BaseException("Specify compartments with a dictionary or JSON file.")

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Recursively pre-process and reshape parameter values for all compartments:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def reshape_param_vals(nested_dict):
            for key, value in nested_dict.items():
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Do not recurse or reshape these params
                if(key in ['transmissibilities', 'initial_prevalence', 'exogenous_prevalence', 'default_state', 'exclude_from_eff_pop', 'flags', 'vaccinated', 'vaccine_series']):
                    pass
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Recurse through sub dictionaries
                elif(isinstance(value, dict)):
                    reshape_param_vals(value)
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Convert all other parameter values to arrays corresponding to the population size:
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                else:
                    nested_dict[key] = utils.param_as_array(value, (self.pop_size,1))
        #----------------------------------------
        reshape_param_vals(new_compartments)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Recursively process transition probabilities:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def process_transition_params(nested_dict):
            for key, value in nested_dict.items():
                if(key == 'transitions' and len(value) > 0):
                    transn_dict = value
                    poststates  = list(transn_dict.keys())
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Ensure all transitions have a specified rate or time, as applicable:
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    self.process_transition_times(transn_dict)
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Decide the transition each individual will take according to given probabilities:
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    self.process_transition_probs(transn_dict)
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Recurse through sub dictionaries
                elif(isinstance(value, dict) and key != 'transitions'):
                    process_transition_params(value)
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Else do nothing
                else:
                    pass
        #----------------------------------------
        process_transition_params(new_compartments)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Transmissibility parameters are preprocessed and shaped into pairwise matrices
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for compartment, comp_dict in new_compartments.items():
            transm_dict = comp_dict['transmissibilities']
            if(len(transm_dict) == 0):
                pass
            #----------------------------------------        
            if('pairwise_mode' not in transm_dict):
                transm_dict['pairwise_mode'] = 'infected' # default when not provided
            #----------------------------------------
            if('local_transm_offset_mode' not in transm_dict):
                transm_dict['local_transm_offset_mode'] = 'none' # default when not provided
            #----------------------------------------    
            self.infectivity_mat[compartment] = {}
            for network in self.networks:
                #----------------------------------------
                # Process local transmissibility parameters for each network:
                #----------------------------------------
                self.process_network_transmissibility(transm_dict, network)
                #----------------------------------------
                # Process frequency-dependent transmission offset factors for each network:
                #----------------------------------------
                self.process_network_transm_offsets(transm_dict, network)
                #----------------------------------------
                # Pre-calculate Infectivity Matrices for each network,
                # which pre-combine transmissibility, adjacency, and freq-dep offset terms.
                #----------------------------------------
                # M_G = (AB)_G * D_G
                self.infectivity_mat[compartment][network] = scipy.sparse.csr_matrix.multiply(transm_dict[network].astype(float), transm_dict['offsets'][network].astype(float))
            #----------------------------------------
            if('exogenous' not in transm_dict or not isinstance(transm_dict['exogenous'], (int, float))):
                transm_dict['exogenous'] = 0.0
            #----------------------------------------
            if('global' not in transm_dict or not isinstance(transm_dict['global'], (int, float))):
                transm_dict['global'] = np.sum([np.sum(transm_dict[network][transm_dict[network]!=0]) for network in self.networks]) / max(np.sum([transm_dict[network].count_nonzero() for network in self.networks]), 1)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check the initial and exogenous_prevalence params for each compartment, defaulting to 0 when missing or invalid:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for compartment, comp_dict in new_compartments.items():
            if('initial_prevalence' not in comp_dict or not isinstance(comp_dict['initial_prevalence'], (int, float))):
                comp_dict['initial_prevalence'] = 0.0
            if('exogenous_prevalence' not in comp_dict or not isinstance(comp_dict['exogenous_prevalence'], (int, float))):
                comp_dict['exogenous_prevalence'] = 0.0

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize compartment IDs/metadata:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for c, compartment in enumerate(new_compartments):
            comp_params = new_compartments[compartment]
            #----------------------------------------
            # Assign state ID number to each compartment (for internal state comparisons):
            if(compartment not in self.stateID):
                self.stateID[compartment] = len(list(self.stateID.keys())) + 1 
            #----------------------------------------
            # Update the default compartment for this model:
            if('default_state' in comp_params and comp_params['default_state']==True):
                self.default_state = compartment
            #----------------------------------------
            # Update which compartments are excluded when calculating effective population size (N):
            if('exclude_from_eff_pop' in comp_params and comp_params['exclude_from_eff_pop']==True):
                self.excludeFromEffPopSize.append(compartment)
            #----------------------------------------
            # Update which compartment flags are in use:
            if('flags' not in new_compartments[compartment]):
                new_compartments[compartment]['flags'] = []
            for flag in new_compartments[compartment]['flags']:
                self.allCompartmentFlags.add(flag)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate data series for counts for any new compartments:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for c, compartment in enumerate(new_compartments):
            if(compartment not in self.counts):
                self.counts[compartment] = (np.zeros_like(self.counts[self.default_state]) if self.default_state is not None and self.default_state in self.counts
                                            else np.zeros(self.pop_size*min(len(self.compartments)+len(new_compartments), 10)))

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the model object with the new processed compartments
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.compartments.update(new_compartments)


    ########################################################
    ########################################################


    def calc_propensities(self):
        # print("CALC PROPENSITIES")
        # time_calcprop_start = time.time()

        propensities     = []
        transitions      = []

        for compartment, comp_params in self.compartments.items():

            # time_calcprop_comp_start = time.time()

            # Skip calculations for this compartment if no nodes are in this state:
            if(not np.any(self.X==self.stateID[compartment])):
                continue

            # print("time_calcprop", compartment, "A", time.time() - time_calcprop_comp_start)

            #----------------------------------------
            # Dict to store calcualted propensities of local infection for each infectious state
            # so that these local propensity terms do not have to be calculated more than once 
            # if needed for multiple susceptible states
            propensity_infection_local = {}

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Calc propensities of temporal transitions:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for destState, transition_params in comp_params['transitions'].items():
                if(destState not in self.compartments):
                    print("Destination state", destState, "is not a defined compartment.")
                    continue

                if(self.transition_mode == 'time_in_state'):
                    propensity_temporal_transition = (self.transition_timer_wt * (np.greater(self.state_timer, transition_params['time']) & (self.X==self.stateID[compartment])) * transition_params['path_taken']) if any(transition_params['time']) else np.zeros_like(self.X)

                else: # exponential_rates
                    propensity_temporal_transition = transition_params['rate'] * (self.X==self.stateID[compartment]) * transition_params['path_taken']

                propensities.append(propensity_temporal_transition)
                transitions.append({'from':compartment, 'to':destState, 'type':'temporal'})

            # print("time_calcprop", compartment, "B", time.time() - time_calcprop_comp_start)

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Calc propensities of transmission-induced transitions:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # print("-----------------")
            # time_calcprop_comp_local_start = time.time()

            for infectiousState, susc_params in comp_params['susceptibilities'].items():
                
                # time_calcprop_comp_inf_start = time.time()

                if(infectiousState not in self.compartments):
                    print("Infectious state", infectiousState, "is not a defined compartment.")
                    continue

                # Skip calculations for this infectious compartment if no nodes are in this state:
                if(not np.any(self.X==self.stateID[infectiousState])):
                    continue

                # print("      time_0", time.time() - time_calcprop_comp_inf_start)

                # #----------------------------------------
                # # Get the number of contacts relevant for the local transmission denominator for each individual:
                # #----------------------------------------
                # self.active_degree = np.zeros((self.pop_size, 1))
                # for netID, G in self.networks.items():
                #     bool_isGactive    = (((G['active']!=0)&(self.isolation==0)) | ((G['active_isolation']!=0)&(self.isolation!=0))).flatten()
                #     self.active_degree += G['adj_matrix'][:,np.argwhere(bool_isGactive).flatten()].sum(axis=1) if self.local_trans_denom_mode=='active_contacts' else G['degree']

                #----------------------------------------
                # Compute the local transmission propensity terms for individuals in each contact network
                #----------------------------------------



                if(infectiousState not in propensity_infection_local):

                    propensity_infection_local[infectiousState] = np.zeros((self.pop_size, 1))

                    # self.active_degree = np.zeros((self.pop_size, 1))

                    # #----------------------------------------
                    # # Get the number of contacts relevant for the local transmission denominator for each individual:
                    # for netID, G in self.networks.items():
                    #     bool_isGactive    = (((G['active']!=0)&(self.isolation==0)) | ((G['active_isolation']!=0)&(self.isolation!=0))).flatten()
                    #     self.active_degree += G['adj_matrix'][:,np.argwhere(bool_isGactive).flatten()].sum(axis=1) if self.local_trans_denom_mode=='active_contacts' else G['degree']

                    #----------------------------------------
                    # Compute the local transmission propensity terms:
                    #----------------------------------------
                    for netID, G in self.networks.items():

                        # timenet_a = time.time()
                        M = self.infectivity_mat[infectiousState][netID]
                        # print("        timenet_a", time.time() - timenet_a)

                        #########################################
                        # VERSION 1
                        #########################################

                        # #----------------------------------------
                        # # Determine which individuals need local transmission propensity calculated (active in G and infectible, non-zero propensity)
                        # # and which individuals are relevant in these calculations (active in G and infectious):
                        # #----------------------------------------
                        # bool_isGactive    = (((G['active']!=0)&(self.isolation==0)) | ((G['active_isolation']!=0)&(self.isolation!=0))).flatten()
                        # bin_isGactive     = [1 if i else 0 for i in bool_isGactive]

                        # bool_isInfectious = (self.X==self.stateID[infectiousState]).flatten()
                        # j_isInfectious    = np.argwhere(bool_isInfectious).flatten()

                        # bool_hasGactiveInfectiousContacts = np.asarray(scipy.sparse.csr_matrix.dot(M, scipy.sparse.diags(bin_isGactive))[:,j_isInfectious].sum(axis=1).astype(bool)).flatten()

                        # bool_isInfectible = (bool_isGactive & bool_hasGactiveInfectiousContacts)
                        # i_isInfectible    = np.argwhere(bool_isInfectible).flatten()

                        # #----------------------------------------
                        # # Compute the local transmission propensity terms for individuals in the current contact network G
                        # #----------------------------------------
                        # propensity_infection_local[infectiousState][i_isInfectible] += np.divide( scipy.sparse.csr_matrix.dot(M[i_isInfectible,:][:,j_isInfectious], (self.X==self.stateID[infectiousState])[j_isInfectious]), self.active_degree[i_isInfectible], out=np.zeros_like(propensity_infection_local[infectiousState][i_isInfectible]), where=self.active_degree[i_isInfectible]!=0 )


                        #########################################
                        # VERSION 2
                        #   (no computing or slicing using i_isInfectible)
                        #########################################

                        # #----------------------------------------
                        # # Determine which individuals are relevant in these calculations (active in G and infectious):
                        # #----------------------------------------
                        # # timenet_c = time.time()
                        # bool_isInfectious = (self.X==self.stateID[infectiousState]).flatten()
                        # j_isInfectious    = np.argwhere(bool_isInfectious).flatten()
                        # # print("        timenet_c", time.time() - timenet_c)

                        # #----------------------------------------
                        # # Compute the local transmission propensity terms for individuals in the current contact network G
                        # #----------------------------------------
                        # # timenet_fv2 = time.time()
                        # propensity_infection_local[infectiousState] += np.divide( scipy.sparse.csr_matrix.dot(M[:,j_isInfectious], (self.X==self.stateID[infectiousState])[j_isInfectious]), self.active_degree, out=np.zeros_like(propensity_infection_local[infectiousState]), where=self.active_degree!=0 )
                        # # print("        timenet_fv2", time.time() - timenet_fv2)


                        #########################################
                        # VERSION 3
                        #   (no computing or slicing using i_isInfectible or j_isInfectious)
                        #########################################
                        # !!!>>> This is the fastest for the N=1000 simulations for the Color isolation-policies memo. Test speeds at larger networks.
                        #----------------------------------------
                        # Compute the local transmission propensity terms for individuals in the current contact network G
                        #----------------------------------------
                        # timenet_fv2 = time.time()
                        propensity_infection_local[infectiousState] += np.divide( scipy.sparse.csr_matrix.dot(M, (self.X==self.stateID[infectiousState])), self.active_degree, out=np.zeros_like(propensity_infection_local[infectiousState]), where=self.active_degree!=0 )
                        # print("        timenet_fv2", time.time() - timenet_fv2)

                #----------------------------------------
                # Compute the propensities of infection for individuals across all transmission modes (exogenous, global, local over all networks)
                #----------------------------------------
                # timeeqn = time.time()
                transm_params = self.compartments[infectiousState]['transmissibilities']

                propensity_infection = ((self.X==self.stateID[compartment]) * 
                                        (
                                            susc_params['susceptibility'] *
                                            (    
                                                 (self.openness) * (transm_params['exogenous']*self.compartments[compartment]['exogenous_prevalence'])
                                             + (1-self.openness) * (
                                                                        (self.mixedness) * ((transm_params['global']*np.count_nonzero((self.isolation^1)*(self.X==self.stateID[infectiousState]).flatten()))/self.N[self.tidx])   # the (self.isolation^1) inverts the 0/1 isolation vals using the ^ XOR operator; this is to exclude isolated individuals from global transmission
                                                                    + (1-self.mixedness) * (propensity_infection_local[infectiousState])
                                                                   )
                                            )
                                        ))
                # print("      timeeqn", time.time() - timeeqn)

                #----------------------------------------
                # Compute the propensities of each possible infection-induced transition according to the disease progression paths of each individual:
                #----------------------------------------
                # time_Z = time.time()
                for destState, transition_params in susc_params['transitions'].items():
                    if(destState not in self.compartments):
                        print("Destination state", destState, "is not a defined compartment.")
                        continue

                    propensity_infection_transition = propensity_infection * transition_params['path_taken']

                    propensities.append(propensity_infection_transition)
                    transitions.append({'from':compartment, 'to':destState, 'type':'infection'})
                # print("      time_Z", time.time() - time_Z)

                # print("  time_calcprop", compartment, infectiousState, time.time() - time_calcprop_comp_inf_start)

            #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            # print("time_calcprop", compartment, "local", time.time() - time_calcprop_comp_local_start)
        
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        propensities = np.hstack(propensities) if len(propensities)>0 else np.array([[]])
        # print("time_calcprop", time.time() - time_calcprop_start)
        return propensities, transitions
        

    ########################################################
    ########################################################


    def run_iteration(self, default_dt=0.1, max_dt=None, tau_step=None):
        
        max_dt = self.tmax if max_dt is None else max_dt

        if(self.tidx >= len(self.tseries)-1):
            # Room has run out in the timeseries storage arrays; double the size of these arrays:
            self.increase_data_series_length()

        # Get the number of contacts relevant for the local transmission denominator for each individual:
        self.active_degree = np.zeros((self.pop_size, 1))
        for netID, G in self.networks.items():
            bool_isGactive    = (((G['active']!=0)&(self.isolation==0)) | ((G['active_isolation']!=0)&(self.isolation!=0))).flatten()
            self.active_degree += G['adj_matrix'][:,np.argwhere(bool_isGactive).flatten()].sum(axis=1) if self.local_trans_denom_mode=='active_contacts' else G['degree']

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Calculate propensities
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        propensities, transitions = self.calc_propensities()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Determine the time step and state update(s):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(propensities.sum() > 0):

            if(tau_step is not None):
                #----------------------------------------
                # Simplified tau-leaping Gillespie (fixed tau):
                #----------------------------------------
                # Set the iteration time step dt to the fixed interval tau_step:
                dt = tau_step  
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Poisson draw the number of events for each node/transition in this interval:
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                numEventOccurences = np.random.poisson(lam=propensities*tau_step)
                numEventOccurences[numEventOccurences > 0] = 1 # Events are defined by a node and a transition and therefore can occur at most once per iteration
                # In the event that node(s) are expected to undergo more than 1 transition event in this interval,
                # randomly select the transition event that actually occurs proportional to their propensities:
                multiEventNodes = np.argwhere(np.sum(numEventOccurences, axis=1) >1).flatten()
                for multiEventNode in multiEventNodes:
                    selectedTransition = np.random.choice(range(len(transitions)), p=propensities[multiEventNode,:].flatten()/np.sum(propensities[multiEventNode,:].flatten()))
                    numEventOccurences[multiEventNode][range(len(transitions))!=selectedTransition] = 0
                # Place each transition event that is to occur in a list for execution below:
                transitionEvents = []
                for transitionNode, transitionIdx in zip(*np.where(numEventOccurences > 0)):
                    transition = copy.deepcopy(transitions[transitionIdx])
                    transition.update({'node': transitionNode})
                    transitionEvents.append( transition )

            else:
                #----------------------------------------
                # Standard Gillespie Stochastic Simulation Algorithm:
                #----------------------------------------
                # Generate 2 random numbers uniformly distributed in (0,1)
                r1 = np.random.rand()
                r2 = np.random.rand()
                # Calculate alpha
                propensities_flat   = propensities.ravel(order='F')
                cumsum              = propensities_flat.cumsum()
                alpha               = propensities_flat.sum()
                # Compute the time until the next event takes place
                tau = (1/alpha)*np.log(float(1/r1))
                # Set the iteration time step dt and determine the event to take place:
                if(tau <= max_dt):
                    # Set the iteration time step dt to the time to next event tau:
                    dt  = tau
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Draw the event to take place proportional to propensities:
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    transitionIdx    = np.searchsorted(cumsum,r2*alpha)
                    transitionNode   = transitionIdx % self.pop_size
                    transition       = transitions[ int(transitionIdx/self.pop_size) ]
                    # Place the transition event that is to occur in a list for execution below:
                    transition.update({'node': transitionNode})
                    transitionEvents = [transition]
                else: # (tau > max_dt):
                    # Set the iteration time step dt to the max time step:
                    dt = max_dt
                    # No event takes place during this time step:
                    transitionEvents = []

        else: # (propensities.sum()==0):
            # Set the iteration time step dt to the default time step:
            dt = default_dt
            # No event takes place during this time step:
            transitionEvents = []

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Advance time:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.t           += dt
        self.state_timer += dt
        self.tidx        += 1

        # print("t = %.3f" % self.t)

        # Update isolation timers/statuses
        i_isolated = np.argwhere(self.isolation==1).flatten()
        self.isolation_timer[i_isolated]    += dt
        self.totalIsolationTime[i_isolated] += dt
        if(self.isolation_period is not None):
            i_exitingIsolation = np.argwhere(self.isolation_timer >= self.isolation_period).flatten()
            for i in i_exitingIsolation:
                self.set_isolation(node=i, isolation=False)

        # Update the current cumulative num cases to the value from the last time point,
        # the value for the current time point will be updated for any new cases below:
        self.cum_num_cases[self.tidx] = self.cum_num_cases[self.tidx-1]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform updates triggered by event:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for transition in transitionEvents:
            assert(self.X[transition['node']]==self.stateID[transition['from']]), "Assertion error: Node "+str(transition['node'])+" has unexpected current state "+str(self.X[transition['node']])+" given the intended transition of "+transition['from']+"->"+transition['to']+"."
            self.set_state(transition['node'], transition['to']) 
            # print('   ', transition['from'], '-->', transition['to'])
            #----------------------------------------
            # Gather and save information about transmission events when they occur:
            #----------------------------------------
            if(transition['type'] == 'infection'):
                self.cum_num_cases[self.tidx] += 1
                if(self.track_case_info):
                    self.process_new_case(transition['node'], transition)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update model data series and metadata:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.update_data_series()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Terminate if tmax reached:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(self.t >= self.tmax):
            self.finalize_data_series()
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        return True


    ########################################################
    ########################################################


    def run(self, T, checkpoints=None, max_dt=None, default_dt=0.1):
        if(T>0):
            self.tmax += T
        else:
            return False

        if(max_dt is None and self.transition_mode=='time_in_state'):
            max_dt = 1

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-process checkpoint values:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # TODO 

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Run the simulation loop:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        running     = True
        while running:

            print(self.t)

            running = self.run_iteration(max_dt=max_dt, default_dt=default_dt)

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Handle checkpoints if applicable:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # TODO

        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        self.finalize_data_series()

        return True
    

    ########################################################
    ########################################################


    def update_data_series(self):
        # time_updtsers_start = time.time()
        # time_updtsers_startA = time.time()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the time series:
        self.tseries[self.tidx]     = self.t
        # print("\ttime_updtsers A\t\t", time.time() - time_updtsers_startA)
        # time_updtsers_startB = time.time()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the data series of counts of nodes in each compartment:
        for compartment in self.compartments:
            self.counts[compartment][self.tidx] = np.count_nonzero(self.X==self.stateID[compartment])
            #------------------------------------
            if(compartment not in self.excludeFromEffPopSize):
                self.N[self.tidx] += self.counts[compartment][self.tidx]
        # print("\ttime_updtsers B\t\t", time.time() - time_updtsers_startB)
        # time_updtsers_startC = time.time()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the data series of counts of nodes with each flag:
        if(self.track_flag_counts):
            for flag in self.allCompartmentFlags.union(self.allNodeFlags):
                # print("\t\tcalc flag count for", flag)
                if(flag not in self.flag_counts):
                    self.flag_counts[flag] = np.zeros_like(self.counts[self.default_state])
                # time_updtsers_startC1 = time.time()
                flag_count = len(self.get_individuals_by_flag(flag))
                # print("\t\ttime_updtsers C1\t", time.time() - time_updtsers_startC1)
                self.flag_counts[flag][self.tidx] = flag_count
        # print("\ttime_updtsers C\t\t", time.time() - time_updtsers_startC)
        # time_updtsers_startD = time.time()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store system states
        if(self.store_Xseries):
            self.Xseries[self.tidx,:] = self.X.T
        # print("\ttime_updtsers D\t\t", time.time() - time_updtsers_startD)
        # time_updtsers_startE = time.time()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store system states for specified subgroups
        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                for compartment in self.compartments:
                    self.nodeGroupData[groupName][compartment][self.tidx] = np.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.stateID[compartment])
                    #------------------------------------
                    if(compartment not in self.excludeFromEffPopSize):
                        self.nodeGroupData[groupName]['N'][self.tidx] += self.counts[compartment][self.tidx]
        # print("\ttime_updtsers E\t\t", time.time() - time_updtsers_startE)
        # time_updtsers_startF = time.time()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize/reset data series for cumulative num cases:
        if(self.tidx == 0):    
            self.cum_num_cases    = np.zeros_like(self.counts[self.default_state])
            self.cum_num_cases[0] = len(self.get_individuals_by_flag(self.prevalence_flags))
        # print("\ttime_updtsers F\t\t", time.time() - time_updtsers_startF)
        # print("time_updtsers\t\t", time.time() - time_updtsers_start)


    ########################################################


    def increase_data_series_length(self):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Allocate more entries for the time series:
        self.tseries = np.pad(self.tseries, [(0, self.pop_size*min(len(self.compartments), 10))], mode='constant', constant_values=0)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Allocate more entries for the cumulative num cases series:
        self.cum_num_cases = np.pad(self.cum_num_cases, [(0, self.pop_size*min(len(self.compartments), 10))], mode='constant', constant_values=0)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Allocate more entries for the data series of counts of nodes in each compartment:
        for compartment in self.compartments:
            self.counts[compartment] = np.pad(self.counts[compartment], [(0, self.pop_size*min(len(self.compartments), 10))], mode='constant', constant_values=0)
        #------------------------------------
        self.N = np.pad(self.N, [(0, self.pop_size*min(len(self.compartments), 10))], mode='constant', constant_values=0)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Allocate more entries for the data series of counts of nodes with each flag:
        if(self.track_flag_counts):
            for flag in self.allCompartmentFlags.union(self.allNodeFlags):
                self.flag_counts[flag] = np.pad(self.flag_counts[flag], [(0, self.pop_size*min(len(self.compartments), 10))], mode='constant', constant_values=0)
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store system states
        if(self.store_Xseries):
            self.Xseries = self.Xseries[:self.tidx+1, :]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store system states for specified subgroups
        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                for compartment in self.compartments:
                    self.nodeGroupData[groupName][compartment] = np.pad(self.nodeGroupData[groupName][compartment], [(0, self.pop_size*min(len(self.compartments), 10))], mode='constant', constant_values=0)
                #------------------------------------
                self.nodeGroupData[groupName]['N'] = np.pad(self.nodeGroupData[groupName]['N'], [(0, self.pop_size*min(len(self.compartments), 10))], mode='constant', constant_values=0)
                #------------------------------------
                # TODO: Allocate more entries for the data series of counts of nodes that have certain conditions?
                #        - infected, tested, vaccinated, positive, etc?


    ########################################################
    

    def finalize_data_series(self):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Finalize the time series:
        self.tseries = np.array(self.tseries, dtype=float)[:self.tidx+1]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Finalize the cumulative num cases series:
        self.cum_num_cases = np.array(self.cum_num_cases, dtype=float)[:self.tidx+1]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Finalize the data series of counts of nodes in each compartment:
        for compartment in self.compartments:
            self.counts[compartment] = np.array(self.counts[compartment], dtype=float)[:self.tidx+1]
        #------------------------------------
        self.N = np.array(self.N, dtype=float)[:self.tidx+1]
        #----------------------------------------
        # TODO: Finalize the data series of counts of nodes that have certain conditions?
        #        - infected, tested, vaccinated, positive, etc?
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store system states
        if(self.store_Xseries):
            self.Xseries = self.Xseries[:self.tidx+1, :]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store system states for specified subgroups
        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                for compartment in self.compartments:
                    self.nodeGroupData[groupName][compartment] = np.array(self.nodeGroupData[groupName][compartment], dtype=float)[:self.tidx+1]
                #------------------------------------
                self.nodeGroupData[groupName]['N'] = np.array(self.nodeGroupData[groupName]['N'], dtype=float)[:self.tidx+1]
                #------------------------------------
                # TODO: Finalize the data series of counts of nodes that have certain conditions?
                #        - infected, tested, vaccinated, positive, etc?

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compile summary results
        self.results = { 'pop_size': self.pop_size,
                         'cumulative_num_cases': np.max(self.cum_num_cases) }
        for compartment in self.compartments:
            self.results.update({ 'total_count_'+str(compartment):  int(self.get_count_by_compartment(compartment)),
                                  'peak_count_'+str(compartment):   int(np.max(self.counts[compartment])) })
            self.results.update({ 'total_pct_'+str(compartment):    self.results['total_count_'+str(compartment)]/self.pop_size,
                                  'peak_pct_'+str(compartment):     self.results['peak_count_'+str(compartment)]/self.pop_size})
        for flag in self.allCompartmentFlags.union(self.allNodeFlags):
            self.results.update({ 'total_count_'+str(flag):         int(self.get_count_by_flag(flag)),
                                  'peak_count_'+str(flag):          int(np.max(self.flag_counts[flag])) })
            self.results.update({ 'total_pct_'+str(flag):           self.results['total_count_'+str(flag)]/self.pop_size,
                                  'peak_pct_'+str(flag):            self.results['peak_count_'+str(flag)]/self.pop_size})

    
    ########################################################
    ########################################################


    def process_transition_times(self, transn_dict):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Ensure all transitions have a specified rate or time, as applicable:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        poststates  = list(transn_dict.keys())
        for poststate in poststates:
            if(self.transition_mode=='exponential_rates'):
                if('rate' in transn_dict[poststate]):
                    # Rate(s) provided, simply ensure correct shape:
                    transn_dict[poststate]['rate'] = np.array(transn_dict[poststate]['rate']).reshape((self.pop_size,1))
                elif('time' in transn_dict[poststate]):
                    # Time(s) provided, compute rates as inverse of times:
                    transn_dict[poststate]['rate'] = np.array(1/transn_dict[poststate]['time']).reshape((self.pop_size,1))
            elif(self.transition_mode=='time_in_state'):
                if('time' in transn_dict[poststate]):
                    # Rate(s) provided, simply ensure correct shape:
                    transn_dict[poststate]['time'] = np.array(transn_dict[poststate]['time']).reshape((self.pop_size,1))
                elif('rate' in transn_dict[poststate]):
                    # Rate(s) provided, compute rates as inverse of rates:
                    transn_dict[poststate]['time'] = np.array(1/transn_dict[poststate]['rate']).reshape((self.pop_size,1))
            else:
                raise BaseException("Unrecognized transmission_mode, "+self.transmission_mode+", provided.")


    ########################################################


    def process_transition_probs(self, transn_dict):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Decide the transition each individual will take according to given probabilities:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        poststates  = list(transn_dict.keys())
        probs = []
        for poststate in poststates:
            try:             
                prob = transn_dict[poststate]['prob']
                prob = utils.param_as_array(prob, (self.pop_size,1))
                transn_dict[poststate]['prob'] = prob
                probs.append(prob)
            except KeyError: 
                if(len(poststates) == 1):
                    transn_dict[poststate]['prob'] = np.ones(shape=(self.pop_size,1))
                    probs.append(transn_dict[poststate]['prob']) 
                else:
                    print("Multiple transitions specified, but not all probabilities provided: Assuming equiprobable.")
                    transn_dict[poststate]['prob'] = np.full(1/len(poststates), shape=(self.pop_size,1))
                    probs.append(transn_dict[poststate]['prob']) 
        probs = np.array(probs).reshape((len(poststates), self.pop_size))
        #----------------------------------------
        rands = [poststates[np.random.choice(len(poststates), p=probs[:,i])] for i in range(self.pop_size)]
        #----------------------------------------
        for poststate in transn_dict:
            transn_dict[poststate]["path_taken"] = np.array([1 if rands[i]==poststate else 0 for i in range(self.pop_size)]).reshape((self.pop_size,1))


    ########################################################


    def process_network_transmissibility(self, transm_dict, network_name):
        #----------------------------------------
        # Process local transmissibility parameters for each network:
        #----------------------------------------
        try:
            # Use transmissibility values provided for this network if given,
            # else use transmissibility values provided under generic 'local' key.
            # (If neither of these are provided, defaults to 0 transmissibility.)
            local_transm_vals = transm_dict[network_name] if network_name in transm_dict else transm_dict['local']
            #----------------------------------------
            # Convert omega value(s) to np array if not already np array or sparse matrix:
            local_transm_vals = utils.param_as_array(local_transm_vals, (self.pop_size,1)) if not isinstance(local_transm_vals, (np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)) else local_transm_vals
            #----------------------------------------
            # Generate matrix of pairwise transmissibility values:
            if(local_transm_vals.ndim == 2 and local_transm_vals.shape[0] == self.pop_size and local_transm_vals.shape[1] == self.pop_size):
                net_transm_mat = local_transm_vals
            elif((local_transm_vals.ndim == 1 and local_transm_vals.shape[0] == self.pop_size) or (local_transm_vals.ndim == 2 and (local_transm_vals.shape[0] == self.pop_size or local_transm_vals.shape[1] == self.pop_size))):
                local_transm_vals = local_transm_vals.reshape((self.pop_size,1))
                # Pre-multiply beta values by the adjacency matrix ("transmission weight connections")
                A_beta_pairwise_byInfected = scipy.sparse.csr_matrix.multiply(self.networks[network_name]["adj_matrix"], local_transm_vals.T).tocsr()
                A_beta_pairwise_byInfectee = scipy.sparse.csr_matrix.multiply(self.networks[network_name]["adj_matrix"], local_transm_vals).tocsr()    
                #------------------------------
                # Compute the effective pairwise beta values as a function of the infected/infectee pair:
                if(transm_dict['pairwise_mode'].lower() == 'infected' or transm_dict['pairwise_mode'] is None):
                    net_transm_mat = A_beta_pairwise_byInfected
                elif(transm_dict['pairwise_mode'].lower() == 'infectee'):
                    net_transm_mat = A_beta_pairwise_byInfectee
                elif(transm_dict['pairwise_mode'].lower() == 'min'):
                    net_transm_mat = scipy.sparse.csr_matrix.minimum(A_beta_pairwise_byInfected, A_beta_pairwise_byInfectee)
                elif(transm_dict['pairwise_mode'].lower() == 'max'):
                    net_transm_mat = scipy.sparse.csr_matrix.maximum(A_beta_pairwise_byInfected, A_beta_pairwise_byInfectee)
                elif(transm_dict['pairwise_mode'].lower() == 'mean'):
                    net_transm_mat = (A_beta_pairwise_byInfected + A_beta_pairwise_byInfectee)/2
                else:
                    raise BaseException("Unrecognized pairwise_mode value (support for 'infected', 'infectee', 'min', 'max', and 'mean').")
            else:
                raise BaseException("Invalid data type/shape for transmissibility values.")
            #----------------------------------------
            # Store the pairwise transmissibility matrix in the compartments dict
            transm_dict[network_name] = net_transm_mat
        except KeyError:
            # print("Transmissibility values not given for", network_name, "-- defaulting to 0.")
            transm_dict[network_name] = scipy.sparse.csr_matrix(np.zeros(shape=(self.pop_size, self.pop_size)))


    ########################################################


    def process_network_transm_offsets(self, transm_dict, network_name):
        #----------------------------------------
        # Process frequency-dependent transmission offset factors for each network:
        #----------------------------------------
        if('offsets' not in transm_dict):
            transm_dict['offsets'] = {}
        #----------------------------------------
        try:
            omega_vals = transm_dict['offsets'][network_name]
            #----------------------------------------
            # Convert omega value(s) to np array if not already np array or sparse matrix:
            omega_vals = utils.param_as_array(omega_vals, (self.pop_size,1)) if not isinstance(omega_vals, (np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)) else omega_vals
            #----------------------------------------
            # Store 2d np matrix of pairwise omega values:
            if(omega_vals.ndim == 2 and omega_vals.shape[0] == self.pop_size and omega_vals.shape[1] == self.pop_size):
                transm_dict['offsets'][network_name] = omega_vals
            else:
                raise BaseException("Explicit omega values should be specified as an NxN 2d array. Else leave unspecified and omega values will be automatically calculated according to local_transm_offset_mode.")
        except KeyError:
            #----------------------------------------
            # Automatically generate omega matrix according to local_transm_offset_mode:
            if(transm_dict['local_transm_offset_mode'].lower() == 'pairwise_log'):
                with np.errstate(divide='ignore'): # ignore log(0) warning, then convert log(0) = -inf -> 0.0
                    omega = np.log(np.maximum(self.networks[network_name]["degree"],2))/np.log(np.mean(self.networks[network_name]["degree"])) 
                    omega[np.isneginf(omega)] = 0.0
            elif(transm_dict['local_transm_offset_mode'].lower() == 'pairwise_linear'):
                omega = np.maximum(self.networks[network_name]["degree"],2)/np.mean(self.networks[network_name]["degree"])
            elif(transm_dict['local_transm_offset_mode'].lower() == 'none'):
                omega = np.ones(shape=(self.pop_size, self.pop_size))
            else:
                raise BaseException("Unrecognized local_transm_offset_mode value (support for 'pairwise_log', 'pairwise_linear', and 'none').")
            omega_pairwise_byInfected = scipy.sparse.csr_matrix.multiply(self.networks[network_name]["adj_matrix"], omega.T).tocsr()
            omega_pairwise_byInfectee = scipy.sparse.csr_matrix.multiply(self.networks[network_name]["adj_matrix"], omega).tocsr()
            omega_mat = (omega_pairwise_byInfected + omega_pairwise_byInfectee)/2
            #----------------------------------------
            # Store the pairwise omega matrix in the compartments dict
            transm_dict['offsets'][network_name] = omega_mat


    ########################################################


    def process_initial_states(self):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Determine the iniital counts for each state given their specified initial prevalences
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        initCountTotal = 0
        for c, compartment in enumerate(self.compartments):
            comp_params = self.compartments[compartment]
            #----------------------------------------
            # Set initial counts for each compartment:
            if(comp_params['initial_prevalence'] > 0):
                self.counts[compartment][0] = min( max(int(self.pop_size*comp_params['initial_prevalence']),1), self.pop_size-initCountTotal )
                initCountTotal += self.counts[compartment][0]
            
        #----------------------------------------
        # Initialize remaining counts to the designated default compartment:
        if(initCountTotal < self.pop_size):
            if(self.default_state is not None):
                self.counts[self.default_state][0] = self.pop_size - initCountTotal
            else:
                raise BaseException("A default compartment must be designated ('default_state':True in config) when the total initial count is less than the population size.")

        #----------------------------------------
        # Initialize data series for effective population size (N):
        self.N = np.zeros(self.pop_size*min(len(self.compartments), 10))
        for c, compartment in enumerate(self.compartments):
            if(compartment not in self.excludeFromEffPopSize):
                self.N[0] += self.counts[compartment][0]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize/Reset the states of individuals:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.X = np.concatenate([[self.stateID[comp]]*int(self.counts[comp][0]) for comp in self.compartments]).reshape((self.pop_size,1))
        np.random.shuffle(self.X)

        if(self.store_Xseries):
            self.Xseries        = np.zeros(shape=(6*self.pop_size, self.pop_size), dtype='uint8')
            self.Xseries[0,:]   = self.X.T

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize/Reset transmission chain data:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.lineages = {}
        self.nodeCaseIDs = [None]*self.pop_size
        # Set the lineage IDs of all initially infected nodes to '0' (maybe rethink later)
        for i, initInfectedNode in enumerate(self.get_individuals_by_flag(self.prevalence_flags)):
            if(self.track_case_info):
                self.add_case_to_lineage(initInfectedNode, parent=None)
                self.add_case_log(infectee_node=initInfectedNode, infector_node=None, infection_transition={'from':self.default_state, 'to':self.get_node_compartment(initInfectedNode), 'type':'initialization'})

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Determine the iniital counts for each flag
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(self.track_flag_counts):
            for flag in self.allCompartmentFlags.union(self.allNodeFlags):
                #----------------------------------------
                # Instantiate data series for counts of nodes with each flag:
                self.flag_counts[flag] = np.zeros(self.pop_size*min(len(self.compartments), 10))
                #----------------------------------------
                # Set initial counts for each flag:
                flag_count = len(self.get_individuals_by_flag(flag))
                self.flag_counts[flag][0] = flag_count

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.update_data_series()


    ########################################################
    ########################################################


    def set_state(self, node, state, update_data_series=True):
        # Using this function instead of setting self.X directly ensures that the data series are updated whenever a state changes.
        nodes = utils.treat_as_list(node)
        for i in nodes:
            if(state in self.compartments):
                self.X[i] = self.stateID[state]
                self.state_timer[i] = 0.0
            elif(state in self.stateID):
                self.X[i] = state
                self.state_timer[i] = 0.0
            else:
                print("Unrecognized state, "+str(state)+". No state update performed.")
                return
        if(update_data_series):
            self.update_data_series() 


    ########################################################


    def set_transition_rate(self, compartment, to, rate):
        # Note that it only makes sense to set a rate for temporal transitions.
        compartments = utils.treat_as_list(compartment)
        destStates   = utils.treat_as_list(to)
        for compartment in compartments:
            transn_dict = self.compartments[compartment]['transitions']
            for destState in destStates:
                try:
                    transn_dict[destState]['rate'] = rate
                    transn_dict[destState]['time'] = 1/rate
                except KeyError:
                    transn_dict[destState] = {'rate': rate}
                    transn_dict[destState] = {'time': 1/rate}
            self.process_transition_times(transn_dict)
            # process probs in case a new transition was added above
            self.process_transition_probs(transn_dict) 


    ########################################################


    def set_transition_time(self, compartment, to, time):
        # Note that it only makes sense to set a time for temporal transitions.
        compartments = utils.treat_as_list(compartment)
        destStates   = utils.treat_as_list(to)
        for compartment in compartments:
            transn_dict = self.compartments[compartment]['transitions']
            for destState in destStates:
                try:
                    transn_dict[destState]['time'] = time
                    transn_dict[destState]['rate'] = 1/time
                except KeyError:
                    transn_dict[destState] = {'time': time}
                    transn_dict[destState] = {'rate': 1/time}
            self.process_transition_times(transn_dict)
            # process probs in case a new transition was added above
            self.process_transition_probs(transn_dict) 


    ########################################################


    def set_transition_probability(self, compartment, probs_dict, upon_exposure_to=None):
        compartments     = utils.treat_as_list(compartment)
        infectiousStates = utils.treat_as_list(upon_exposure_to)
        for compartment in compartments:
            if(upon_exposure_to is None):
                transn_dict = self.compartments[compartment]['transitions']
                for destState in probs_dict:    
                    try:
                        transn_dict[destState]['prob'] = probs_dict[destState]
                    except KeyError:
                        # print("Compartment", compartment, "has no specified transition to", destState, "for which to set a probability.")
                        transn_dict[destState] = {'prob': probs_dict[destState]}
            else:
                for infectiousState in infectiousStates:
                    transn_dict = self.compartments[compartment]['susceptibilities'][infectiousState]['transitions']
                    for destState in probs_dict:    
                        try:
                            transn_dict[destState]['prob'] = probs_dict[destState]
                        except KeyError:
                            # print("Compartment", compartment, "has no specified transition to", destState, "for which to set a probability.")
                            transn_dict[destState] = {'prob': probs_dict[destState]}
            self.process_transition_probs(transn_dict)


    ########################################################


    def set_susceptibility(self, compartment, to, susceptibility):
        compartments     = utils.treat_as_list(compartment)
        infectiousStates = utils.treat_as_list(to)
        susceptibility   = utils.param_as_array(susceptibility, (self.pop_size,1)) # np.array(susceptibility).reshape((self.pop_size,1))
        for compartment in compartments:
            for infectiousState in infectiousStates:
                susc_dict = self.compartments[compartment]['susceptibilities']
                try:
                    susc_dict[infectiousState]['susceptibility'] = copy.deepcopy(susceptibility)
                except KeyError:
                    susc_dict[infectiousState] = {'susceptibility': copy.deepcopy(susceptibility)}


    ########################################################


    def set_transmissibility(self, compartment, transm_context, transmissibility):
        compartments = utils.treat_as_list(compartment)
        contexts     = utils.treat_as_list(transm_context)
        for compartment in compartments:
            transm_dict = self.compartments[compartment]['transmissibilities']
            for context in contexts:
                #----------------------------------------
                # Handle update to local transmissibility over the named network (or generic 'local'):
                if(context in self.networks or context=='local'):
                    # Update the transmission subdictionary for this compartment with the new values:
                    if(context in self.networks):
                        # Transmissibility is being set for a particular network context,
                        # update the transmissibility data structures for that network.
                        transm_dict[context] = copy.deepcopy(transmissibility)
                        self.process_network_transmissibility(transm_dict, context)
                        # Re-calculate Infectivity Matrices for updated compartments/networks
                        self.infectivity_mat[compartment][context] = scipy.sparse.csr_matrix.multiply(transm_dict[context], transm_dict['offsets'][context])
                    elif(context == 'local'):
                        # Transmissibility is being set for the generic 'local' context:
                        # Use these values and update transmissibility data structures for *all* networks.
                        transm_dict['local'] = copy.deepcopy(transmissibility)
                        for network in self.networks:
                            transm_dict[network] = copy.deepcopy(transmissibility)
                            self.process_network_transmissibility(transm_dict, network)
                            # Re-calculate Infectivity Matrices for updated compartments/networks
                            self.infectivity_mat[compartment][network] = scipy.sparse.csr_matrix.multiply(transm_dict[network], transm_dict['offsets'][network])
                    else:
                        print("Unexpected transm_context if-else case encountered in CompartmentNetworkBuilder.set_transmissibility()")
                #----------------------------------------
                # Handle update to exogenous transmissibility:
                elif(context=='exogenous'):
                    transm_dict['exogenous'] = transmissibility
                    # self.exogenous_prevalence[compartment] = transm_dict['exogenous']
                #----------------------------------------
                else:
                    print("Transmission mode,", transm_context, "not recognized (expected 'exogenous', 'local', or network name in "+str(list(self.networks.keys()))+"); no update.")
            #----------------------------------------
            # Re-calculate global transmissibility as the mean of local transmissibilities.
            #---------------------------------------
            transm_dict['global'] = np.sum([np.sum(transm_dict[network][transm_dict[network]!=0]) for network in self.networks]) / max(np.sum([transm_dict[network].count_nonzero() for network in self.networks]), 1)


    ########################################################


    def set_initial_prevalence(self, compartment, prevalence):
        compartments   = utils.treat_as_list(compartment)
        for compartment in compartments:
            self.compartments[compartment]['initial_prevalence'] = prevalence
        self.process_initial_states()


    ########################################################

    
    def set_exogenous_prevalence(self, compartment, prevalence):
        compartments   = utils.treat_as_list(compartment)
        for compartment in compartments:
            # Update the compartment model definition dictionary
            self.compartments[compartment]['exogenous_prevalence'] = prevalence
            # Update the exogenous prevalence variable in the model object
            # self.exogenous_prevalence[compartment] = prevalence


    ########################################################

    
    def set_default_state(self, compartment):
        # Must be a single compartment given
        for c in self.compartments:
            self.compartments[c]['default_state'] = (c == compartment)
            if(c == compartment):
                self.default_state = c
       

    ########################################################

    
    def set_exclude_from_eff_pop(self, compartment, exclude=True):
        compartments = utils.treat_as_list(compartment)
        for compartment in compartments:
            self.compartments[compartment]['exclude_from_eff_pop'] = exclude
            if(exclude and not compartment in self.excludeFromEffPopSize):
                self.excludeFromEffPopSize.append(compartment)
            elif(not exclude and compartment in self.excludeFromEffPopSize):
                self.excludeFromEffPopSize = [c for c in self.excludeFromEffPopSize if c!=compartment] # remove all occurrences of compartment


    ########################################################


    def set_isolation(self, node, isolation):
        nodes = utils.treat_as_list(node)
        for node in nodes:
            if(isolation == True):
                # self.calc_infectious_time(node) <- TODO handle this?
                self.isolation[node] = 1
            elif(isolation==False):
                self.isolation[node] = 0
            # Reset the isolation timer:
            self.isolation_timer[node] = 0


    ########################################################


    def set_network_activity(self, network, node='all', active=None, active_isolation=None):
        nodes = list(range(self.pop_size)) if (isinstance(node, str) and node=='all') else utils.treat_as_list(node)
        networks = utils.treat_as_list(network)
        for i in nodes:
            for G in networks:
                if(active is not None):
                    self.networks[G]['active'][i] = 1 if active else 0
                if(active_isolation is not None):
                    self.networks[G]['active_isolation'][i] = 1 if active_isolation else 0


    ########################################################


    def set_node_attribute(self, node, attribute_name, attribute_value):
        nodes = list(range(self.pop_size)) if (isinstance(node, str) and node=='all') else utils.treat_as_list(node)
        avals = utils.treat_as_list(attribute_value)
        if(attribute_name not in self.node_attributes):
            self.node_attributes[attribute_name] = np.full(shape=self.pop_size, fill_value=None)
        for i, node in enumerate(nodes):
            self.node_attributes[attribute_name][node] = avals[i]

        # self.node_attributes[attribute_name] = utils.param_as_array(attribute_values, (1, self.pop_size)).flatten()


    ########################################################


    def get_node_compartment(self, node):
        node_list_provided = isinstance(node, (list, np.ndarray))
        nodes = list(range(self.pop_size)) if node=='all' else utils.treat_as_list(node)
        compartments = []
        for node in nodes:
            stateID     = self.X[node][0]
            compartments.append( list(self.stateID.keys())[list(self.stateID.values()).index(stateID)] )
        return compartments if node_list_provided else compartments[0] if len(compartments)>0 else None


    ########################################################


    def get_individuals_by_compartment(self, compartment, combine='any'):
        compartments = utils.treat_as_list(compartment)
        compartment_individuals_sets = []
        for compartment in compartments:
            compartment_individuals = set( np.where(np.in1d(self.X, [self.stateID[c] for c in compartments]))[0] )
            compartment_individuals_sets.append(compartment_individuals)
        if(combine=='any'):
            return list( set().union(*compartment_individuals_sets) )
        elif(combine=='all'): 
            return list( set(range(self.pop_size)).intersection(*compartment_individuals_sets) )
        else:
            print("Combine mode", combine, "not recognized: support for 'any' (default) or 'all'.")
            exit()


    ########################################################


    def get_count_by_compartment(self, compartment):
        compartments = utils.treat_as_list(compartment)
        compartment_counts_ = {}
        for compartment in compartments:
            compartment_counts_[compartment] = self.counts[compartment][self.tidx]
        return compartment_counts_ if len(compartment_counts_)>1 else np.sum([compartment_counts_[c] for c in compartment_counts_]) if len(compartment_counts_)>0 else None


    ########################################################


    def get_compartment_by_state_id(self, state_id):
        return next((comp_name for comp_name, id_num in self.stateID.items() if id_num==state_id), None)


    ########################################################
    ########################################################


    def add_compartment_flag(self, compartment, flag):
        compartments = list(range(self.pop_size)) if compartment=='all' else utils.treat_as_list(compartment)
        flags = utils.treat_as_list(flag)
        for compartment in compartments:
            for flag in flags:
                self.compartments[compartment]['flags'].append(flag)
                self.allCompartmentFlags.add(flag)
                if(self.track_flag_counts and flag not in self.flag_counts):
                    self.flag_counts[flag] = np.zeros_like(self.counts[compartment])
                    self.update_data_series()


    def remove_compartment_flag(self, compartment, flag):
        compartments = list(range(self.pop_size)) if compartment=='all' else utils.treat_as_list(compartment)
        flags = utils.treat_as_list(flag)
        for compartment in compartments:
            for flag in flags:
                self.compartments[compartment]['flags'] = [f for f in self.compartments[compartment]['flags'] if f!=flag] # remove all occurrences of flag


    ########################################################


    def add_individual_flag(self, node, flag):
        nodes = list(range(self.pop_size)) if node=='all' else utils.treat_as_list(node)
        flags = utils.treat_as_list(flag)
        for node in nodes:
            for flag in flags:
                self.node_flags[node].append(flag)
                self.allNodeFlags.add(flag)
                if(self.track_flag_counts and flag not in self.flag_counts):
                    self.flag_counts[flag] = np.zeros_like(self.counts[list(self.counts.keys())[0]])
                    self.update_data_series()


    ########################################################


    def remove_individual_flag(self, node, flag):
        nodes = list(range(self.pop_size)) if node=='all' else utils.treat_as_list(node)
        flags = utils.treat_as_list(flag)
        for node in nodes:
            for flag in flags:
                self.node_flags[node] = [f for f in self.node_flags[node] if f!=flag] # remove all occurrences of flag


    ########################################################


    def get_compartments_by_flag(self, flag, has_flag=True):
        flags = utils.treat_as_list(flag)
        flagged_compartments = set()
        for compartment, comp_dict in self.compartments.items():
            if(any([flag in comp_dict['flags'] for flag in flags]) == has_flag):
                flagged_compartments.add(compartment)
        return list(flagged_compartments)


    ########################################################


    def get_individuals_by_flag(self, flag, has_flag=True, combine='any'):
        flags = utils.treat_as_list(flag)
        flagged_individuals_sets = []
        for flag in flags:
            flagged_individuals = set()
            if(len(self.allCompartmentFlags) > 0):
                flagged_individuals.update( np.where(np.in1d(self.X, [self.stateID[c] for c in self.get_compartments_by_flag(flag)]))[0] )
            if(len(self.allNodeFlags) > 0):
                flagged_individuals.update( [i for i in range(self.pop_size) if flag in self.node_flags[i]] )
            flagged_individuals_sets.append(flagged_individuals)
        if((combine=='any' and has_flag) or (combine=='all' and not has_flag)):
            return list( set().union(*flagged_individuals_sets) )
        else: 
            return list( set(range(self.pop_size)).intersection(*flagged_individuals_sets) )
        

    ########################################################


    def get_count_by_flag(self, flag, has_flag=True, combine='any'):
        return len(self.get_individuals_by_flag(flag, has_flag, combine))


    ########################################################
    ########################################################


    def introduce_random_exposures(self, num, compartment='all', exposed_to='any'):
        num = int(num)
        compartments      = list(self.compartments.keys()) if compartment=='all' else utils.treat_as_list(compartment)
        infectiousStates  = list(self.compartments.keys()) if exposed_to=='any'  else utils.treat_as_list(exposed_to)
        exposure_susceptibilities = []
        exposedNodes = []
        for exposure in range(num):
            for compartment in compartments:
                for infectiousState in infectiousStates:
                    if(infectiousState in self.compartments[compartment]['susceptibilities']):
                        exposure_susceptibilities.append({'susc_state': compartment, 
                                                          'inf_state': infectiousState, 
                                                          'susceptibilities': self.compartments[compartment]['susceptibilities'][infectiousState]['susceptibility'].flatten(),
                                                          'mean_susceptibility': np.mean(self.compartments[compartment]['susceptibilities'][infectiousState]['susceptibility']),
                                                          'susc_state_prevalence': self.get_count_by_compartment(compartment),
                                                          })
            exposureType   = np.random.choice(exposure_susceptibilities, p=[d['mean_susceptibility']*d['susc_state_prevalence'] for d in exposure_susceptibilities]/np.sum([d['mean_susceptibility']*d['susc_state_prevalence'] for d in exposure_susceptibilities]))
            exposableNodes = [i for i in range(self.pop_size) if self.X[i]==self.stateID[exposureType['susc_state']]]
            if(len(exposableNodes) > 0):
                exposedNode    = np.random.choice(exposableNodes, p=exposureType['susceptibilities'][exposableNodes]/np.sum(exposureType['susceptibilities'][exposableNodes]))
                exposedNodes.append(exposedNode)
                #--------------------
                exposureTransitions = self.compartments[exposureType['susc_state']]['susceptibilities'][exposureType['inf_state']]['transitions']
                exposureTransitionsActiveStatuses = [exposureTransitions[dest]['path_taken'].flatten()[exposedNode] for dest in exposureTransitions]
                destState = np.random.choice(list(exposureTransitions.keys()), p=exposureTransitionsActiveStatuses/np.sum(exposureTransitionsActiveStatuses))
                #--------------------
                if(self.track_case_info):
                    self.add_case_to_lineage(exposedNode, parent=None)
                    self.add_case_log(infectee_node=exposedNode, infector_node=None, infection_transition={'from':self.get_node_compartment(exposedNode), 'to':destState, 'type':'introduction'})
                #--------------------
                self.set_state(exposedNode, destState, update_data_series=False) # too slow to update data series after every node state update, will updata data series after loop
            self.update_data_series()
        return exposedNodes


    ########################################################
    ########################################################


    def add_case_to_lineage(self, new_case_node, parent=None):
        # Exogenous or otherwise unsourced case:
        if(parent is None):
            # Add case as a new top-level lineage root:
            newCaseID = str( len(self.lineages.keys())+1 )
            self.lineages.update({ newCaseID:{} })
            self.nodeCaseIDs[new_case_node] = newCaseID
        # Endogenous transmission case with given parent:
        else:
            parentCaseID = self.nodeCaseIDs[parent]
            if('.' in parentCaseID):
                parentCaseID_parts = parentCaseID.split('.')
                lineageSubtree = self.lineages
                for l in range(1, len(parentCaseID_parts)+1):
                    lineageSubtree = lineageSubtree['.'.join(parentCaseID_parts[:l])]
            else:
                lineageSubtree = self.lineages[parentCaseID]
            newCaseID = parentCaseID +'.'+ str(len(lineageSubtree.keys())+1)
            lineageSubtree[newCaseID] = {}
            self.nodeCaseIDs[new_case_node] = newCaseID
        return newCaseID


    ########################################################


    def process_new_case(self, infectee_node, infection_transition):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Identify the infector node probabilistically based on 
        # the relative propensities of transmission to the 
        # infected individual from all other individuals:
        #----------------------------------------
        infectivityWts_exogenous = 0
        infectivityWts_global    = np.zeros(self.pop_size)
        infectivityWts_local     = np.zeros(self.pop_size)
        # Consider the infectious states that a) the infected node is susceptible to 
        # and b) can trigger the current infection transition upon exposure:
        for infectiousState, susc_params in self.compartments[infection_transition['from']]['susceptibilities'].items():                
            if(infection_transition['to'] in susc_params['transitions'] and susc_params['transitions'][infection_transition['to']]['prob'][infectee_node]>0):
                bool_inInfectiousState = (self.X==self.stateID[infectiousState]).flatten()
                bin_inInfectiousState  = [1 if i else 0 for i in bool_inInfectiousState]
                j_inInfectiousState    = np.argwhere(bool_inInfectiousState).flatten()
                #----------------------------------------
                # Calulate Exogenous transmission propensity weights:
                # print("exogenous_prevalence", self.compartments[infectiousState]['exogenous_prevalence'])
                # print("exogenous_trasms", self.compartments[infectiousState]['transmissibilities']['exogenous'])
                infectivityWts_exogenous += self.openness *  self.compartments[infectiousState]['transmissibilities']['exogenous'] * self.compartments[infectiousState]['exogenous_prevalence']
                #----------------------------------------
                # Calulate Global transmission propensity weights:
                infectivityWts_global[j_inInfectiousState] += ((1-self.openness) * ((self.mixedness) * ((self.compartments[infectiousState]['transmissibilities']['global']*self.counts[infectiousState][self.tidx])/self.N[self.tidx])))
                #----------------------------------------
                # Calulate Global transmission propensity weights:
                bool_inInfectiousState = (self.X==self.stateID[infectiousState]).flatten()
                bin_inInfectiousState  = [1 if i else 0 for i in bool_inInfectiousState]
                if(np.any(bool_inInfectiousState)):
                    for network, G in self.networks.items():
                        # print(network)
                        bool_isGactive = (((G['active']!=0)&(self.isolation==0)) | ((G['active_isolation']!=0)&(self.isolation!=0))).flatten()
                        bin_isGactive  = [1 if i else 0 for i in bool_isGactive]
                        if(any(bool_isGactive) and any(bool_inInfectiousState)):
                            localInfectivity = self.infectivity_mat[infectiousState][network][infectee_node,:]
                            if(localInfectivity.sum() > 0):
                                localInfectivity_ofActiveInfectious = localInfectivity.toarray().flatten() * bin_isGactive * bin_inInfectiousState
                                # print("localInfectivity_ofActiveInfectious", np.argwhere(localInfectivity_ofActiveInfectious).flatten())
                                infectivityWts_local += (1-self.openness) * ((1-self.mixedness) * (localInfectivity_ofActiveInfectious/self.active_degree[infectee_node]))
        #----------------------------------------
        # Select the infector probabilisitically, 
        # proportional to total propensity of transmission to infectee:
        #----------------------------------------
        # Combine propensity weights for each transmission modality into a probability vector:
        possibleInfectorProbs = list(infectivityWts_global + infectivityWts_local)
        possibleInfectorProbs.append(infectivityWts_exogenous)
        possibleInfectorProbs = possibleInfectorProbs/np.sum(possibleInfectorProbs)
        #--------------------
        possibleInfectors = list(range(self.pop_size))
        possibleInfectors.append('exogenous')
        #--------------------
        # Randomly draw the infector node:
        infector_node = int( np.random.choice(possibleInfectors, p=possibleInfectorProbs) )
        #----------------------------------------
        # Add the infected node as a new case on the infector's lineage:
        #----------------------------------------
        self.add_case_to_lineage(infectee_node, parent=(infector_node if infector_node!='exogenous' else None))
        #----------------------------------------
        # Add the infectee/infector information to the case logs:
        #----------------------------------------
        self.add_case_log(infectee_node, infector_node, infection_transition)

    
    ########################################################


    def add_case_log(self, infectee_node, infector_node=None, infection_transition=None):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Record data about the transmission event
        # and the infector/infectee nodes involved:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #----------------------------------------
        # Log basic case and state information:
        caseLog = { 
            'transmission_num':             len(self.caseLogs)+1,             
            'transmission_time':            self.t,
            'transmission_type':            infection_transition['type'],
            # Infectee info:
            'infectee_individual':          infectee_node,
            'infectee_case_id':             self.nodeCaseIDs[infectee_node],
            'infectee_lineage':             self.nodeCaseIDs[infectee_node].split('.')[0],
            'infectee_lineage_depth':       len(self.nodeCaseIDs[infectee_node].split('.')),
            'infectee_flags':               '|'.join(self.node_flags[infectee_node]),
            'infectee_state_preinfection':  infection_transition['from'],
            'infectee_state_postinfection': infection_transition['to'],
            'infectee_isolation_status':    False if self.isolation[infectee_node]==0 else True,
            'infectee_susceptibility':      self.compartments[infection_transition['from']]['susceptibilities'][self.get_node_compartment(infector_node)]['susceptibility'][infectee_node][0]
                                                if infector_node is not None
                                                else np.mean([ self.compartments[infection_transition['from']]['susceptibilities'][c]['susceptibility'] for c in self.compartments[infection_transition['from']]['susceptibilities'].keys() ]),
            # Infector info:
            'infector_individual':          infector_node if infector_node is not None else None,
            'infector_case_id':             self.nodeCaseIDs[infector_node] if infector_node is not None else None,
            'infector_lineage':             self.nodeCaseIDs[infector_node].split('.')[0] if infector_node is not None else None,
            'infector_lineage_depth':       len(self.nodeCaseIDs[infector_node].split('.')) if infector_node is not None else None,
            'infector_flags':               '|'.join(self.node_flags[infector_node]) if infector_node is not None else None,
            'infector_state':               self.get_node_compartment(infector_node) if infector_node is not None else None,
            'infector_time_in_state':       self.state_timer[infector_node][0] if infector_node is not None else None,
            'infector_isolation_status':    (False if self.isolation[infector_node]==0 else True) if infector_node is not None else None,
            }
        #----------------------------------------
        # Log network-related information:
        infectee_contacts_overall = set()
        infector_contacts_overall = set()
        bin_isGactive_overall     = np.zeros(self.pop_size)
        for network, G in self.networks.items():
            infectee_contacts = list(G['networkx'].neighbors(infectee_node))
            infectee_contacts_overall.update(infectee_contacts)
            infector_contacts = list(G['networkx'].neighbors(infector_node)) if infector_node is not None else []
            infector_contacts_overall.update(infector_contacts)
            bool_isGactive = (((G['active']!=0)&(self.isolation==0)) | ((G['active_isolation']!=0)&(self.isolation!=0))).flatten()
            bin_isGactive  = [1 if i else 0 for i in bool_isGactive]
            bin_isGactive_overall += bin_isGactive
            caseLog.update({
                'infectee_total_degree_'+network:       len(infectee_contacts),  
                'infectee_active_degree_'+network:      np.count_nonzero(np.array(bin_isGactive)[infectee_contacts]),
                'infector_total_degree_'+network:       len(infector_contacts),
                'infector_active_degree_'+network:      np.count_nonzero(np.array(bin_isGactive)[infector_contacts]),
                'infector_is_contact_'+network:         (infector_node in infectee_contacts) if infector_node is not None else False,
                'infector_transmissibility_'+network:   np.mean(self.infectivity_mat[self.get_node_compartment(infector_node)]['workplace'][:,infector_node][self.infectivity_mat[self.get_node_compartment(infector_node)]['workplace'][:,infector_node] > 0]) if infector_node is not None else None
                })
        caseLog.update({
            'infectee_total_degree_overall':    len(infectee_contacts_overall),
            'infectee_active_degree_overall':   np.count_nonzero(np.array(bin_isGactive_overall)[list(infectee_contacts_overall)]),
            'infector_total_degree_overall':    len(infector_contacts_overall),
            'infector_active_degree_overall':   np.count_nonzero(np.array(bin_isGactive_overall)[list(infector_contacts_overall)]),
            'infector_is_contact_overall':      (infector_node in infectee_contacts_overall) if infector_node is not None else False
            })
        #----------------------------------------
        # Log node attribute information:
        for attribute_name, attribute_values in self.node_attributes.items():
            caseLog.update({
                'infectee_'+attribute_name: attribute_values[infectee_node],
                'infector_'+attribute_name: attribute_values[infector_node] if(infector_node is not None) else None
                })
        #----------------------------------------
        self.caseLogs.append(caseLog)
        # print(self.caseLogs)
        # exit()


    ########################################################


    def get_results_dataframe(self):
        import pandas as pd
        return pd.DataFrame([self.results])


    ########################################################


    def get_case_log_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.caseLogs)


    ########################################################
    ########################################################


    def update_test_parameters(self, new_test_params='default', prevalence_flags=None):
        if(isinstance(new_test_params, str) and '.json' in new_test_params):
            new_test_params = utils.load_config(new_test_params)
            # with open(new_test_params) as new_test_params_file:
            #     new_test_params = json.load(new_test_params_file)
        elif(isinstance(new_test_params, dict)):
            pass
        elif(new_test_params == 'default'):
            # If no test params are given, default to a test that is 100% sensitive/specific to all compartments with the prevalence flag(s) given by arg or model attribute:
            new_test_params = {}
            infectedCompartments = self.get_compartments_by_flag(prevalence_flags) if prevalence_flags is not None else self.prevalence_flags
            for compartment in self.compartments:
                new_test_params.update({compartment: {"default_test": {"sensitivity": 1.0 if compartment in infectedCompartments else 0.0, "specificity": 1.0}}})
        elif(new_test_params is None):
            return
        else:
            raise BaseException("Specify test parameters with a dictionary or JSON file.")
        #----------------------------------------
        test_types = set()
        for compartment, comp_params in new_test_params.items():
            for test_type, testtype_params in comp_params.items():
                test_types.add(test_type)
                # Process sensitivity values for the current compartment and test type:
                try: # convert sensitivity(s) provided to a list of values (will be interpreted as time course) 
                    testtype_params['sensitivity'] = [testtype_params['sensitivity']] if not (isinstance(testtype_params['sensitivity'], (list, np.ndarray))) else testtype_params['sensitivity']
                except KeyError:
                    testtype_params['sensitivity'] = [0.0]
                # Process sensitivity values for the current compartment and test type:
                try: # convert sensitivity(s) provided to a list of values (will be interpreted as time course) 
                    testtype_params['specificity'] = [testtype_params['specificity']] if not (isinstance(testtype_params['specificity'], (list, np.ndarray))) else testtype_params['specificity']
                except KeyError:
                    testtype_params['specificity'] = [0.0]
        self.test_params.update(new_test_params)
        self.test_types.update(test_types)
        # return self.test_params, self.test_types
        

    ########################################################


    def test(self, node, test_type):
        node_list_provided = isinstance(node, (list, np.ndarray))
        nodes = list(range(self.pop_size)) if node=='all' else [node] if not node_list_provided else node
        results  = []
        trueness = []
        for node in nodes:
            # print("tttttttttt")
            node_compartment = self.get_node_compartment(node)
            node_daysInCompartment = int(self.state_timer[node])
            # print("Test node", node, "in state", node_compartment, "day", node_daysInCompartment)
            #----------------------------------------
            # Perform the test on the selected individuals:
            #----------------------------------------
            sensitivities_timeCourse = self.test_params[node_compartment][test_type]['sensitivity']
            specificities_timeCourse = self.test_params[node_compartment][test_type]['specificity']
            sensitivity = sensitivities_timeCourse[node_daysInCompartment if node_daysInCompartment<len(sensitivities_timeCourse) else -1]
            specificity = specificities_timeCourse[node_daysInCompartment if node_daysInCompartment<len(specificities_timeCourse) else -1]
            if(sensitivity > 0.0): # individual is in a state where the test can return a true positive
                result_positive = (np.random.rand() < sensitivity)
                result_trueness = (result_positive==True) # result should be positive if sensitivity > 0
                # print(sensitivity, "> 0.0:")
            elif(specificity < 1.0): # individual is in a state where the test can return a false positive
                result_positive = (np.random.rand() > specificity)
                result_trueness = (result_positive==False) # result should be negative if sensitivity = 0
                # print(sensitivity, "= 0.0,", specificity, "< 1.0")
            else:
                result_positive = False
                result_trueness = (result_positive==False) # result should be negative if sensitivity = 0
                # print(sensitivity, "= 0.0,", specificity, "= 1.0")
            results.append(result_positive)
            trueness.append(result_trueness)
            # print("\t", node_compartment, ('*' if not any(substr in node_compartment for substr in ['S', 'R']) else ' '), result_positive, result_trueness, sensitivity, specificity)
        return (results, trueness) if node_list_provided else (results[0], trueness[0]) if len(results)>0 else (None, None)


    ########################################################
    

    def add_vaccine(self, name, susc_effectiveness, transm_effectiveness, series=None, compartment_name_suffix='v', flag_vaccinated=True, flag_series=True, flag_name=True):
        suffix_found   = False
        suffix_attempt = 0
        suffix = compartment_name_suffix
        while(not suffix_found):
            if(not any(suffix in comp_name for comp_name in self.compartments.keys())):
                suffix_found = True
            else:
                suffix_attempt += 1
                suffix = str(compartment_name_suffix)+str(suffix_attempt+1)
        # print(suffix)
        #----------------------------------------
        # Add this vaccine to the given vaccine series, 
        # creating a new vaccine series if an existing one is not provided:
        #----------------------------------------
        if(series in self.vaccine_series):
            self.vaccine_series[series].append( {'vaccine_name': name, 'susc_effectiveness': susc_effectiveness, 'transm_effectiveness': transm_effectiveness} )
        else:
            series = series if series is not None else name
            self.vaccine_series[series] = [{'vaccine_name': name, 'susc_effectiveness': susc_effectiveness, 'transm_effectiveness': transm_effectiveness} ]
        numberInSeries = len(self.vaccine_series[series])
        #----------------------------------------
        # Create new compartments for vaccinated states:
        #----------------------------------------
        vax_compartments = {}
        for comp_name, comp_dict in self.compartments.items():
            if('vaccinated' not in comp_dict or comp_dict['vaccinated']==False):
                # This is an unvaccinated state. Create a new, vaccinated version of it.
                vaxxed_comp_name = comp_name+suffix
                # Mark/Flag this compartment as unvaccinated for future reference:
                comp_dict['vaccinated'] = False
                if(flag_vaccinated and 'unvaccinated' not in comp_dict['flags']):
                    self.add_compartment_flag(comp_name, 'unvaccinated')
                # Store information about the vaccine series and related vaccination transition if not already present:
                if('vaccine_series' not in comp_dict):
                    comp_dict['vaccine_series'] = {series: {'vaccine_name': None, 'num_in_series': 0, 'unvaccinated_state': comp_name, 'vaccination_transition': vaxxed_comp_name}}
                # Add the vaccinated versions of infectious states to the unvaccinated compartment's list of susceptibilities:
                new_susceptibilities = {}
                for infectiousState, susc_params in comp_dict['susceptibilities'].items():
                    if('vaccinated' not in self.compartments[infectiousState] or self.compartments[infectiousState]['vaccinated']==False):
                        new_susceptibilities.update({infectiousState+suffix: copy.deepcopy(susc_params)})
                comp_dict['susceptibilities'].update(new_susceptibilities)
                # Copy the compartment params dict of the unvaccinated compartment to be updated for the vaccinated version below:
                vaxxed_comp_dict = copy.deepcopy(comp_dict)
                # Mark/Flag the new compartment as being vaccinated:
                vaxxed_comp_dict['vaccinated'] = True 
                vaxxed_comp_dict['flags'] = [flag for flag in vaxxed_comp_dict['flags'] if flag != 'unvaccinated']
                if(flag_vaccinated): vaxxed_comp_dict['flags'].append('vaccinated')
                if(flag_series):     vaxxed_comp_dict['flags'].append(series)
                if(flag_name):       vaxxed_comp_dict['flags'].append(name)
                # Update the vaccine series / transition info for the new vaccinated compartment 
                vaxxed_comp_dict['vaccine_series'][series] = {'vaccine_name': name, 'num_in_series': numberInSeries, 'unvaccinated_state': comp_name, 'vaccination_transition': None}
                # Make sure the new compartment doesn't inherit a default state flag:
                vaxxed_comp_dict['default_state']  = False 
                # Update susceptibility values and susceptibility-related transmission transitions for the vaccinated state:
                for infectiousState, susc_dict in vaxxed_comp_dict['susceptibilities'].items():
                    # Calculate susceptibilities for the vaccinated compartment based on the vaccine's effectiveness for susceptibility:
                    susc_dict['susceptibility'] *= 1 - susc_effectiveness
                    # Convert the transmission transition destination states to the vaccinated versions of those states:
                    new_susc_transitions = copy.deepcopy(susc_dict['transitions'])
                    for destState, transition_params in susc_dict['transitions'].items():
                        if('vaccinated' not in self.compartments[destState] or self.compartments[destState]['vaccinated']==False):
                            new_susc_transitions[destState+suffix] = new_susc_transitions.pop(destState)
                    susc_dict['transitions'] = new_susc_transitions
                # Calculate transmissibilities for the vaccinated compartment based on the vaccine's effectiveness for transmissibility:
                for transm_context, transm_vals in comp_dict['transmissibilities'].items():
                    if(transm_context in ['pairwise_mode', 'infected', 'local_transm_offset_mode', 'offsets']):
                        pass
                    else:
                        vaxxed_comp_dict['transmissibilities'][transm_context] *= 1 - transm_effectiveness
                # Convert the temporal transition destination states to the vaccinated versions of those states:
                new_transitions = copy.deepcopy(vaxxed_comp_dict['transitions'])
                for destState, transition_params in vaxxed_comp_dict['transitions'].items():
                    if('vaccinated' not in self.compartments[destState] or self.compartments[destState]['vaccinated']==False):
                        new_transitions[destState+suffix] = new_transitions.pop(destState)
                vaxxed_comp_dict['transitions'] = new_transitions
                # Add the parameterized vaccinated compartment to a dict of new compartments:
                vax_compartments.update({vaxxed_comp_name: vaxxed_comp_dict})
                # Copy the test parameters for the new vaccinated state from the unvaccinated state:
                if(comp_name in self.test_params):
                    self.test_params.update({vaxxed_comp_name: copy.deepcopy(self.test_params[comp_name])})
            #-----
            elif(comp_dict['vaccine_series'][series]['num_in_series'] == numberInSeries-1):
                # This comp_dict is for a state that is vaccinated with the vaccine that comes before the currently-being-added vaccine in the series:
                # Update the vaccine transition info for the prev vaccinated compartment such that it transitions to the new vaccinated compartment:
                print(comp_name, numberInSeries, "-1", numberInSeries-1, comp_dict['vaccine_series'][series])
                comp_dict['vaccine_series'][series]['vaccination_transition'] = comp_dict['vaccine_series'][series]['unvaccinated_state']+suffix
                print(comp_name, numberInSeries, "-1", numberInSeries-1, comp_dict['vaccine_series'][series])
            #-----
            else:
                # This is another vaccinated state. Only create new versions of unvaccinated states.
                pass
        #----------------------------------------
        self.update_compartments(vax_compartments)
        self.update_data_series()


    ########################################################


    def vaccinate(self, node, vaccine_series=None):
        nodes = list(range(self.pop_size)) if isinstance(node, str) and node=='all' else utils.treat_as_list(node)
        if(len(self.vaccine_series)==0):
            print("No vaccines have been added to the model, so the effects of vaccination are unspecified.")
            exit()
        elif(vaccine_series==None):
            if(len(self.vaccine_series)==1):
                vaccine_series = list(self.vaccine_series)[0]
            else:
                print("More than one vaccine series have been added to the model, so the series must be specified when calling vaccinate().")
                exit()
        elif(vaccine_series not in self.vaccine_series):
            print("The vaccine series specified when calling vaccinate() is not recognized (i.e., has not been previously added to the model).")
            exit()
        for node in nodes:
            # print("vaccinating node", node)
            vaxxedDestState = self.compartments[self.get_compartment_by_state_id(self.X[node])]['vaccine_series'][vaccine_series]['vaccination_transition']
            self.set_state(node, vaxxedDestState, update_data_series=False) # too slow to update data series after every node state update, will updata data series after loop
        self.update_data_series()










