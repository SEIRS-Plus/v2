"""
Custom compartment models with contact networks
"""
# Standard Libraries
import json

# External Libraries
import networkx as networkx
import numpy as np
import scipy as scipy
import scipy.integrate

# Internal Libraries
from seirsplus.models.compartment_model_builder import CompartmentModelBuilder


class CompartmentNetworkModel():

    def __init__(self, 
                    compartments, 
                    networks,
                    mixedness=0.0, 
                    openness=0.0,
                    isolation_period=None,
                    transition_mode='exponential_rates', 
                    local_trans_denom_mode='all_contacts',
                    log_infection_info=False,
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
        self.log_infection_info     = log_infection_info

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the contact networks specifications:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.pop_size = None # will be updated in update_networks()
        self.networks = {}
        self.update_networks(networks)

        self.mixedness = mixedness
        self.openness  = openness

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
        self.isolation          = np.zeros(self.pop_size)
        self.isolation_period   = isolation_period
        self.isolation_timer    = np.zeros(self.pop_size)
        self.totalIsolationTime = np.zeros(self.pop_size)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize compartment IDs/metadata:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.stateID               = {}
        self.default_state         = list(self.compartments.keys())[0] # default to first compartment specified
        self.excludeFromEffPopSize = []
        self.node_flags            = [[]]*self.pop_size
        self.allNodeFlags          = set() 
        self.allCompartmentFlags   = set()
        for c, compartment in enumerate(self.compartments):
            comp_params = self.compartments[compartment]
            #----------------------------------------
            # Assign state ID number to each compartment (for internal state comparisons):
            self.stateID[compartment] = c
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
            if('flags' not in self.compartments[compartment]):
                self.compartments[compartment]['flags'] = []
            for flag in self.compartments[compartment]['flags']:
                self.allCompartmentFlags.add(flag)
            
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize other metadata:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.infectionLogs = []

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
        # Initialize counts/prevalences and the states of individuals:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.counts            = {}
        self.flag_counts       = {} 
        self.track_flag_counts = True
        self.store_Xseries     = store_Xseries
        self.process_initial_states()

        #----------------------------------------
        # Initialize exogenous prevalence for each compartment:
        self.exogenous_prevalence  = {}
        for c, compartment in enumerate(self.compartments):
            comp_params = self.compartments[compartment]
            self.exogenous_prevalence[compartment] = comp_params['exogenous_prevalence']
            
    
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
                if(key in ['transmissibilities', 'initial_prevalence', 'exogenous_prevalence', 'default_state', 'exclude_from_eff_pop', 'flags']):
                    pass
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Recurse through sub dictionaries
                elif(isinstance(value, dict)):
                    reshape_param_vals(value)
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Convert all other parameter values to arrays corresponding to the population size:
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                else:
                    nested_dict[key] = np.array(value).reshape((self.pop_size, 1)) if isinstance(value, (list, np.ndarray)) else np.full(fill_value=value, shape=(self.pop_size,1))

        reshape_param_vals(new_compartments)

        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
            for G in self.networks:
                #----------------------------------------
                # Process local transmissibility parameters for each network:
                #----------------------------------------
                self.process_local_transmissibility(transm_dict, G)
                #----------------------------------------
                # Process frequency-dependent transmission offset factors for each network:
                #----------------------------------------
                self.process_local_transm_offsets(transm_dict, G)
                #----------------------------------------
                # Pre-calculate Infectivity Matrices for each network,
                # which pre-combine transmissibility, adjacency, and freq-dep offset terms.
                #----------------------------------------
                # M_G = (AB)_G * D_G
                self.infectivity_mat[compartment][G] = scipy.sparse.csr_matrix.multiply(transm_dict[G], transm_dict['offsets'][G])
            #----------------------------------------
            if('exogenous' not in transm_dict or not isinstance(transm_dict['exogenous'], (int, float))):
                transm_dict['exogenous'] = 0.0
            #----------------------------------------
            if('global' not in transm_dict or not isinstance(transm_dict['global'], (int, float))):
                transm_dict['global'] = np.sum([np.sum(transm_dict[G][transm_dict[G]!=0]) for G in  self.networks]) / max(np.sum([transm_dict[G].count_nonzero() for G in  self.networks]), 1)

        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check the initial and exogenous_prevalence params for each compartment, defaulting to 0 when missing or invalid:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for compartment, comp_dict in new_compartments.items():
            if('initial_prevalence' not in comp_dict or not isinstance(comp_dict['initial_prevalence'], (int, float))):
                comp_dict['initial_prevalence'] = 0.0
            if('exogenous_prevalence' not in comp_dict or not isinstance(comp_dict['exogenous_prevalence'], (int, float))):
                comp_dict['exogenous_prevalence'] = 0.0

        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the model object with the new processed compartments
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.compartments.update(new_compartments)


    ########################################################
    ########################################################


    def calc_propensities(self):

        propensities     = []
        transitions      = []

        for compartment, comp_params in self.compartments.items():

            # Skip calculations for this compartment if no nodes are in this state:
            if(not np.any(self.X==self.stateID[compartment])):
                continue

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
                    propensity_temporal_transition = (self.transition_timer_wt * (np.greater(self.state_timer, transition_params['time']) & (self.X==self.stateID[compartment])) * transition_params['active_path']) if any(transition_params['time']) else np.zeros_like(self.X)

                else: # exponential_rates
                    propensity_temporal_transition = transition_params['rate'] * (self.X==self.stateID[compartment]) * transition_params['active_path']

                propensities.append(propensity_temporal_transition)
                transitions.append({'from':compartment, 'to':destState, 'type':'temporal'})

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Calc propensities of transmission-induced transitions:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for infectiousState, susc_params in comp_params['susceptibilities'].items():
                if(infectiousState not in self.compartments):
                    print("Infectious state", infectiousState, "is not a defined compartment.")
                    continue

                # Skip calculations for this infectious compartment if no nodes are in this state:
                if(not np.any(self.X==self.stateID[infectiousState])):
                    continue

                #----------------------------------------
                # Compute the local transmission propensity terms for individuals in each contact network
                #----------------------------------------
                if(infectiousState not in propensity_infection_local):

                    propensity_infection_local[infectiousState] = np.zeros((self.pop_size, 1))

                    denom_numContacts = np.zeros((self.pop_size, 1))

                    #----------------------------------------
                    # Get the number of contacts relevant for the local transmission denominator for each individual:
                    for netID, G in self.networks.items():
                        bool_isGactive    = (((G['active']!=0)&(self.isolation==0)) | ((G['active_isolation']!=0)&(self.isolation!=0))).flatten()
                        denom_numContacts += G['adj_matrix'][:,np.argwhere(bool_isGactive).flatten()].sum(axis=1) if self.local_trans_denom_mode=='active_contacts' else G['degree']

                    #----------------------------------------
                    # Compute the local transmission propensity terms:
                    #----------------------------------------
                    for netID, G in self.networks.items():

                        M = self.infectivity_mat[infectiousState][netID]

                        #----------------------------------------
                        # Determine which individuals need local transmission propensity calculated (active in G and infectible, non-zero propensity)
                        # and which individuals are relevant in these calculations (active in G and infectious):
                        #----------------------------------------
                        bool_isGactive    = (((G['active']!=0)&(self.isolation==0)) | ((G['active_isolation']!=0)&(self.isolation!=0))).flatten()
                        bin_isGactive     = [1 if i else 0 for i in bool_isGactive]

                        bool_isInfectious = (self.X==self.stateID[infectiousState]).flatten()
                        j_isInfectious    = np.argwhere(bool_isInfectious).flatten()

                        bool_hasGactiveInfectiousContacts = np.asarray(scipy.sparse.csr_matrix.dot(M, scipy.sparse.diags(bin_isGactive))[:,j_isInfectious].sum(axis=1).astype(bool)).flatten()

                        bool_isInfectible = (bool_isGactive & bool_hasGactiveInfectiousContacts)
                        i_isInfectible    = np.argwhere(bool_isInfectible).flatten()

                        #----------------------------------------
                        # Compute the local transmission propensity terms for individuals in the current contact network G
                        #----------------------------------------
                        propensity_infection_local[infectiousState][i_isInfectible] += np.divide( scipy.sparse.csr_matrix.dot(M[i_isInfectible,:][:,j_isInfectious], (self.X==self.stateID[infectiousState])[j_isInfectious]), denom_numContacts[i_isInfectible], out=np.zeros_like(propensity_infection_local[infectiousState][i_isInfectible]), where=denom_numContacts[i_isInfectible]!=0 )

                #----------------------------------------
                # Compute the propensities of infection for individuals across all transmission modes (exogenous, global, local over all networks)
                #----------------------------------------
                transm_params = self.compartments[infectiousState]['transmissibilities']

                propensity_infection = ((self.X==self.stateID[compartment]) * 
                                        (
                                            susc_params['susceptibility'] *
                                            (    
                                                 (self.openness) * (transm_params['exogenous']*self.exogenous_prevalence[compartment])
                                             + (1-self.openness) * (
                                                                        (self.mixedness) * ((transm_params['global']*self.counts[infectiousState][self.tidx])/self.N[self.tidx])
                                                                    + (1-self.mixedness) * (propensity_infection_local[infectiousState])
                                                                   )
                                            )
                                        ))

                #----------------------------------------
                # Compute the propensities of each possible infection-induced transition according to the disease progression paths of each individual:
                #----------------------------------------
                for destState, transition_params in susc_params['transitions'].items():
                    if(destState not in self.compartments):
                        print("Destination state", destState, "is not a defined compartment.")
                        continue

                    propensity_infection_transition = propensity_infection * transition_params['active_path']

                    propensities.append(propensity_infection_transition)
                    transitions.append({'from':compartment, 'to':destState, 'type':'infection'})

            #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        propensities = np.hstack(propensities) if len(propensities)>0 else np.array([[]])

        return propensities, transitions
        

    ########################################################
    ########################################################


    def run_iteration(self, max_dt=None, default_dt=0.1):

        max_dt = self.tmax if max_dt is None else max_dt

        if(self.tidx >= len(self.tseries)-1):
            # Room has run out in the timeseries storage arrays; double the size of these arrays:
            self.increase_data_series_length()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate 2 random numbers uniformly distributed in (0,1)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        r1 = np.random.rand()
        r2 = np.random.rand()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Calculate propensities
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        propensities, transitions = self.calc_propensities()

        if(propensities.sum() > 0):

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Calculate alpha
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            propensities_flat   = propensities.ravel(order='F')
            cumsum              = propensities_flat.cumsum()
            alpha               = propensities_flat.sum()

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute the time until the next event takes place
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            tau = (1/alpha)*np.log(float(1/r1))

            #----------------------------------------
            # If the time to next event exceeds the max allowed interval,
            # advance the system time by the max allowed interval,
            # but do not execute any events (recalculate Gillespie interval/event next iteration)
            if(tau > max_dt):

                # Advance time by max_dt step
                self.t           += max_dt
                self.state_timer += max_dt
                self.tidx        += 1

                # Update isolation timers/statuses
                i_isolated = np.argwhere(self.isolation==1).flatten()
                self.isolation_timer[i_isolated]    += max_dt
                self.totalIsolationTime[i_isolated] += max_dt
                if(self.isolation_period is not None):
                    i_exitingIsolation = np.argwhere(self.isolation_timer >= self.isolation_period).flatten()
                    for i in i_exitingIsolation:
                        self.set_isolation(node=i, isolation=False)

                # return without any further event execution
                self.update_data_series()
                return True

                #----------------------------------------

            else:
                # Advance time by tau
                self.t           += tau
                self.state_timer += tau
                self.tidx        += 1

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute which event takes place
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            transitionIdx  = np.searchsorted(cumsum,r2*alpha)
            transitionNode = transitionIdx % self.pop_size
            transition     = transitions[ int(transitionIdx/self.pop_size) ]

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Perform updates triggered by event:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            assert(self.X[transitionNode]==self.stateID[transition['from']]), "Assertion error: Node "+str(transitionNode)+" has unexpected current state "+str(self.X[transitionNode])+" given the intended transition of "+transition['from']+"->"+transition['to']+"."
            self.set_state(transitionNode, transition['to']) # self.X[transitionNode] = self.stateID[transition['to']]
            self.state_timer[transitionNode] = 0.0
            # TODO: some version of this?   self.testedInCurrentState[transitionNode] = False

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save information about infection events when they occur:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if(self.log_infection_info and transition['type']=='infection'):
                infectionLog = {'t':                  self.t,
                                'infected_node':      transitionNode,
                                'preInfectionState':  transition['from'],
                                'postInfectionState': transition['to'],
                                'contact_info':       {} 
                                }
                for netID, G in self.networks.items():
                    infectionLog['contact_info'].update({'num_contacts':G['degree'][transitionNode,0],
                                                         'contacts_states': self.X[ np.argwhere(G['adj_matrix'][transitionNode,:]>0) ].flatten().tolist(),
                                                         'contacts_isolations': self.isolation[ np.argwhere(G['adj_matrix'][transitionNode,:]>0) ].flatten().tolist()
                                                        })
                self.infectionLogs.append(infectionLog)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        else: # propensities.sum() == 0
            # No tau calculated, advance time by default step
            tau               = default_dt
            self.t           += tau
            self.state_timer += tau
            self.tidx        += 1

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        self.update_data_series()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Update isolation timers/statuses
        i_isolated = np.argwhere(self.isolation==1).flatten()
        self.isolation_timer[i_isolated]    += tau
        self.totalIsolationTime[i_isolated] += tau
        if(self.isolation_period is not None):
            i_exitingIsolation = np.argwhere(self.isolation_timer >= self.isolation_period).flatten()
            for i in i_exitingIsolation:
                self.set_isolation(node=i, isolation=False)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Terminate if tmax reached:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(self.t >= self.tmax):
            self.finalize_data_series()
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        return True


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

            running = self.run_iteration(max_dt, default_dt)

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
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the time series:
        self.tseries[self.tidx]     = self.t
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the data series of counts of nodes in each compartment:
        for compartment in self.compartments:
            self.counts[compartment][self.tidx] = np.count_nonzero(self.X==self.stateID[compartment])
            #------------------------------------
            if(compartment not in self.excludeFromEffPopSize):
                self.N[self.tidx] += self.counts[compartment][self.tidx]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the data series of counts of nodes with each flag:
        if(self.track_flag_counts):
            for flag in self.allCompartmentFlags.union(self.allNodeFlags):
                flag_count = len(self.get_individuals_by_flag(flag))
                self.flag_counts[flag][self.tidx] = flag_count
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store system states
        if(self.store_Xseries):
            self.Xseries[self.tidx,:] = self.X.T

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store system states for specified subgroups
        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                for compartment in self.compartments:
                    self.nodeGroupData[groupName][compartment][self.tidx] = np.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.stateID[compartment])
                    #------------------------------------
                    if(compartment not in self.excludeFromEffPopSize):
                        self.nodeGroupData[groupName]['N'][self.tidx] += self.counts[compartment][self.tidx]


    ########################################################


    def increase_data_series_length(self):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Allocate more entries for the time series:
        self.tseries = np.pad(self.tseries, [(0, self.pop_size*min(len(self.compartments), 10))], mode='constant', constant_values=0)
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
                prob = np.array(prob).reshape((self.pop_size, 1)) if isinstance(prob, (list, np.ndarray)) else np.full(fill_value=prob, shape=(self.pop_size,1))
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
            transn_dict[poststate]["active_path"] = np.array([1 if rands[i]==poststate else 0 for i in range(self.pop_size)]).reshape((self.pop_size,1))


    ########################################################


    def process_local_transmissibility(self, transm_dict, network):

        #----------------------------------------
        # Process local transmissibility parameters for each network:
        #----------------------------------------
        try:
            # Use transmissibility values provided for this network if given,
            # else us transmissibility values provided under generic 'local' key.
            # (If neither of these are provided, defaults to 0 transmissibility in except)
            local_transm_vals = transm_dict[network] if network in transm_dict else transm_dict['local']
            #----------------------------------------
            # Convert transmissibility value(s) to np array:
            local_transm_vals = np.array(local_transm_vals) if isinstance(local_transm_vals, (list, np.ndarray)) else np.full(fill_value=local_transm_vals, shape=(self.pop_size,1))
            #----------------------------------------
            # Generate matrix of pairwise transmissibility values:
            if(local_transm_vals.ndim == 2 and local_transm_vals.shape[0] == self.pop_size and local_transm_vals.shape[1] == self.pop_size):
                net_transm_mat = local_transm_vals
            elif((local_transm_vals.ndim == 1 and local_transm_vals.shape[0] == self.pop_size) or (local_transm_vals.ndim == 2 and (local_transm_vals.shape[0] == self.pop_size or local_transm_vals.shape[1] == self.pop_size))):
                local_transm_vals = local_transm_vals.reshape((self.pop_size,1))
                # Pre-multiply beta values by the adjacency matrix ("transmission weight connections")
                A_beta_pairwise_byInfected = scipy.sparse.csr_matrix.multiply(self.networks[network]["adj_matrix"], local_transm_vals.T).tocsr()
                A_beta_pairwise_byInfectee = scipy.sparse.csr_matrix.multiply(self.networks[network]["adj_matrix"], local_transm_vals).tocsr()    
                #------------------------------
                # Compute the effective pairwise beta values as a function of the infected/infectee pair:
                if(transm_dict['pairwise_mode'].lower() == 'infected'):
                    net_transm_mat = A_beta_pairwise_byInfected
                elif(transm_dict['pairwise_mode'].lower() == 'infectee'):
                    net_transm_mat = A_beta_pairwise_byInfectee
                elif(transm_dict['pairwise_mode'].lower() == 'min'):
                    net_transm_mat = scipy.sparse.csr_matrix.minimum(A_beta_pairwise_byInfected, A_beta_pairwise_byInfectee)
                elif(transm_dict['pairwise_mode'].lower() == 'max'):
                    net_transm_mat = scipy.sparse.csr_matrix.maximum(A_beta_pairwise_byInfected, A_beta_pairwise_byInfectee)
                elif(transm_dict['pairwise_mode'].lower() == 'mean' or transm_dict['pairwise_mode'] is None):
                    net_transm_mat = (A_beta_pairwise_byInfected + A_beta_pairwise_byInfectee)/2
                else:
                    raise BaseException("Unrecognized pairwise_mode value (support for 'infected', 'infectee', 'min', 'max', and 'mean').")
            else:
                raise BaseException("Invalid data type/shape for transmissibility values.")
            #----------------------------------------
            # Store the pairwise transmissibility matrix in the compartments dict
            transm_dict[network] = net_transm_mat
        except KeyError:
            # print("Transmissibility values not given for \""+str(G)+"\" network -- defaulting to 0.")
            transm_dict[network] = scipy.sparse.csr_matrix(np.zeros(shape=(self.pop_size, self.pop_size)))


    ########################################################


    def process_local_transm_offsets(self, transm_dict, network):
        #----------------------------------------
        # Process frequency-dependent transmission offset factors for each network:
        #----------------------------------------
        if('offsets' not in transm_dict):
            transm_dict['offsets'] = {}
        #----------------------------------------
        try:
            omega_vals = transm_dict['offsets'][network]
            #----------------------------------------
            # Convert omega value(s) to np array:
            omega_vals = np.array(omega_vals) if isinstance(omega_vals, (list, np.ndarray)) else np.full(fill_value=omega_vals, shape=(self.pop_size,1))
            #----------------------------------------
            # Store 2d np matrix of pairwise omega values:
            if(omega_vals.ndim == 2 and omega_vals.shape[0] == self.pop_size and omega_vals.shape[1] == self.pop_size):
                nested_dic[G+"_omega"] = omega_vals
            else:
                raise BaseException("Explicit omega values should be specified as an NxN 2d array. Else leave unspecified and omega values will be automatically calculated according to local_transm_offset_mode.")
        except KeyError:
            #----------------------------------------
            # Automatically generate omega matrix according to local_transm_offset_mode:
            if(transm_dict['local_transm_offset_mode'].lower() == 'pairwise_log'):
                with np.errstate(divide='ignore'): # ignore log(0) warning, then convert log(0) = -inf -> 0.0
                    omega = np.log(np.maximum(self.networks[network]["degree"],2))/np.log(np.mean(self.networks[network]["degree"])) 
                    omega[np.isneginf(omega)] = 0.0
            elif(transm_dict['local_transm_offset_mode'].lower() == 'pairwise_linear'):
                omega = np.maximum(self.networks[network]["degree"],2)/np.mean(self.networks[network]["degree"])
            elif(transm_dict['local_transm_offset_mode'].lower() == 'none'):
                omega = np.ones(shape=(self.pop_size, self.pop_size))
            else:
                raise BaseException("Unrecognized local_transm_offset_mode value (support for 'pairwise_log', 'pairwise_linear', and 'none').")
            omega_pairwise_byInfected = scipy.sparse.csr_matrix.multiply(self.networks[network]["adj_matrix"], omega.T).tocsr()
            omega_pairwise_byInfectee = scipy.sparse.csr_matrix.multiply(self.networks[network]["adj_matrix"], omega).tocsr()
            omega_mat = (omega_pairwise_byInfected + omega_pairwise_byInfectee)/2
            #----------------------------------------
            # Store the pairwise omega matrix in the compartments dict
            transm_dict['offsets'][network] = omega_mat


    ########################################################


    def process_initial_states(self):
        
        #----------------------------------------
        # Determine the iniital counts for each state given their specified initial prevalences
        initCountTotal = 0
        for c, compartment in enumerate(self.compartments):
            comp_params = self.compartments[compartment]
            #----------------------------------------
            # Instantiate data series for counts of nodes in each compartment:
            self.counts[compartment] = np.zeros(self.pop_size*min(len(self.compartments), 10))
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
        # Initialize the states of individuals:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.X = np.concatenate([[self.stateID[comp]]*int(self.counts[comp][0]) for comp in self.compartments]).reshape((self.pop_size,1))
        np.random.shuffle(self.X)

        if(self.store_Xseries):
            self.Xseries        = np.zeros(shape=(6*self.pop_size, self.pop_size), dtype='uint8')
            self.Xseries[0,:]   = self.X.T

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #----------------------------------------
        # Determine the iniital counts for each flag
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


    def set_state(self, node, state):
        # Using this function instead of setting self.X directly ensures that the data series are updated whenever a state changes.
        nodes = [node] if not isinstance(node, (list, np.ndarray)) else node
        for i in nodes:
            if(state in self.compartments):
                self.X[i] = self.stateID[state]
            elif(state in self.stateID):
                self.X[i] = state
            else:
                print("Unrecognized state, "+str(state)+". No state update performed.")
                return
        self.update_data_series()


    ########################################################


    def set_transition_rate(self, compartment, to, rate):
        # Note that it only makes sense to set a rate for temporal transitions.
        compartments = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        destStates   = [to] if not isinstance(to, (list, np.ndarray)) else to
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
        compartments = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        destStates   = [to] if not isinstance(to, (list, np.ndarray)) else to
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
        compartments     = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        infectiousStates = [upon_exposure_to] if (not isinstance(upon_exposure_to, (list, np.ndarray)) and upon_exposure_to is not None) else upon_exposure_to
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
        compartments     = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        infectiousStates = [to] if not isinstance(to, (list, np.ndarray)) else to
        susceptibility   = np.array(susceptibility).reshape((self.pop_size,1))
        for compartment in compartments:
            for infectiousState in infectiousStates:
                susc_dict = self.compartments[compartment]['susceptibilities']
                try:
                    susc_dict[infectiousState]['susceptibility'] = susceptibility
                except KeyError:
                    susc_dict[infectiousState] = {'susceptibility': susceptibility}


    ########################################################


    def set_transmissibility(self, compartment, transm_mode, transmissibility):
        compartments = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        transmModes  = [transm_mode] if not isinstance(transm_mode, (list, np.ndarray)) else transm_mode
        for compartment in compartments:
            transm_dict = self.compartments[compartment]['transmissibilities']
            for transmMode in transmModes:
                #----------------------------------------
                # Handle update to local transmissibility over the specified network:
                if(transmMode in self.networks):
                    transm_dict[transmMode] = transmissibility
                    self.process_local_transmissibility(transm_dict, transmMode)
                    #----------------------------------------
                    # Re-calculate Infectivity Matrices for updated compartments/networks,
                    # which pre-combine transmissibility, adjacency, and freq-dep offset terms.
                    #----------------------------------------
                    # M_G = (AB)_G * D_G
                    self.infectivity_mat[compartment][transmMode] = scipy.sparse.csr_matrix.multiply(transm_dict[transmMode], transm_dict['offsets'][transmMode])
                #----------------------------------------
                # Handle update to exogenous transmissibility:
                elif(transmMode=='exogenous'):
                    transm_dict['exogenous'] = np.array(transmissibility).reshape((self.pop_size,1))
                    self.exogenous_prevalence[compartment] = transm_dict['exogenous']
                #----------------------------------------
                else:
                    print("Transmission mode,", transmMode, "not recognized (expected 'exogenous', or network name in "+str(list(self.networks.keys()))+"); no update.")
            #----------------------------------------
            # Re-calculate global transmissibility as the mean of local transmissibilities.
            #----------------------------------------
            transm_dict['global'] = np.sum([np.sum(transm_dict[G][transm_dict[G]!=0]) for G in self.networks]) / max(np.sum([transm_dict[G].count_nonzero() for G in  self.networks]), 1)


    ########################################################


    def set_initial_prevalence(self, compartment, prevalence):
        compartments = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        for compartment in compartments:
            self.compartments[compartment]['initial_prevalence'] = prevalence
        self.process_initial_states()

    ########################################################

    
    def set_exogenous_prevalence(self, compartment, prevalence):
        compartments = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        for compartment in compartments:
            # Update the compartment model definition dictionary
            self.compartments[compartment]['exogenous_prevalence'] = prevalence
            # Update the exogenous prevalence variable in the model object
            self.exogenous_prevalence[compartment] = prevalence


    ########################################################

    
    def set_default_state(self, compartment):
        # Must be a single compartment given
        for c in self.compartments:
            self.compartments[c]['default_state'] = (c == compartment)
            if(c == compartment):
                self.default_state = self.stateID[c]    
       

    ########################################################

    
    def set_exclude_from_eff_pop(self, compartment, exclude=True):
        compartments = [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        for compartment in compartments:
            self.compartments[compartment]['exclude_from_eff_pop'] = exclude
            if(exclude and not compartment in self.excludeFromEffPopSize):
                self.excludeFromEffPopSize.append(compartment)
            elif(not exclude and compartment in self.excludeFromEffPopSize):
                self.excludeFromEffPopSize = [c for c in self.excludeFromEffPopSize if c!=compartment] # remove all occurrences of compartment


    ########################################################


    def set_isolation(self, node, isolation):
        nodes = [node] if not isinstance(node, (list, np.ndarray)) else node
        for node in nodes:
            if(isolation == True):
                # self.calc_infectious_time(node) <- TODO handle this?
                self.isolation[node] = 1
            elif(isolation == False):
                self.isolation[node] = 0
            # Reset the isolation timer:
            self.isolation_timer[node] = 0


    ########################################################


    def set_network_activity(self, network, node='all', active=None, active_isolation=None):
        nodes      = list(range(self.pop_size)) if node=='all' else [node] if not isinstance(node, (list, np.ndarray)) else node
        networks = [network] if not isinstance(network, (list, np.ndarray)) else network
        for i in nodes:
            for G in networks:
                if(active is not None):
                    self.networks[G]['active'][i] = 1 if active else 0
                if(active_isolation is not None):
                    self.networks[G]['active_isolation'][i] = 1 if active_isolation else 0


    ########################################################


    def get_node_compartment(self, node):
        node_list_provided = isinstance(node, (list, np.ndarray))
        nodes = list(range(self.pop_size)) if node=='all' else [node] if not node_list_provided else node
        compartments = []
        for node in nodes:
            stateID     = self.X[node][0]
            compartments.append( list(self.stateID.keys())[list(self.stateID.values()).index(stateID)] )
        return compartments if node_list_provided else compartments[0] if len(compartments)>0 else None



    ########################################################
    ########################################################


    def add_compartment_flag(self, compartment, flag):
        compartments = list(range(self.pop_size)) if compartment=='all' else [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        flags        = [flag] if not isinstance(flag, (list, np.ndarray)) else flag
        for compartment in compartments:
            for flag in flags:
                self.compartments[compartment]['flags'].append(flag)
                self.allCompartmentFlags.add(flag)
                if(flag not in self.flag_counts):
                    self.flag_counts[flag] = np.zeros_like(self.counts[compartment])
                    self.update_data_series()


    def remove_compartment_flag(self, compartment, flag):
        compartments = list(range(self.pop_size)) if compartment=='all' else [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        flags        = [flag] if not isinstance(flag, (list, np.ndarray)) else flag
        for compartment in compartments:
            for flag in flags:
                self.compartments[compartment]['flags'] = [f for f in self.compartments[compartment]['flags'] if f!=flag] # remove all occurrences of flag


    ########################################################


    def add_node_flag(self, node, flag):
        nodes = list(range(self.pop_size)) if node=='all' else [node] if not isinstance(node, (list, np.ndarray)) else node
        flags = [flag] if not isinstance(flag, (list, np.ndarray)) else flag
        for node in nodes:
            for flag in flags:
                self.node_flags.append(flag)
                self.allNodeFlags.add(flag)
                if(flag not in self.flag_counts):
                    self.flag_counts[flag] = np.zeros_like(self.counts[list(self.counts.keys())[0]])
                    self.update_data_series()


    def remove_node_flag(self, node, flag):
        nodes = list(range(self.pop_size)) if node=='all' else [node] if not isinstance(node, (list, np.ndarray)) else node
        flags = [flag] if not isinstance(flag, (list, np.ndarray)) else flag
        for node in nodes:
            for flag in flags:
                self.node_flags = [f for f in self.node_flags if f!=flag] # remove all occurrences of flag


    ########################################################


    def get_compartments_by_flag(self, flag, has_flag=True):
        flags = [flag] if not isinstance(flag, (list, np.ndarray)) else flag
        flagged_compartments = set()
        for compartment, comp_dict in self.compartments.items():
            if(any([flag in comp_dict['flags'] for flag in flags]) == has_flag):
                flagged_compartments.add(compartment)
        return list(flagged_compartments)


    ########################################################


    def get_individuals_by_flag(self, flag, has_flag=True):
        flags = [flag] if not isinstance(flag, (list, np.ndarray)) else flag
        flagged_compartments = self.get_compartments_by_flag(flags)
        node_flagged_individuals = set()
        comp_flagged_individuals = set()
        for i in range(self.pop_size):
            # Check if individual i has this node flag:
            if(any([flag in self.node_flags[i] for flag in flags]) == has_flag):
                node_flagged_individuals.add(i)
            # Check if individual i is in a compartment with this flag:
            if(any([ self.X[i]==self.stateID[c] for c in flagged_compartments ]) == has_flag):
                comp_flagged_individuals.add(i)
        if(has_flag):
            return list(node_flagged_individuals | comp_flagged_individuals)
        else:
            return list(node_flagged_individuals & comp_flagged_individuals)


    ########################################################


    def get_count_by_flag(self, flag, has_flag=True):
        flags = [flag] if not isinstance(flag, (list, np.ndarray)) else flag
        flag_counts_ = {}
        for flag in flags:
            flag_counts_[flag] = len(self.get_individuals_by_flag(flag, has_flag))
        return flag_counts_ if len(flag_counts_)>1 else np.sum([flag_counts_[f] for f in flag_counts_]) if len(flag_counts_)>0 else None


    ########################################################
    ########################################################


    def introduce_random_exposures(self, num, compartment='all', exposed_to='any'):
        compartments      = list(self.compartments.keys()) if compartment=='all' else [compartment] if not isinstance(compartment, (list, np.ndarray)) else compartment
        infectiousStates  = list(self.compartments.keys()) if exposed_to=='any'  else [exposed_to] if (not isinstance(exposed_to, (list, np.ndarray)) and exposed_to is not None) else exposed_to
        exposure_susceptibilities = []
        exposedNodes = []
        for exposure in range(num):
            for compartment in compartments:
                for infectiousState in infectiousStates:
                    if(infectiousState in self.compartments[compartment]['susceptibilities']):
                        exposure_susceptibilities.append({'susc_state': compartment, 'inf_state': infectiousState, 
                                                          'susceptibilities': self.compartments[compartment]['susceptibilities'][infectiousState]['susceptibility'].flatten(),
                                                          'mean_susceptibility': np.mean(self.compartments[compartment]['susceptibilities'][infectiousState]['susceptibility']),
                                                          })
            exposureType   = np.random.choice(exposure_susceptibilities, p=[d['mean_susceptibility'] for d in exposure_susceptibilities]/np.sum([d['mean_susceptibility'] for d in exposure_susceptibilities]))
            exposableNodes = [i for i in range(self.pop_size) if self.X[i]==self.stateID[exposureType['susc_state']]]
            if(len(exposableNodes) > 0):
                exposedNode    = np.random.choice(exposableNodes, p=exposureType['susceptibilities'][exposableNodes]/np.sum(exposureType['susceptibilities'][exposableNodes]))
                exposedNodes.append(exposedNode)
                #--------------------
                exposureTransitions = self.compartments[exposureType['susc_state']]['susceptibilities'][exposureType['inf_state']]['transitions']
                exposureTransitionsActiveStatuses = [exposureTransitions[dest]['active_path'].flatten()[exposedNode] for dest in exposureTransitions]
                destState = np.random.choice(list(exposureTransitions.keys()), p=exposureTransitionsActiveStatuses/np.sum(exposureTransitionsActiveStatuses))
                #--------------------
                self.set_state(exposedNode, destState)
        return exposedNodes


    ########################################################


    def test(self, node, test_type):
        node_list_provided = isinstance(node, (list, np.ndarray))
        nodes = list(range(self.pop_size)) if node=='all' else [node] if not node_list_provided else node
        results = []
        for node in nodes:
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
                positive_result = (np.random.rand() < sensitivity)
            elif(specificity < 1.0): # individual is in a state where the test can return a false positive
                positive_result = (np.random.rand() > specificity)
            else:
                positive_result = False
            results.append(positive_result)
        return results if node_list_provided else results[0] if len(results)>0 else None


    ########################################################
    ########################################################


    def run_with_interventions(self, T, max_dt=0.1, default_dt=0.1, run_full_duration=False,
                                    # Intervention timing params:
                                    cadence_dt=1, 
                                    cadence_cycle_length=28,
                                    init_cadence_offset=0,
                                    cadence_presets='default',
                                    intervention_start_time=0,
                                    intervention_start_prevalence=0,
                                    prevalence_flags=['infected'],
                                    # State onset intervention params:
                                    onset_compartments=[], # not yet used
                                    onset_flags=[], 
                                    # Isolation params:
                                    isolation_delay_onset=0,
                                    isolation_delay_onset_groupmate=0,
                                    isolation_delay_positive=1,
                                    isolation_delay_positive_groupmate=1,
                                    isolation_delay_traced=0,
                                    isolation_compliance_onset=True, 
                                    isolation_compliance_onset_groupmate=False,
                                    isolation_compliance_positive=True,
                                    isolation_compliance_positive_groupmate=False,
                                    isolation_compliance_traced=False,
                                    isolation_exclude_compartments=[],          
                                    isolation_exclude_flags=[],      
                                    isolation_exclude_isolated=False,           
                                    isolation_exclude_afterNumTests=None,       
                                    isolation_exclude_afterNumVaccineDoses=None,
                                    # Testing params:
                                    test_params=None, 
                                    test_type_proactive=None,
                                    test_type_onset=None,
                                    test_type_traced=None, 
                                    proactive_testing_cadence='never',
                                    testing_capacity_max=1.0,
                                    testing_capacity_proactive=0.0,
                                    testing_delay_proactive=0,
                                    testing_delay_onset=1,
                                    testing_delay_onset_groupmate=1,
                                    testing_delay_positive_groupmate=1,
                                    testing_delay_traced=1,                                    
                                    testing_compliance_proactive=True,
                                    testing_compliance_onset=False, 
                                    testing_compliance_onset_groupmate=False,
                                    testing_compliance_positive_groupmate=False,
                                    testing_compliance_traced=False,
                                    testing_exclude_compartments=[],
                                    testing_exclude_flags=[],
                                    testing_exclude_isolated=False,
                                    testing_exclude_afterNumTests=None,
                                    testing_exclude_afterNumVaccineDoses=None,
                                    # Tracing params:                                                                       
                                    tracing_num_contacts=None, 
                                    tracing_pct_contacts=0,
                                    tracing_delay=1,
                                    tracing_compliance=True,
                                    # Misc. params:
                                    intervention_groups=None
                                ):

        if(T>0):
            self.tmax += T
        else:
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize intervention parameters:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #----------------------------------------
        # Initialize intervention-related model parameters:
        #----------------------------------------
        self.num_tests         = np.zeros(self.pop_size)
        self.num_vaccine_doses = np.zeros(self.pop_size)

        #----------------------------------------
        # Initialize cadence and intervention time parameters:
        #----------------------------------------
        interventionOn = False
        interventionStartTime = None

        # Cadences involve a repeating (default 28 day) cycle starting on a Monday
        # (0:Mon, 1:Tue, 2:Wed, 3:Thu, 4:Fri, 5:Sat, 6:Sun, 7:Mon, 8:Tues, ...)
        # For each cadence, actions are done on the cadence intervals included in the associated list.
        if(cadence_presets == 'default'):
            cadence_presets    = {
                                        'everyday':     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
                                        'workday':      [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25],
                                        'semiweekly':   [0, 3, 7, 10, 14, 17, 21, 24],
                                        'weekly':       [0, 7, 14, 21],
                                        'biweekly':     [0, 14],
                                        'monthly':      [0],
                                        'initial':      [0],
                                        'never':        []
                                    }
        if(init_cadence_offset == 'random'):
            init_cadence_offset = np.random.choice(range(cadence_cycle_length))

        last_cadence_time  = -1

        #----------------------------------------
        # Initialize onset parameters:
        #----------------------------------------
        onset_flags = [onset_flags] if not isinstance(onset_flags, (list, np.ndarray)) else onset_flags

        flag_onset = {flag: [False]*self.pop_size for flag in onset_flags} # bools for tracking which onsets have triggered for each individual
        
        #----------------------------------------
        # Initialize testing parameters:
        #----------------------------------------
        #........................................
        def process_test_parameters(test_params):
            if(isinstance(test_params, str) and '.json' in test_params):
                with open(test_params) as test_params_file:
                    test_params = json.load(test_params_file)
            elif(isinstance(test_params, dict)):
                pass
            elif(test_params is None):
                # If no test params are given, default to a test that is 100% sensitive/specific to all compartments with the 'infected' flag:
                test_params = {}
                infectedFlagCompartments = self.get_compartments_by_flag(prevalence_flags)
                for compartment in self.compartments:
                    test_params.update({compartment: {"default_test": {"sensitivity": 1.0 if compartment in infectedFlagCompartments else 0.0, "specificity": 1.0}}})
            else:
                raise BaseException("Specify test parameters with a dictionary or JSON file.")
            #----------------------------------------
            test_types = set()
            for compartment, comp_params in test_params.items():
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
            self.test_params = test_params
            self.test_types  = test_types
            return test_params, test_types
        #........................................

        process_test_parameters(test_params)

        test_type_onset     = test_type_onset if test_type_onset is not None else list(self.test_types)[0] if len(self.test_types)>0 else None
        test_type_traced    = test_type_traced if test_type_traced is not None else list(self.test_types)[0] if len(self.test_types)>0 else None
        test_type_proactive = test_type_proactive if test_type_proactive is not None else list(self.test_types)[0] if len(self.test_types)>0 else None

        proactiveTestingTimes = [cadence_presets[individual_cadence] for individual_cadence in proactive_testing_cadence] if isinstance(proactive_testing_cadence, (list, np.ndarray)) else [cadence_presets[proactive_testing_cadence]]*self.pop_size

        #----------------------------------------
        # Initialize individual compliances:
        #----------------------------------------
        isolation_compliance_onset              = np.array([isolation_compliance_onset]*self.pop_size if not isinstance(isolation_compliance_onset, (list, np.ndarray)) else isolation_compliance_onset)
        isolation_compliance_onset_groupmate    = np.array([isolation_compliance_onset_groupmate]*self.pop_size if not isinstance(isolation_compliance_onset_groupmate, (list, np.ndarray)) else isolation_compliance_onset_groupmate)
        isolation_compliance_positive           = np.array([isolation_compliance_positive]*self.pop_size if not isinstance(isolation_compliance_positive, (list, np.ndarray)) else isolation_compliance_positive)
        isolation_compliance_positive_groupmate = np.array([isolation_compliance_positive_groupmate]*self.pop_size if not isinstance(isolation_compliance_positive_groupmate, (list, np.ndarray)) else isolation_compliance_positive_groupmate)
        isolation_compliance_traced             = np.array([isolation_compliance_traced]*self.pop_size if not isinstance(isolation_compliance_traced, (list, np.ndarray)) else isolation_compliance_traced)
        testing_compliance_proactive            = np.array([testing_compliance_proactive]*self.pop_size if not isinstance(testing_compliance_proactive, (list, np.ndarray)) else testing_compliance_proactive)
        testing_compliance_onset                = np.array([testing_compliance_onset]*self.pop_size if not isinstance(testing_compliance_onset, (list, np.ndarray)) else testing_compliance_onset)
        testing_compliance_onset_groupmate      = np.array([testing_compliance_onset_groupmate]*self.pop_size if not isinstance(testing_compliance_onset_groupmate, (list, np.ndarray)) else testing_compliance_onset_groupmate)
        testing_compliance_positive_groupmate   = np.array([testing_compliance_positive_groupmate]*self.pop_size if not isinstance(testing_compliance_positive_groupmate, (list, np.ndarray)) else testing_compliance_positive_groupmate)
        testing_compliance_traced               = np.array([testing_compliance_traced]*self.pop_size if not isinstance(testing_compliance_traced, (list, np.ndarray)) else testing_compliance_traced)        
        tracing_compliance                      = np.array([tracing_compliance]*self.pop_size if not isinstance(tracing_compliance, (list, np.ndarray)) else tracing_compliance)        

        #----------------------------------------
        # Initialize intervention exclusion criteria:
        #----------------------------------------
        isolation_exclude_afterNumTests        = np.inf if isolation_exclude_afterNumTests is None else isolation_exclude_afterNumTests
        isolation_exclude_afterNumVaccineDoses = np.inf if isolation_exclude_afterNumVaccineDoses is None else isolation_exclude_afterNumVaccineDoses
        testing_exclude_afterNumTests          = np.inf if testing_exclude_afterNumTests is None else testing_exclude_afterNumTests
        testing_exclude_afterNumVaccineDoses   = np.inf if testing_exclude_afterNumVaccineDoses is None else testing_exclude_afterNumVaccineDoses
        testing_exclude_compartments           = [self.stateID[c] for c in testing_exclude_compartments]

        #----------------------------------------
        # Initialize intervention queues:
        #----------------------------------------
        isolationQueue_onset                    = [set() for i in range(int(isolation_delay_onset/cadence_dt) + (1 if np.fmod(isolation_delay_onset, cadence_dt)>0 else 0))]
        isolationQueue_onset_groupmate          = [set() for i in range(int(isolation_delay_onset_groupmate/cadence_dt) + (1 if np.fmod(isolation_delay_onset_groupmate, cadence_dt)>0 else 0))]
        isolationQueue_positive                 = [set() for i in range(int(isolation_delay_positive/cadence_dt) + (1 if np.fmod(isolation_delay_positive, cadence_dt)>0 else 0))]
        isolationQueue_positive_groupmate       = [set() for i in range(int(isolation_delay_positive_groupmate/cadence_dt) + (1 if np.fmod(isolation_delay_positive_groupmate, cadence_dt)>0 else 0))]
        isolationQueue_traced                   = [set() for i in range(int(isolation_delay_traced/cadence_dt) + (1 if np.fmod(isolation_delay_traced, cadence_dt)>0 else 0))]
        testingQueue_onset                      = [set() for i in range(int(testing_delay_onset/cadence_dt) + (1 if np.fmod(testing_delay_onset, cadence_dt)>0 else 0))]
        testingQueue_onset_groupmate            = [set() for i in range(int(testing_delay_onset_groupmate/cadence_dt) + (1 if np.fmod(testing_delay_onset_groupmate, cadence_dt)>0 else 0))]
        testingQueue_positive_groupmate         = [set() for i in range(max(1, (int(testing_delay_positive_groupmate/cadence_dt) + (1 if np.fmod(testing_delay_positive_groupmate, cadence_dt)>0 else 0))))]
        testingQueue_traced                     = [set() for i in range(max(1, (int(testing_delay_traced/cadence_dt) + (1 if np.fmod(testing_delay_traced, cadence_dt)>0 else 0))))]
        testingQueue_proactive                  = [set() for i in range(int(testing_delay_proactive/cadence_dt) + (1 if np.fmod(testing_delay_proactive, cadence_dt)>0 else 0))]
        tracingQueue                            = [set() for i in range(int(tracing_delay/cadence_dt) + (1 if np.fmod(tracing_delay, cadence_dt)>0 else 0))]

        #----------------------------------------
        # Initialize intervention stats:
        #----------------------------------------
        totalNumTests_proactive               = 0
        totalNumTests_onset                   = 0
        totalNumTests_onset_groupmate         = 0
        totalNumTests_positive_groupmate      = 0
        totalNumTests_traced                  = 0
        totalNumTests                         = 0
        totalNumPositives_proactive           = 0
        totalNumPositives_onset               = 0
        totalNumPositives_onset_groupmate     = 0
        totalNumPositives_positive_groupmate  = 0
        totalNumPositives_traced              = 0
        totalNumPositives                     = 0
        totalNumIsolations_onset              = 0
        totalNumIsolations_onset_groupmate    = 0
        totalNumIsolations_positive           = 0
        totalNumIsolations_positive_groupmate = 0
        totalNumIsolations_traced             = 0
        totalNumIsolations                    = 0


        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Run the simulation loop:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        running     = True
        while running: 

            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # print("t = ", self.t)

            current_cadence_time = ((self.t + init_cadence_offset) - np.fmod((self.t + init_cadence_offset), cadence_dt)) % (cadence_cycle_length - np.fmod(cadence_cycle_length, cadence_dt))
            if(current_cadence_time != last_cadence_time):

                last_cadence_time = current_cadence_time

                currentNumInfected = self.get_count_by_flag(prevalence_flags)
                currentPrevalence  = currentNumInfected/self.N[self.tidx]
                currentNumIsolated = np.count_nonzero(self.isolation)

                if(currentPrevalence >= intervention_start_prevalence and not interventionOn):
                    interventionOn        = True
                    interventionStartTime = self.t
                
                if(interventionOn):

                    print("[INTERVENTIONS @ t = %.2f (t_cadence ~%.2f) :: Currently %d infected (%.2f%%), %d isolated]" % (self.t, current_cadence_time, currentNumInfected, currentPrevalence*100, currentNumIsolated))
                    print("\tState counts: ", list(zip(np.unique(self.X, return_counts=True)[0], np.unique(self.X, return_counts=True)[-1])))

                    isolationSet_onset              = set()
                    isolationSet_onset_groupmate    = set()
                    isolationSet_positive           = set()
                    isolationSet_positive_groupmate = set()
                    isolationSet_traced             = set()
                    
                    testingSet_onset                = set()
                    testingSet_onset_groupmate      = set()
                    testingSet_positive_groupmate   = set()
                    testingSet_traced               = set()
                    testingSet_proactive            = set()

                    tracingSet                      = set()

                    #---------------------------------------------
                    # Exclude the following individuals from all isolation:
                    # (these lists referenced in proactive isolation selection and isolation execution below)
                    #---------------------------------------------
                    isolation_excluded_byFlags        = (np.isin(range(self.pop_size), self.get_individuals_by_flag(isolation_exclude_flags))).flatten()
                    isolation_excluded_byCompartments = (np.isin(self.X, isolation_exclude_compartments)).flatten()
                    isolation_excluded_byIsolation    = (self.isolation == True).flatten() if isolation_exclude_isolated else np.array([False]*self.pop_size)
                    isolation_excluded_byNumTests     = (self.num_tests >= isolation_exclude_afterNumTests).flatten()
                    isolation_excluded_byVaccineDoses = (self.num_vaccine_doses >= isolation_exclude_afterNumVaccineDoses).flatten()
                    
                    isolation_excluded                = (isolation_excluded_byFlags | isolation_excluded_byCompartments | isolation_excluded_byIsolation | isolation_excluded_byNumTests | isolation_excluded_byVaccineDoses)

                    isolation_nonExcludedIndividuals  = set(np.argwhere(isolation_excluded==False).flatten())

                    #---------------------------------------------
                    # Exclude the following individuals from all testing:
                    # (these lists referenced in proactive testing selection and testing execution below)
                    #---------------------------------------------
                    testing_excluded_byFlags        = (np.isin(range(self.pop_size), self.get_individuals_by_flag(testing_exclude_flags))).flatten()
                    testing_excluded_byCompartments = (np.isin(self.X, testing_exclude_compartments)).flatten()
                    testing_excluded_byIsolation    = (self.isolation == True).flatten() if testing_exclude_isolated else np.array([False]*self.pop_size)
                    testing_excluded_byNumTests     = (self.num_tests >= testing_exclude_afterNumTests).flatten()
                    testing_excluded_byVaccineDoses = (self.num_vaccine_doses >= testing_exclude_afterNumVaccineDoses).flatten()
                    
                    testing_excluded                = (testing_excluded_byFlags | testing_excluded_byCompartments | testing_excluded_byIsolation | testing_excluded_byNumTests | testing_excluded_byVaccineDoses)

                    testing_nonExcludedIndividuals  = set(np.argwhere(testing_excluded==False).flatten())


                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Upon onset of flagged state (e.g., symptoms):
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    if(any(isolation_compliance_onset) or any(testing_compliance_onset)
                       or (intervention_groups is not None and (any(isolation_compliance_onset_groupmate) or any(testing_compliance_onset_groupmate)))):
                        for isoflag in onset_flags:
                            for flaggedIndividual in self.get_individuals_by_flag(isoflag):
                                if(flag_onset[isoflag][flaggedIndividual]==False):
                                    # This is the onset (first cadence interval) of this flag for this individual.
                                    flag_onset[isoflag][flaggedIndividual] = True
                                    #---------------------------------------------
                                    # Isolate individual upon onset of this flag:
                                    #---------------------------------------------
                                    if(isolation_compliance_onset[flaggedIndividual]):
                                        isolationSet_onset.add(flaggedIndividual)
                                    #---------------------------------------------
                                    # Test individual upon onset of this flag:
                                    #---------------------------------------------
                                    if(testing_compliance_onset[flaggedIndividual]):
                                        testingSet_onset.add(flaggedIndividual)
                                    #---------------------------------------------
                                    # Isolate and/or Test groupmates of individuals with onset of this flag:
                                    #---------------------------------------------
                                    if(intervention_groups is not None and (any(isolation_compliance_onset_groupmate) or any(testing_compliance_onset_groupmate))):
                                        groupmates = next((group for group in intervention_groups if flaggedIndividual in group), None)
                                        if(groupmates is not None):
                                            for groupmate in groupmates:
                                                if(groupmate != flaggedIndividual):
                                                    #----------------------
                                                    # Isolate  groupmates:
                                                    if(isolation_compliance_onset_groupmate[groupmate]):
                                                        isolationSet_onset_groupmate.add(groupmate)                                                        
                                                    #----------------------
                                                    # Test  groupmates:
                                                    if(testing_compliance_onset_groupmate[groupmate]):
                                                        testingSet_onset_groupmate.add(groupmate)                                                        


                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Upon being traced as contacts of positive cases:
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    tracingCohort = tracingQueue.pop(0)
                    if(len(tracingCohort) > 0 and (any(isolation_compliance_traced) or any(testing_compliance_traced))):
                        for tracedIndividual in tracingCohort:
                            #---------------------------------------------
                            # Isolate individual upon being traced:
                            #---------------------------------------------
                            if(isolation_compliance_traced[tracedIndividual]):
                                isolationSet_traced.add(tracedIndividual)
                            #---------------------------------------------
                            # Test individual upon being traced:
                            #---------------------------------------------
                            if(testing_compliance_traced[tracedIndividual]):
                                testingSet_traced.add(tracedIndividual)

                    
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Select individuals for proactive testing (on cadence days): 
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    if(any(current_cadence_time in individual_times for individual_times in proactiveTestingTimes)):
                        if(any(testing_compliance_proactive)):
                            #---------------------------------------------
                            # Include in the proactive testing pool individuals that meet the following criteria:
                            #---------------------------------------------
                            proactiveTestingPool = np.argwhere( #Proactive testing scheduled at this time:
                                                                (np.array([current_cadence_time in individual_times for individual_times in proactiveTestingTimes]))
                                                                #Compliant with proactive testing:
                                                                & (testing_compliance_proactive==True)
                                                                # Not excluded by compartment, flags, num tests, or num vaccine doses:
                                                                & (testing_excluded==False)
                                                              ).flatten()
                            #---------------------------------------------
                            # Distribute proactive tests randomly
                            #---------------------------------------------
                            numRandomTests = min( int(self.pop_size*testing_capacity_proactive), len(proactiveTestingPool))
                            if(numRandomTests > 0):
                                testingSet_proactive = set(np.random.choice(proactiveTestingPool, numRandomTests, replace=False))


                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Select individuals for vaccination (on cadence days): 
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # for individuals that meet criteria for vaccination:
                        #   add to vaccination queue


                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Execute testing:
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    testingQueue_onset.append(testingSet_onset)
                    testingQueue_onset_groupmate.append(testingSet_onset_groupmate)
                    testingQueue_traced.append(testingSet_traced)
                    testingQueue_proactive.append(testingSet_proactive)

                    testedIndividuals   = set()
                    positiveIndividuals = set()

                    #.............................................
                    # Define how positive test results will be responded to:
                    #.............................................
                    def handle_positive_result(positive_individual):
                        #.............................................
                        # Isolate individual upon positive test result:
                        #.............................................
                        if(isolation_compliance_positive[positive_individual]):
                            isolationSet_positive.add(positive_individual)
                        #.............................................
                        # Isolate and/or Test groupmates of individuals with positive test result:
                        #.............................................
                        if(intervention_groups is not None and any(isolation_compliance_positive_groupmate)):
                            groupmates = next((group for group in intervention_groups if positive_individual in group), None)
                            if(groupmates is not None):
                                for groupmate in groupmates:
                                    if(groupmate != positive_individual):
                                        #----------------------
                                        # Isolate groupmates:
                                        if(isolation_compliance_positive_groupmate[groupmate]):
                                            isolationSet_positive_groupmate.add(groupmate)
                                        #----------------------
                                        # Test groupmates:
                                        if(testing_compliance_positive_groupmate[groupmate]):
                                            testingSet_positive_groupmate.add(groupmate)
                        #.............................................
                        # Trace contacts of individuals with positive test result:
                        #.............................................
                        if(tracing_compliance[positive_individual] and (any(isolation_compliance_traced) or any(testing_compliance_traced))):
                            contactsOfPositive = set()
                            for netID, network_data in self.networks.items():
                                contactsOfPositive.update( list(network_data['networkx'][positive_individual].keys()) )
                            contactsOfPositive = list(contactsOfPositive)
                            #.................
                            numTracedContacts  = tracing_num_contacts if tracing_num_contacts is not None else int(len(contactsOfPositive)*tracing_pct_contacts)
                            if(len(contactsOfPositive) > 0 and numTracedContacts > 0):
                                tracedContacts = np.random.choice(contactsOfPositive, numTracedContacts, replace=False)
                                tracingSet.update(tracedContacts)                        
                    #.............................................

                    #---------------------------------------------
                    # Administer onset tests:
                    #---------------------------------------------
                    numTested_onset   = 0
                    numPositive_onset = 0
                    testingCohort_onset = (testingQueue_onset.pop(0) & testing_nonExcludedIndividuals)
                    for testIndividual in testingCohort_onset:
                        if(len(testedIndividuals) >= self.pop_size*testing_capacity_max):
                            break
                        if(testIndividual not in testedIndividuals):
                            testResult = self.test(testIndividual, test_type_onset)
                            numTested_onset += 1
                            testedIndividuals.add(testIndividual)
                            if(testResult == True):
                                positiveIndividuals.add(testIndividual)
                                handle_positive_result(testIndividual)
                                numPositive_onset += 1
                    #---------------------------------------------
                    # Administer onset groupmate tests:
                    #---------------------------------------------
                    numTested_onset_groupmate   = 0
                    numPositive_onset_groupmate = 0
                    testingCohort_onset_groupmate = (testingQueue_onset_groupmate.pop(0) & testing_nonExcludedIndividuals)
                    for testIndividual in testingCohort_onset_groupmate:
                        if(len(testedIndividuals) >= self.pop_size*testing_capacity_max):
                            break
                        if(testIndividual not in testedIndividuals):
                            testResult = self.test(testIndividual, test_type_onset_groupmate)
                            numTested_onset_groupmate += 1
                            testedIndividuals.add(testIndividual)
                            if(testResult == True):
                                positiveIndividuals.add(testIndividual)
                                handle_positive_result(testIndividual)
                                numPositive_onset_groupmate += 1
                    #---------------------------------------------
                    # Administer positive groupmate tests:
                    #---------------------------------------------
                    numTested_positive_groupmate   = 0
                    numPositive_positive_groupmate = 0
                    testingCohort_positive_groupmate = (testingQueue_positive_groupmate.pop(0) & testing_nonExcludedIndividuals)
                    for testIndividual in testingCohort_positive_groupmate:
                        if(len(testedIndividuals) >= self.pop_size*testing_capacity_max):
                            break
                        if(testIndividual not in testedIndividuals):
                            testResult = self.test(testIndividual, test_type_positive_groupmate)
                            numTested_positive_groupmate += 1
                            testedIndividuals.add(testIndividual)
                            if(testResult == True):
                                positiveIndividuals.add(testIndividual)
                                handle_positive_result(testIndividual)
                                numPositive_positive_groupmate += 1
                    #---------------------------------------------
                    # Administer tracing tests:
                    #---------------------------------------------
                    numTested_traced   = 0
                    numPositive_traced = 0
                    testingCohort_traced = (testingQueue_traced.pop(0) & testing_nonExcludedIndividuals)
                    for testIndividual in testingCohort_traced:
                        if(len(testedIndividuals) >= self.pop_size*testing_capacity_max):
                            break
                        if(testIndividual not in testedIndividuals):
                            testResult = self.test(testIndividual, test_type_traced)
                            numTested_traced += 1
                            testedIndividuals.add(testIndividual)
                            if(testResult == True):
                                positiveIndividuals.add(testIndividual)
                                handle_positive_result(testIndividual)
                                numPositive_traced += 1
                    #---------------------------------------------
                    # Administer proactive tests:
                    #---------------------------------------------
                    numTested_proactive   = 0
                    numPositive_proactive = 0
                    testingCohort_proactive = (testingQueue_proactive.pop(0) & testing_nonExcludedIndividuals)
                    for testIndividual in testingCohort_proactive:
                        if(len(testedIndividuals) >= self.pop_size*testing_capacity_max):
                            break
                        if(testIndividual not in testedIndividuals):
                            testResult = self.test(testIndividual, test_type_proactive)
                            numTested_proactive += 1
                            testedIndividuals.add(testIndividual)
                            if(testResult == True):
                                positiveIndividuals.add(testIndividual)
                                handle_positive_result(testIndividual)
                                numPositive_proactive += 1
                    #---------------------------------------------
                    # After all positive test results have been handled...
                    #   Add groupmates and/or traced contacts of positives identified in this step to the queue:
                    #   (testing traced contacts / positive groupmates must have at least 1 cadence_dt delay)
                    testingQueue_positive_groupmate.append(testingSet_positive_groupmate)
                    tracingQueue.append(tracingSet)


                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Execute vaccination:
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # for individual being vaccinated:
                        #   call vaccinate()


                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Execute isolation:
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    isolationQueue_onset.append(isolationSet_onset)
                    isolationQueue_onset_groupmate.append(isolationSet_onset_groupmate)
                    isolationQueue_positive.append(isolationSet_positive)
                    isolationQueue_positive_groupmate.append(isolationSet_positive_groupmate)
                    isolationQueue_traced.append(isolationSet_traced)

                    isolationCohort_onset              = (isolationQueue_onset.pop(0) & isolation_nonExcludedIndividuals)
                    isolationCohort_onset_groupmate    = (isolationQueue_onset_groupmate.pop(0) & isolation_nonExcludedIndividuals)
                    isolationCohort_positive           = (isolationQueue_positive.pop(0) & isolation_nonExcludedIndividuals)
                    isolationCohort_positive_groupmate = (isolationQueue_positive_groupmate.pop(0) & isolation_nonExcludedIndividuals)
                    isolationCohort_traced             = (isolationQueue_traced.pop(0) & isolation_nonExcludedIndividuals)
                    isolationCohort = (isolationCohort_onset | isolationCohort_onset_groupmate | isolationCohort_positive | isolationCohort_positive_groupmate | isolationCohort_traced) 

                    for isoIndividual in isolationCohort:
                        self.set_isolation(isoIndividual, True)


                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    totalNumTests_proactive               += numTested_proactive
                    totalNumTests_onset                   += numTested_onset
                    totalNumTests_onset_groupmate         += numTested_onset_groupmate
                    totalNumTests_positive_groupmate      += numTested_positive_groupmate
                    totalNumTests_traced                  += numTested_traced
                    totalNumTests                         += len(testedIndividuals)
                    totalNumPositives_proactive           += numPositive_proactive
                    totalNumPositives_onset               += numPositive_onset
                    totalNumPositives_onset_groupmate     += numPositive_onset_groupmate
                    totalNumPositives_positive_groupmate  += numPositive_positive_groupmate
                    totalNumPositives_traced              += numPositive_traced
                    totalNumPositives                     += len(positiveIndividuals)
                    totalNumIsolations_onset              += len(isolationCohort_onset)
                    totalNumIsolations_onset_groupmate    += len(isolationCohort_onset_groupmate)
                    totalNumIsolations_positive           += len(isolationCohort_positive)
                    totalNumIsolations_positive_groupmate += len(isolationCohort_positive_groupmate)
                    totalNumIsolations_traced             += len(isolationCohort_traced)
                    totalNumIsolations                    += len(isolationCohort)

                    print("\t"+str(numTested_proactive)          +"\ttested proactively                     [+ "+str(numPositive_proactive)+" positive (%.2f %%) +]" % (numPositive_proactive/numTested_proactive*100 if numTested_proactive>0 else 0))
                    print("\t"+str(numTested_onset)              +"\ttested "+str(testing_delay_onset)+" days after onset              [+ "+str(numPositive_onset)+" positive (%.2f %%) +]" % (numPositive_onset/numTested_onset*100 if numTested_onset>0 else 0))                    
                    print("\t"+str(numTested_onset_groupmate)    +"\ttested "+str(testing_delay_onset_groupmate)+" days after groupmate onset    [+ "+str(numPositive_onset_groupmate)+" positive (%.2f %%) +]" % (numPositive_onset_groupmate/numTested_onset_groupmate*100 if numTested_onset_groupmate>0 else 0))
                    print("\t"+str(numTested_positive_groupmate) +"\ttested "+str(testing_delay_positive_groupmate)+" days after groupmate positive [+ "+str(numPositive_positive_groupmate)+" positive (%.2f %%) +]" % (numPositive_positive_groupmate/numTested_positive_groupmate*100 if numTested_positive_groupmate>0 else 0))
                    print("\t"+str(numTested_traced)             +"\ttested "+str(testing_delay_traced)+" days after being traced       [+ "+str(numPositive_traced)+" positive (%.2f %%) +]" % (numPositive_traced/numTested_traced*100 if numTested_traced>0 else 0))
                    print("\t"+str(len(testedIndividuals))       +"\tTESTED TOTAL                           [+ "+str(len(positiveIndividuals))+" positive (%.2f %%) +]" % (len(positiveIndividuals)/len(testedIndividuals)*100 if len(testedIndividuals)>0 else 0))

                    print("\t"+str(len(isolationCohort_onset))              +"\tisolated "+str(isolation_delay_onset)+" days after onset")
                    print("\t"+str(len(isolationCohort_onset_groupmate))    +"\tisolated "+str(isolation_delay_onset_groupmate)+" days after groupmate onset")
                    print("\t"+str(len(isolationCohort_positive))           +"\tisolated "+str(isolation_delay_positive)+" days after positive")
                    print("\t"+str(len(isolationCohort_positive_groupmate)) +"\tisolated "+str(isolation_delay_positive_groupmate)+" days after groupmate positive")
                    print("\t"+str(len(isolationCohort_traced))             +"\tisolated "+str(isolation_delay_traced)+" days after traced")
                    print("\t"+str(len(isolationCohort))                    +"\tISOLATED TOTAL")
                    

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
            running = self.run_iteration(max_dt=max_dt)
            if(run_full_duration):
                running = self.t < T

            # while loop
            #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Finalize model and simulation data:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.finalize_data_series()

        sim_results = { 'sim_duration':                             self.t,
                        # 'active_outbreak_duration':                 activeOutbreakDuration,
                        'intervention_duration':                    self.t - interventionStartTime,
                        'intervention_start_time':                  interventionStartTime,
                        'intervention_end_time':                    self.t,
                        'init_cadence_offset':                      init_cadence_offset,
                        # 'total_num_introductions':                totalNumIntroductions,
                        'total_num_tests_proactive':                totalNumTests_proactive,              
                        'total_num_tests_onset':                    totalNumTests_onset,                  
                        'total_num_tests_groupmate':                totalNumTests_onset_groupmate,        
                        'total_num_tests_positive_groupmate':       totalNumTests_positive_groupmate,     
                        'total_num_tests_traced':                   totalNumTests_traced,                 
                        'total_num_tests':                          totalNumTests,                        
                        'total_num_positives_proactive':            totalNumPositives_proactive,          
                        'total_num_positives_onset':                totalNumPositives_onset,              
                        'total_num_positives_onset_groupmate':      totalNumPositives_onset_groupmate,    
                        'total_num_positives_positive_groupmate':   totalNumPositives_positive_groupmate, 
                        'total_num_positives_traced':               totalNumPositives_traced,             
                        'total_num_positives':                      totalNumPositives,                    
                        'total_num_isolations_onset':               totalNumIsolations_onset,             
                        'total_num_isolations_onset_groupmate':     totalNumIsolations_onset_groupmate,   
                        'total_num_isolations_positive':            totalNumIsolations_positive,          
                        'total_num_isolations_positive_groupmate':  totalNumIsolations_positive_groupmate,
                        'total_num_isolations_traced':              totalNumIsolations_traced,            
                        'total_num_isolations':                     totalNumIsolations }

        print(sim_results)

        #---------------------------------------------

        return sim_results

        