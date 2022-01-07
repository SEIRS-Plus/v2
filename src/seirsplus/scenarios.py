import numpy as np

from seirsplus import utils

import time


def run_interventions_scenario(model, T, max_dt=0.1, default_dt=0.1, tau_step=None,
                                    terminate_at_zero_cases=False,
                                    # Intervention timing params:
                                    cadence_dt=1, 
                                    cadence_cycle_length=28,
                                    init_cadence_offset=0,
                                    cadence_presets='default',
                                    intervention_start_time=0,
                                    intervention_start_prevalence=0,
                                    prevalence_flags=['infected'],
                                    case_introduction_rate=0,
                                    # State onset intervention params:
                                    onset_compartments=[], # not yet used
                                    onset_flags=[], 
                                    # Network change params:
                                    network_active_cadences=None,
                                    # Isolation params:
                                    isolation_period=None,
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
                                    test_type_deisolation=None, 
                                    test_result_delay=0,
                                    proactive_testing_cadence='never',
                                    proactive_testing_synchronize=True,
                                    num_deisolation_tests=1,
                                    testing_capacity_max=1.0,
                                    testing_capacity_proactive=0.0,
                                    testing_delay_proactive=0,
                                    testing_delay_onset=1,
                                    testing_delay_onset_groupmate=1,
                                    testing_delay_positive_groupmate=1,
                                    testing_delay_traced=1, 
                                    testing_delay_deisolation=5,                                   
                                    testing_compliance_proactive=1.0,
                                    testing_compliance_onset=0.0, 
                                    testing_compliance_onset_groupmate=0.0,
                                    testing_compliance_positive_groupmate=0.0,
                                    testing_compliance_traced=0.0,
                                    testing_compliance_deisolation=0.0,
                                    testing_exclude_compartments=[],
                                    testing_exclude_flags=[],
                                    testing_exclude_isolated=0.0,
                                    testing_exclude_afterNumTests=None,
                                    testing_exclude_afterNumVaccineDoses=None,
                                    # Tracing params:                                                                       
                                    tracing_num_contacts=None, 
                                    tracing_pct_contacts=0,
                                    tracing_delay=1,
                                    tracing_compliance=1.0,
                                    # Misc. params:
                                    intervention_groups=None
                                ):

        if(T>0):
            model.tmax += T
        else:
            return False

        if(max_dt is None and model.transition_mode=='time_in_state'):
            max_dt = cadence_dt

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize intervention parameters:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #----------------------------------------
        # Initialize intervention-related model parameters:
        #----------------------------------------
        model.num_tests         = np.zeros(model.pop_size)
        model.num_vaccine_doses = np.zeros(model.pop_size)

        #----------------------------------------
        # Initialize cadence and intervention time parameters:
        #----------------------------------------
        interventionOn = False
        interventionStartTime = None

        # Cadences involve a repeating (default 28 day) cycle starting on a Monday
        # (0:Mon, 1:Tue, 2:Wed, 3:Thu, 4:Fri, 5:Sat, 6:Sun, 7:Mon, 8:Tues, ...)
        # For each cadence, actions are done on the cadence intervals included in the associated list.
        if(cadence_presets == 'default'):
            cadence_presets   = {
                                    'semidaily':  [int(d) for d in np.arange(start=0, stop=cadence_cycle_length, step=0.5)],
                                    'daily':      [int(d) for d in np.arange(start=0, stop=cadence_cycle_length, step=1)],
                                    'nightly':    [int(d) for d in np.arange(start=0.5, stop=cadence_cycle_length, step=1)],
                                    'weekday':    [int(d) for d in np.arange(start=0, stop=cadence_cycle_length, step=1) if (d%7!=5 and d%7!=6)],
                                    'weekend':    [int(d) for d in np.arange(start=0, stop=cadence_cycle_length, step=1) if (d%7==5 or d%7==6)],
                                    '3x-weekly':  [int(d) for d in np.arange(start=0, stop=cadence_cycle_length, step=7/3)],
                                    '2x-weekly':  [int(d) for d in np.arange(start=0, stop=cadence_cycle_length, step=7/2)],
                                    'semiweekly': [int(d) for d in np.arange(start=0, stop=cadence_cycle_length, step=7/2)],
                                    'weekly':     [int(d) for d in np.arange(start=0, stop=cadence_cycle_length, step=7/1)],
                                    'biweekly':   [int(d) for d in np.arange(start=0, stop=cadence_cycle_length, step=7*2)],
                                    'monthly':    [int(d) for d in np.arange(start=0, stop=cadence_cycle_length, step=7*4)],
                                    'initial':    [0],
                                    'never':      []
                                }

        if(init_cadence_offset == 'random'):
            init_cadence_offset = np.random.choice(range(cadence_cycle_length))

        last_cadence_time  = -1

        #----------------------------------------
        # Initialize network activity parameters:
        #----------------------------------------
        if(network_active_cadences is not None):
            networkActiveTimes = {network: ([cadence_presets[individual_cadence] for individual_cadence in network_active_cadences] 
                                                if isinstance(network_active_cadences, (list, np.ndarray)) 
                                                else [cadence_presets[network_active_cadences[network]]]*model.pop_size) 
                                            for network in model.networks}
        
        #----------------------------------------
        # Initialize onset parameters:
        #----------------------------------------
        onset_flags = [onset_flags] if not isinstance(onset_flags, (list, np.ndarray)) else onset_flags

        flag_onset = {flag: [False]*model.pop_size for flag in onset_flags} # bools for tracking which onsets have triggered for each individual
        
        #----------------------------------------
        # Initialize testing parameters:
        #----------------------------------------
        model.update_test_parameters(test_params, prevalence_flags=prevalence_flags)

        test_type_onset       = test_type_onset if test_type_onset is not None else list(model.test_types)[0] if len(model.test_types)>0 else None
        test_type_traced      = test_type_traced if test_type_traced is not None else list(model.test_types)[0] if len(model.test_types)>0 else None
        test_type_deisolation = test_type_deisolation if test_type_deisolation is not None else list(model.test_types)[0] if len(model.test_types)>0 else None
        test_type_proactive   = test_type_proactive if test_type_proactive is not None else list(model.test_types)[0] if len(model.test_types)>0 else None

        test_result_delay     = {test_type: test_result_delay for test_type in model.test_types} if not isinstance(test_result_delay, dict) else test_result_delay

        proactiveTestingTimes = [cadence_presets[individual_cadence] for individual_cadence in proactive_testing_cadence] if isinstance(proactive_testing_cadence, (list, np.ndarray)) else [cadence_presets[proactive_testing_cadence]]*model.pop_size
        if(not proactive_testing_synchronize):
            print("scatter")
            for i, individualCadence in enumerate(proactiveTestingTimes):
                proactiveTestingTimes[i] = np.sort(np.fmod(individualCadence + np.random.choice(range(cadence_cycle_length)), cadence_cycle_length))

        #----------------------------------------
        # Initialize individual compliances:
        #----------------------------------------
        isolation_compliance_onset              = utils.param_as_bool_array(isolation_compliance_onset, (1, model.pop_size)).flatten()
        isolation_compliance_onset_groupmate    = utils.param_as_bool_array(isolation_compliance_onset_groupmate, (1, model.pop_size)).flatten()
        isolation_compliance_positive           = utils.param_as_bool_array(isolation_compliance_positive, (1, model.pop_size)).flatten()
        isolation_compliance_positive_groupmate = utils.param_as_bool_array(isolation_compliance_positive_groupmate, (1, model.pop_size)).flatten()
        isolation_compliance_traced             = utils.param_as_bool_array(isolation_compliance_traced, (1, model.pop_size)).flatten()
        testing_compliance_proactive            = utils.param_as_bool_array(testing_compliance_proactive, (1, model.pop_size)).flatten()
        testing_compliance_onset                = utils.param_as_bool_array(testing_compliance_onset, (1, model.pop_size)).flatten()
        testing_compliance_onset_groupmate      = utils.param_as_bool_array(testing_compliance_onset_groupmate, (1, model.pop_size)).flatten()
        testing_compliance_positive_groupmate   = utils.param_as_bool_array(testing_compliance_positive_groupmate, (1, model.pop_size)).flatten()
        testing_compliance_traced               = utils.param_as_bool_array(testing_compliance_traced, (1, model.pop_size)).flatten()
        testing_compliance_deisolation          = utils.param_as_bool_array(testing_compliance_deisolation, (1, model.pop_size)).flatten()
        tracing_compliance                      = utils.param_as_bool_array(tracing_compliance, (1, model.pop_size)).flatten()
        # Store compliances as node attributes in the model object (e.g., for case logging purposes)
        model.set_node_attribute(node=list(range(model.pop_size)), attribute_name='isolation_compliance_onset', attribute_value=isolation_compliance_onset)
        model.set_node_attribute(node=list(range(model.pop_size)), attribute_name='isolation_compliance_onset_groupmate', attribute_value=isolation_compliance_onset_groupmate)
        model.set_node_attribute(node=list(range(model.pop_size)), attribute_name='isolation_compliance_positive', attribute_value=isolation_compliance_positive)
        model.set_node_attribute(node=list(range(model.pop_size)), attribute_name='isolation_compliance_positive_groupmate', attribute_value=isolation_compliance_positive_groupmate)
        model.set_node_attribute(node=list(range(model.pop_size)), attribute_name='isolation_compliance_traced', attribute_value=isolation_compliance_traced)
        model.set_node_attribute(node=list(range(model.pop_size)), attribute_name='testing_compliance_proactive', attribute_value=testing_compliance_proactive)
        model.set_node_attribute(node=list(range(model.pop_size)), attribute_name='testing_compliance_onset', attribute_value=testing_compliance_onset)
        model.set_node_attribute(node=list(range(model.pop_size)), attribute_name='testing_compliance_onset_groupmate', attribute_value=testing_compliance_onset_groupmate)
        model.set_node_attribute(node=list(range(model.pop_size)), attribute_name='testing_compliance_positive_groupmate', attribute_value=testing_compliance_positive_groupmate)
        model.set_node_attribute(node=list(range(model.pop_size)), attribute_name='testing_compliance_traced', attribute_value=testing_compliance_traced)
        model.set_node_attribute(node=list(range(model.pop_size)), attribute_name='testing_compliance_deisolation', attribute_value=testing_compliance_deisolation)
        model.set_node_attribute(node=list(range(model.pop_size)), attribute_name='tracing_compliance', attribute_value=tracing_compliance)

        #----------------------------------------
        # Initialize intervention exclusion criteria:
        #----------------------------------------
        isolation_exclude_afterNumTests        = np.inf if isolation_exclude_afterNumTests is None else isolation_exclude_afterNumTests
        isolation_exclude_afterNumVaccineDoses = np.inf if isolation_exclude_afterNumVaccineDoses is None else isolation_exclude_afterNumVaccineDoses
        testing_exclude_afterNumTests          = np.inf if testing_exclude_afterNumTests is None else testing_exclude_afterNumTests
        testing_exclude_afterNumVaccineDoses   = np.inf if testing_exclude_afterNumVaccineDoses is None else testing_exclude_afterNumVaccineDoses
        testing_exclude_compartments           = [model.stateID[c] for c in testing_exclude_compartments]

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
        testingQueue_deisolation                = [set() for i in range(max(1, (int(testing_delay_deisolation/cadence_dt) + (1 if np.fmod(testing_delay_deisolation, cadence_dt)>0 else 0))))]
        testingQueue_proactive                  = [set() for i in range(int(testing_delay_proactive/cadence_dt) + (1 if np.fmod(testing_delay_proactive, cadence_dt)>0 else 0))]
        tracingQueue                            = [set() for i in range(int(tracing_delay/cadence_dt) + (1 if np.fmod(tracing_delay, cadence_dt)>0 else 0))]

        positiveResultQueue                     = {test_type: [set() for i in range(int(test_result_delay[test_type]/cadence_dt) + (1 if np.fmod(test_result_delay[test_type], cadence_dt)>0 else 0))]
                                                    for test_type in model.test_types}

        #----------------------------------------
        # Initialize intervention stats:
        #----------------------------------------
        totalNumTested_proactive              = 0
        totalNumTested_onset                  = 0
        totalNumTested_onset_groupmate        = 0
        totalNumTested_positive_groupmate     = 0
        totalNumTested_traced                 = 0
        totalNumTested_deisolation            = 0
        totalNumTested                        = 0
        totalNumPositives_proactive           = 0
        totalNumPositives_onset               = 0
        totalNumPositives_onset_groupmate     = 0
        totalNumPositives_positive_groupmate  = 0
        totalNumPositives_traced              = 0
        totalNumPositives_deisolation         = 0
        totalNumPositives                     = 0
        totalNumTruePositives                 = 0
        totalNumFalsePositives                = 0
        totalNumTrueNegatives                 = 0
        totalNumFalseNegatives                = 0
        totalNumIsolations_onset              = 0
        totalNumIsolations_onset_groupmate    = 0
        totalNumIsolations_positive           = 0
        totalNumIsolations_positive_groupmate = 0
        totalNumIsolations_traced             = 0
        totalNumIsolations                    = 0
        totalNumDeisolations                  = 0
        totalNumIntroductions                 = 0
        peakNumIsolated                       = 0

        #----------------------------------------
        # Initialize other metadata:
        #----------------------------------------
        individual_testing_times = [ [] for i in range(model.pop_size) ]
        individual_tracing_times = [ [] for i in range(model.pop_size) ]

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Run the simulation loop:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        running     = True
        while running: 
            # time_scenaro_start = time.time()

            current_cadence_time = ((model.t + init_cadence_offset) - np.fmod((model.t + init_cadence_offset), cadence_dt)) % (cadence_cycle_length - np.fmod(cadence_cycle_length, cadence_dt))
            if(current_cadence_time != last_cadence_time):

                last_cadence_time = current_cadence_time

                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Introduce exogenous cases randomly:
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                numNewExposures   = np.random.poisson(lam=case_introduction_rate*cadence_dt)
                if(numNewExposures > 0):
                    introductionNodes = model.introduce_random_exposures(numNewExposures, compartment='all', exposed_to='any')
                    if(len(introductionNodes) > 0):
                        print("[NEW INTRODUCTION @ t = %.2f (%d exposed)]" % (model.t, len(introductionNodes)))
                    totalNumIntroductions += len(introductionNodes)

                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Update network activities:
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if(network_active_cadences is not None):
                    for network, active_cadence in network_active_cadences.items():
                        activeIndividuals   = np.argwhere( np.array([current_cadence_time in individual_times for individual_times in networkActiveTimes[network]]) )
                        inactiveIndividuals = np.argwhere( np.array([current_cadence_time not in individual_times for individual_times in networkActiveTimes[network]]) )
                        model.set_network_activity(network, node=activeIndividuals, active=True)
                        model.set_network_activity(network, node=inactiveIndividuals, active=False)

                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                currentNumInfected = model.get_count_by_flag(prevalence_flags)
                currentPrevalence  = currentNumInfected/model.N[model.tidx]
                currentNumIsolated = np.count_nonzero(model.isolation)

                if(currentPrevalence >= intervention_start_prevalence and not interventionOn):
                    interventionOn        = True
                    interventionStartTime = model.t
                
                if(interventionOn):

                    print("[SCENARIO @ t = %.2f (t_cadence ~%.2f) :: Currently %d infected (%.2f%%), %d isolated]" % (model.t, current_cadence_time, currentNumInfected, ((currentNumInfected)/model.N[model.tidx])*100, currentNumIsolated))
                    print(" State counts: ", list(zip([model.get_compartment_by_state_id(sid) for sid in np.unique(model.X, return_counts=True)[0]], np.unique(model.X, return_counts=True)[-1])))

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
                    isolation_excluded_byFlags        = (np.isin(range(model.pop_size), model.get_individuals_by_flag(isolation_exclude_flags))).flatten()
                    isolation_excluded_byCompartments = (np.isin(model.X, isolation_exclude_compartments)).flatten()
                    isolation_excluded_byIsolation    = (model.isolation == True).flatten() if isolation_exclude_isolated else np.array([False]*model.pop_size)
                    isolation_excluded_byNumTests     = (model.num_tests >= isolation_exclude_afterNumTests).flatten()
                    isolation_excluded_byVaccineDoses = (model.num_vaccine_doses >= isolation_exclude_afterNumVaccineDoses).flatten()
                    
                    isolation_excluded                = (isolation_excluded_byFlags | isolation_excluded_byCompartments | isolation_excluded_byIsolation | isolation_excluded_byNumTests | isolation_excluded_byVaccineDoses)

                    isolation_nonExcludedIndividuals  = set(np.argwhere(isolation_excluded==False).flatten())

                    #---------------------------------------------
                    # Exclude the following individuals from testing:
                    # (these lists referenced in proactive testing selection and testing execution below)
                    #---------------------------------------------
                    testing_excluded_byFlags        = (np.isin(range(model.pop_size), model.get_individuals_by_flag(testing_exclude_flags))).flatten()
                    testing_excluded_byCompartments = (np.isin(model.X, testing_exclude_compartments)).flatten()
                    testing_excluded_byIsolation    = (model.isolation == True).flatten() if testing_exclude_isolated else np.array([False]*model.pop_size)
                    testing_excluded_byNumTests     = (model.num_tests >= testing_exclude_afterNumTests).flatten()
                    testing_excluded_byVaccineDoses = (model.num_vaccine_doses >= testing_exclude_afterNumVaccineDoses).flatten()
                    
                    testing_excluded                = (testing_excluded_byFlags | testing_excluded_byCompartments | testing_excluded_byIsolation | testing_excluded_byNumTests | testing_excluded_byVaccineDoses)

                    testing_nonExcludedIndividuals  = set(np.argwhere(testing_excluded==False).flatten())

                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Upon onset of flagged state (e.g., symptoms):
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    if(any(isolation_compliance_onset) or any(testing_compliance_onset)
                       or (intervention_groups is not None and (any(isolation_compliance_onset_groupmate) or any(testing_compliance_onset_groupmate)))):
                        for isoflag in onset_flags:
                            for flaggedIndividual in model.get_individuals_by_flag(isoflag):
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
                    if(len(tracingQueue) > 0):
                        tracingCohort = tracingQueue.pop(0)
                        if(len(tracingCohort) > 0 and (any(isolation_compliance_traced) or any(testing_compliance_traced))):
                            for tracedIndividual in tracingCohort:
                                individual_tracing_times[tracedIndividual].append(model.t)
                                model.set_node_attribute(node=tracedIndividual, attribute_name='num_traces', attribute_value=len(individual_tracing_times[tracedIndividual]))
                                model.set_node_attribute(node=tracedIndividual, attribute_name='time_of_last_trace', attribute_value=individual_tracing_times[tracedingIndividual][-1])
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
                            numRandomTests = min( int(model.pop_size*testing_capacity_proactive), len(proactiveTestingPool))
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

                    positiveResultSet   = {test_type: set() for test_type in model.test_types}

                    testedIndividuals   = set()
                    positiveIndividuals = set()

                    #---------------------------------------------
                    # Administer onset tests:
                    #---------------------------------------------
                    numTested_onset   = 0
                    numPositive_onset = 0
                    testingCohort_onset = (testingQueue_onset.pop(0) & testing_nonExcludedIndividuals)
                    for testIndividual in testingCohort_onset:
                        if(len(testedIndividuals) >= model.pop_size*testing_capacity_max):
                            break
                        if(testIndividual not in testedIndividuals):
                            testResult, testValidity = model.test(testIndividual, test_type_onset)
                            numTested_onset += 1
                            testedIndividuals.add(testIndividual)
                            individual_testing_times[testIndividual].append(model.t)
                            model.set_node_attribute(node=testIndividual, attribute_name='num_tests', attribute_value=len(individual_testing_times[testIndividual]))
                            model.set_node_attribute(node=testIndividual, attribute_name='time_of_last_test', attribute_value=individual_testing_times[testIndividual][-1])
                            model.set_node_attribute(node=testIndividual, attribute_name='result_of_last_test', attribute_value=testResult)
                            model.set_node_attribute(node=testIndividual, attribute_name='validity_of_last_test', attribute_value=testValidity)
                            if(testResult == True):
                                positiveIndividuals.add(testIndividual)
                                positiveResultSet[test_type_onset].add(testIndividual)
                                numPositive_onset += 1
                            if(testResult==True and testValidity==True):     totalNumTruePositives += 1
                            elif(testResult==True and testValidity==False):  totalNumFalsePositives += 1
                            elif(testResult==False and testValidity==True):  totalNumTrueNegatives += 1
                            elif(testResult==False and testValidity==False): totalNumFalseNegatives += 1
                    #---------------------------------------------
                    # Administer onset groupmate tests:
                    #---------------------------------------------
                    numTested_onset_groupmate   = 0
                    numPositive_onset_groupmate = 0
                    testingCohort_onset_groupmate = (testingQueue_onset_groupmate.pop(0) & testing_nonExcludedIndividuals)
                    for testIndividual in testingCohort_onset_groupmate:
                        if(len(testedIndividuals) >= model.pop_size*testing_capacity_max):
                            break
                        if(testIndividual not in testedIndividuals):
                            testResult, testValidity = model.test(testIndividual, test_type_onset_groupmate)
                            numTested_onset_groupmate += 1
                            testedIndividuals.add(testIndividual)
                            individual_testing_times[testIndividual].append(model.t)
                            model.set_node_attribute(node=testIndividual, attribute_name='num_tests', attribute_value=len(individual_testing_times[testIndividual]))
                            model.set_node_attribute(node=testIndividual, attribute_name='time_of_last_test', attribute_value=individual_testing_times[testIndividual][-1])
                            model.set_node_attribute(node=testIndividual, attribute_name='result_of_last_test', attribute_value=testResult)
                            model.set_node_attribute(node=testIndividual, attribute_name='validity_of_last_test', attribute_value=testValidity)
                            if(testResult == True):
                                positiveIndividuals.add(testIndividual)
                                positiveResultSet[test_type_onset_groupmate].add(testIndividual)
                                numPositive_onset_groupmate += 1
                            if(testResult==True and testValidity==True):     totalNumTruePositives += 1
                            elif(testResult==True and testValidity==False):  totalNumFalsePositives += 1
                            elif(testResult==False and testValidity==True):  totalNumTrueNegatives += 1
                            elif(testResult==False and testValidity==False): totalNumFalseNegatives += 1
                    #---------------------------------------------
                    # Administer positive groupmate tests:
                    #---------------------------------------------
                    numTested_positive_groupmate   = 0
                    numPositive_positive_groupmate = 0
                    testingCohort_positive_groupmate = (testingQueue_positive_groupmate.pop(0) & testing_nonExcludedIndividuals)
                    for testIndividual in testingCohort_positive_groupmate:
                        if(len(testedIndividuals) >= model.pop_size*testing_capacity_max):
                            break
                        if(testIndividual not in testedIndividuals):
                            testResult, testValidity = model.test(testIndividual, test_type_positive_groupmate)
                            numTested_positive_groupmate += 1
                            testedIndividuals.add(testIndividual)
                            individual_testing_times[testIndividual].append(model.t)
                            model.set_node_attribute(node=testIndividual, attribute_name='num_tests', attribute_value=len(individual_testing_times[testIndividual]))
                            model.set_node_attribute(node=testIndividual, attribute_name='time_of_last_test', attribute_value=individual_testing_times[testIndividual][-1])
                            model.set_node_attribute(node=testIndividual, attribute_name='result_of_last_test', attribute_value=testResult)
                            model.set_node_attribute(node=testIndividual, attribute_name='validity_of_last_test', attribute_value=testValidity)
                            if(testResult == True):
                                positiveIndividuals.add(testIndividual)
                                positiveResultSet[test_type_positive_groupmate].add(testIndividual)
                                numPositive_positive_groupmate += 1
                            if(testResult==True and testValidity==True):     totalNumTruePositives += 1
                            elif(testResult==True and testValidity==False):  totalNumFalsePositives += 1
                            elif(testResult==False and testValidity==True):  totalNumTrueNegatives += 1
                            elif(testResult==False and testValidity==False): totalNumFalseNegatives += 1
                    #---------------------------------------------
                    # Administer tracing tests:
                    #---------------------------------------------
                    numTested_traced   = 0
                    numPositive_traced = 0
                    if(len(testingQueue_traced) > 0):
                        testingCohort_traced = (testingQueue_traced.pop(0) & testing_nonExcludedIndividuals)
                        for testIndividual in testingCohort_traced:
                            if(len(testedIndividuals) >= model.pop_size*testing_capacity_max):
                                break
                            if(testIndividual not in testedIndividuals):
                                testResult, testValidity = model.test(testIndividual, test_type_traced)
                                numTested_traced += 1
                                testedIndividuals.add(testIndividual)
                                individual_testing_times[testIndividual].append(model.t)
                                model.set_node_attribute(node=testIndividual, attribute_name='num_tests', attribute_value=len(individual_testing_times[testIndividual]))
                                model.set_node_attribute(node=testIndividual, attribute_name='time_of_last_test', attribute_value=individual_testing_times[testIndividual][-1])
                                model.set_node_attribute(node=testIndividual, attribute_name='result_of_last_test', attribute_value=testResult)
                                model.set_node_attribute(node=testIndividual, attribute_name='validity_of_last_test', attribute_value=testValidity)
                                if(testResult == True):
                                    positiveIndividuals.add(testIndividual)
                                    positiveResultSet[test_type_traced].add(testIndividual)
                                    numPositive_traced += 1
                                if(testResult==True and testValidity==True):     totalNumTruePositives += 1
                                elif(testResult==True and testValidity==False):  totalNumFalsePositives += 1
                                elif(testResult==False and testValidity==True):  totalNumTrueNegatives += 1
                                elif(testResult==False and testValidity==False): totalNumFalseNegatives += 1
                    #---------------------------------------------
                    # Administer proactive tests:
                    #---------------------------------------------
                    numTested_proactive   = 0
                    numPositive_proactive = 0
                    testingCohort_proactive = (testingQueue_proactive.pop(0) & testing_nonExcludedIndividuals)
                    for testIndividual in testingCohort_proactive:
                        if(len(testedIndividuals) >= model.pop_size*testing_capacity_max):
                            break
                        if(testIndividual not in testedIndividuals):
                            testResult, testValidity = model.test(testIndividual, test_type_proactive)
                            numTested_proactive += 1
                            testedIndividuals.add(testIndividual)
                            individual_testing_times[testIndividual].append(model.t)
                            model.set_node_attribute(node=testIndividual, attribute_name='num_tests', attribute_value=len(individual_testing_times[testIndividual]))
                            model.set_node_attribute(node=testIndividual, attribute_name='time_of_last_test', attribute_value=individual_testing_times[testIndividual][-1])
                            model.set_node_attribute(node=testIndividual, attribute_name='result_of_last_test', attribute_value=testResult)
                            model.set_node_attribute(node=testIndividual, attribute_name='validity_of_last_test', attribute_value=testValidity)
                            if(testResult == True):
                                positiveIndividuals.add(testIndividual)
                                positiveResultSet[test_type_proactive].add(testIndividual)
                                numPositive_proactive += 1
                            if(testResult==True and testValidity==True):     totalNumTruePositives += 1
                            elif(testResult==True and testValidity==False):  totalNumFalsePositives += 1
                            elif(testResult==False and testValidity==True):  totalNumTrueNegatives += 1
                            elif(testResult==False and testValidity==False): totalNumFalseNegatives += 1
                    #---------------------------------------------
                    # Administer de-isolation checkpoint tests:
                    #---------------------------------------------
                    numTested_deisolation = 0
                    numPositive_deisolation = 0
                    numDeisolating = 0
                    if(len(testingQueue_deisolation) > 0):
                        testingCohort_deisolation = (testingQueue_deisolation.pop(0)) # no testing exclusions
                        for testIndividual in testingCohort_deisolation:
                            if(len(testedIndividuals) >= model.pop_size*testing_capacity_max):
                                break
                            testedIndividuals.add(testIndividual)
                            numTested_deisolation += 1
                            numNegativeResults = 0
                            for test_rep in range(num_deisolation_tests):
                                testResult, testValidity = model.test(testIndividual, test_type_deisolation)
                                individual_testing_times[testIndividual].append(model.t)
                                model.set_node_attribute(node=testIndividual, attribute_name='num_tests', attribute_value=len(individual_testing_times[testIndividual]))
                                model.set_node_attribute(node=testIndividual, attribute_name='time_of_last_test', attribute_value=individual_testing_times[testIndividual][-1])
                                model.set_node_attribute(node=testIndividual, attribute_name='result_of_last_test', attribute_value=testResult)
                                model.set_node_attribute(node=testIndividual, attribute_name='validity_of_last_test', attribute_value=testValidity)
                                if(testResult==False):
                                    numNegativeResults += 1
                                if(testResult==True and testValidity==True):     totalNumTruePositives += 1
                                elif(testResult==True and testValidity==False):  totalNumFalsePositives += 1
                                elif(testResult==False and testValidity==True):  totalNumTrueNegatives += 1
                                elif(testResult==False and testValidity==False): totalNumFalseNegatives += 1
                            if(numNegativeResults==num_deisolation_tests):
                                # Set this individual to exit isolation:
                                model.set_isolation(testIndividual, False)
                                numDeisolating += 1
                            else:
                                numPositive_deisolation += 1
                            
                    #---------------------------------------------
                    
                    for test_type in positiveResultQueue:
                        positiveResultQueue[test_type].append(positiveResultSet[test_type])

                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Handle positive test results:
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                            for netID, network_data in model.networks.items():
                                contactsOfPositive.update( list(network_data['networkx'][positive_individual].keys()) )
                            contactsOfPositive = list(contactsOfPositive)
                            #.................
                            numTracedContacts  = tracing_num_contacts if tracing_num_contacts is not None else int(len(contactsOfPositive)*tracing_pct_contacts)
                            if(len(contactsOfPositive) > 0 and numTracedContacts > 0):
                                tracedContacts = np.random.choice(contactsOfPositive, numTracedContacts, replace=False)
                                tracingSet.update(tracedContacts)                        
                    #.............................................

                    positiveResultCohort = {}
                    for test_type in positiveResultQueue:
                        positiveResultCohort[test_type] = positiveResultQueue[test_type].pop(0)
                        for positiveIndividual in positiveResultCohort[test_type]:
                            handle_positive_result(positiveIndividual)

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

                    testingSet_deisolation = set()

                    for isoIndividual in isolationCohort:
                        # Set this individual to be in isolation:
                        model.set_isolation(isoIndividual, True, isolation_period)
                        # If compliant, put this individual in a queue for de-isolation testing:
                        if(testing_compliance_deisolation[isoIndividual]):
                            # print("putting in deiso queue", isoIndividual)
                            testingSet_deisolation.add(isoIndividual)

                    testingQueue_deisolation.append(testingSet_deisolation)



                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    totalNumTested_proactive              += numTested_proactive
                    totalNumTested_onset                  += numTested_onset
                    totalNumTested_onset_groupmate        += numTested_onset_groupmate
                    totalNumTested_positive_groupmate     += numTested_positive_groupmate
                    totalNumTested_traced                 += numTested_traced
                    totalNumTested_deisolation            += numTested_deisolation
                    totalNumTested                        += len(testedIndividuals)
                    totalNumPositives_proactive           += numPositive_proactive
                    totalNumPositives_onset               += numPositive_onset
                    totalNumPositives_onset_groupmate     += numPositive_onset_groupmate
                    totalNumPositives_positive_groupmate  += numPositive_positive_groupmate
                    totalNumPositives_traced              += numPositive_traced
                    totalNumPositives_deisolation         += numPositive_deisolation
                    totalNumPositives                     += len(positiveIndividuals) 
                    totalNumIsolations_onset              += len(isolationCohort_onset)
                    totalNumIsolations_onset_groupmate    += len(isolationCohort_onset_groupmate)
                    totalNumIsolations_positive           += len(isolationCohort_positive)
                    totalNumIsolations_positive_groupmate += len(isolationCohort_positive_groupmate)
                    totalNumIsolations_traced             += len(isolationCohort_traced)
                    totalNumIsolations                    += len(isolationCohort)
                    totalNumTested_deisolation            += numTested_deisolation*num_deisolation_tests
                    totalNumDeisolations                  += numDeisolating

                    peakNumIsolated                       = max(peakNumIsolated, np.count_nonzero(model.isolation))

                    print("\t"+str(numTested_proactive)          +"\ttested proactively                       [+ "+str(numPositive_proactive)+" positive (%.2f %%) +]" % (numPositive_proactive/numTested_proactive*100 if numTested_proactive>0 else 0))
                    print("\t"+str(numTested_onset)              +"\ttested "+str(testing_delay_onset)+" days after onset              [+ "+str(numPositive_onset)+" positive (%.2f %%) +]" % (numPositive_onset/numTested_onset*100 if numTested_onset>0 else 0))                    
                    print("\t"+str(numTested_onset_groupmate)    +"\ttested "+str(testing_delay_onset_groupmate)+" days after groupmate onset    [+ "+str(numPositive_onset_groupmate)+" positive (%.2f %%) +]" % (numPositive_onset_groupmate/numTested_onset_groupmate*100 if numTested_onset_groupmate>0 else 0))
                    print("\t"+str(numTested_positive_groupmate) +"\ttested "+str(testing_delay_positive_groupmate)+" days after groupmate positive [+ "+str(numPositive_positive_groupmate)+" positive (%.2f %%) +]" % (numPositive_positive_groupmate/numTested_positive_groupmate*100 if numTested_positive_groupmate>0 else 0))
                    print("\t"+str(numTested_traced)             +"\ttested "+str(testing_delay_traced)+" days after being traced       [+ "+str(numPositive_traced)+" positive (%.2f %%) +]" % (numPositive_traced/numTested_traced*100 if numTested_traced>0 else 0))
                    print("\t"+str(len(testedIndividuals))       +"\tTESTED TOTAL                             [+ "+str(len(positiveIndividuals))+" positive (%.2f %%) +]" % (len(positiveIndividuals)/len(testedIndividuals)*100 if len(testedIndividuals)>0 else 0))

                    for test_type in positiveResultQueue:
                        print("\t"+str(len(positiveResultCohort[test_type]))+"\tpositive result "+str(test_result_delay[test_type])+" days after "+test_type+" test")

                    print("\t"+str(len(isolationCohort_onset))              +"\tisolated "+str(isolation_delay_onset)+" days after onset")
                    print("\t"+str(len(isolationCohort_onset_groupmate))    +"\tisolated "+str(isolation_delay_onset_groupmate)+" days after groupmate onset")
                    print("\t"+str(len(isolationCohort_positive))           +"\tisolated "+str(isolation_delay_positive)+" days after positive")
                    print("\t"+str(len(isolationCohort_positive_groupmate)) +"\tisolated "+str(isolation_delay_positive_groupmate)+" days after groupmate positive")
                    print("\t"+str(len(isolationCohort_traced))             +"\tisolated "+str(isolation_delay_traced)+" days after traced")
                    print("\t"+str(len(isolationCohort))                    +"\tISOLATED TOTAL")

                    print("\t"+str(numTested_deisolation)       +"\ttested for de-isolation "+str(testing_delay_deisolation)+" days after entering isolation [+ "+str(numPositive_deisolation)+" positive (%.2f %%) +]" % (numPositive_deisolation/numTested_deisolation*100 if numTested_deisolation>0 else 0))
                    print("\t"+str(numDeisolating)              +"\tDE-ISOLATED with "+str(num_deisolation_tests)+" negative tests.")
                    

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
            running = model.run_iteration(max_dt=max_dt, default_dt=default_dt, tau_step=tau_step)
            
            if(terminate_at_zero_cases):
                running = running and (currentNumInfected > 0) # or currentNumIsolated > 0) if false positives occur at non-negligible rate then this ends up running for a long time after 0 true cases

            # print("time_scenaro\t\t", time.time() - time_scenaro_start, "\n")
            # while loop
            #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Finalize model and simulation data:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        model.finalize_data_series()    # this function populates the model.results dict with basic stats

        #---------------------------------------------

        model.results.update({ 
            'sim_duration':                             model.t,
            'intervention_start_time':                  interventionStartTime,
            'intervention_end_time':                    model.t,
            'init_cadence_offset':                      init_cadence_offset,
            'total_num_introductions':                  totalNumIntroductions,
            'total_num_tests_proactive':                totalNumTested_proactive,              
            'total_num_tests_onset':                    totalNumTested_onset,                  
            'total_num_tests_groupmate':                totalNumTested_onset_groupmate,        
            'total_num_tests_positive_groupmate':       totalNumTested_positive_groupmate,     
            'total_num_tests_traced':                   totalNumTested_traced,                 
            'total_num_tests_deisolation':              totalNumTested_deisolation,                 
            'total_num_tests':                          totalNumTested,                        
            'total_num_positives_proactive':            totalNumPositives_proactive,          
            'total_num_positives_onset':                totalNumPositives_onset,              
            'total_num_positives_onset_groupmate':      totalNumPositives_onset_groupmate,    
            'total_num_positives_positive_groupmate':   totalNumPositives_positive_groupmate, 
            'total_num_positives_traced':               totalNumPositives_traced,             
            'total_num_positives_deisolation':          totalNumPositives_deisolation,             
            'total_num_positives':                      totalNumPositives,                    
            'total_num_true_positives':                 totalNumTruePositives,                    
            'total_num_false_positives':                totalNumFalsePositives,                    
            'total_num_true_negatives':                 totalNumTrueNegatives,                    
            'total_num_false_negatives':                totalNumFalseNegatives,                    
            'total_num_isolations_onset':               totalNumIsolations_onset,             
            'total_num_isolations_onset_groupmate':     totalNumIsolations_onset_groupmate,   
            'total_num_isolations_positive':            totalNumIsolations_positive,          
            'total_num_isolations_positive_groupmate':  totalNumIsolations_positive_groupmate,
            'total_num_isolations_traced':              totalNumIsolations_traced,            
            'total_num_isolations':                     totalNumIsolations,
            'total_num_deisolations':                   totalNumDeisolations,
            'peak_num_isolated':                        peakNumIsolated 
            })

        #---------------------------------------------

        return True

