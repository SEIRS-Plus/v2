# External libraries
import numpy as np
import itertools
# seirsplus libraries
from seirsplus.models.preconfig_disease_models import *
from seirsplus.networks import *
from seirsplus import utils
from seirsplus.sim_loops import *

#------------------------------

# Example calls

# run_SARSCoV2_interventions_scenario(parameters={'R0_MEAN': [3.0, 6.0], 'PROACTIVE_TESTING_CADENCE':['never', 'weekly', 'daily']}, reps=3)

# run_SARSCoV2_community_scenario(parameters={'R0_MEAN': [3.0, 6.0], 'PROACTIVE_TESTING_CADENCE':['never', 'weekly', 'daily']}, reps=3)

# run_SARSCoV2_primary_school_scenario(parameters={'R0_MEAN': [3.0, 6.0], 'PROACTIVE_TESTING_CADENCE':['never', 'weekly', 'daily']}, reps=3)

# run_SARSCoV2_secondary_school_scenario(parameters={'R0_MEAN': [3.0, 6.0], 'PROACTIVE_TESTING_CADENCE':['never', 'weekly', 'daily']}, reps=3)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def run_SARSCoV2_interventions_scenario(model=None, parameters=None, reps=1, metadata={}, outdir='./', run_label=None, save_results=True, save_caselogs=False, save_partial_results=True, results_columns=None, caselog_columns=None, output_file_extn='.csv'):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load in model and scenario parameter specifications:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load SEIRS+ default parameters for this scenario:
    params = utils.load_config('scenario_SARSCoV2.json')
    # Update parameters with any user provided values:
    if(parameters is not None):
        params.update(parameters)

    if(model is None):
        model = generate_SARSCoV2_interventions_model

    results, caselogs = run_interventions_scenario(model=model, parameters=params, reps=reps, metadata=metadata, outdir=outdir, run_label=run_label, save_results=save_results, save_caselogs=save_caselogs, save_partial_results=save_partial_results, results_columns=results_columns, caselog_columns=caselog_columns, output_file_extn=output_file_extn)

    return results, caselogs


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def run_SARSCoV2_community_scenario(parameters=None, reps=1, metadata={}, outdir='./', run_label=None, save_results=True, save_caselogs=False, save_partial_results=True, results_columns=None, caselog_columns=None, output_file_extn='.csv'):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load in model and scenario parameter specifications:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load SEIRS+ default parameters for this scenario:
    params = utils.load_config('scenario_SARSCoV2_community.json')
    # Update parameters with any user provided values:
    if(parameters is not None):
        params.update(parameters)

    results, caselogs = run_interventions_scenario(model=generate_SARSCoV2_community_model, parameters=params, reps=reps, metadata=metadata, outdir=outdir, run_label=run_label, save_results=save_results, save_caselogs=save_caselogs, save_partial_results=save_partial_results, results_columns=results_columns, caselog_columns=caselog_columns, output_file_extn=output_file_extn)

    return results, caselogs


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def run_SARSCoV2_primary_school_scenario(parameters=None, reps=1, metadata={}, outdir='./', run_label=None, save_results=True, save_caselogs=False, save_partial_results=True, results_columns=None, caselog_columns=None, output_file_extn='.csv'):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load in model and scenario parameter specifications:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load SEIRS+ default parameters for this scenario:
    params = utils.load_config('scenario_SARSCoV2_primaryschool.json')
    # Update parameters with any user provided values:
    if(parameters is not None):
        params.update(parameters)

    results, caselogs = run_interventions_scenario(model=generate_SARSCoV2_primary_school_model, parameters=params, reps=reps, metadata=metadata, outdir=outdir, run_label=run_label, save_results=save_results, save_caselogs=save_caselogs, save_partial_results=save_partial_results, results_columns=results_columns, caselog_columns=caselog_columns, output_file_extn=output_file_extn)

    return results, caselogs


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def run_SARSCoV2_secondary_school_scenario(parameters=None, reps=1, metadata={}, outdir='./', run_label=None, save_results=True, save_caselogs=False, save_partial_results=True, results_columns=None, caselog_columns=None, output_file_extn='.csv'):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load in model and scenario parameter specifications:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load SEIRS+ default parameters for this scenario:
    params = utils.load_config('scenario_SARSCoV2_secondaryschool.json')
    # Update parameters with any user provided values:
    if(parameters is not None):
        params.update(parameters)

    results, caselogs = run_interventions_scenario(model=generate_SARSCoV2_secondary_school_model, parameters=params, reps=reps, metadata=metadata, outdir=outdir, run_label=run_label, save_results=save_results, save_caselogs=save_caselogs, save_partial_results=save_partial_results, results_columns=results_columns, caselog_columns=caselog_columns, output_file_extn=output_file_extn)

    return results, caselogs


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def run_interventions_scenario(model, parameters, reps=1, metadata={}, outdir='./', run_label=None, save_results=True, save_caselogs=False, save_partial_results=True, results_columns=None, caselog_columns=None, output_file_extn='.csv'):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load in model and scenario parameter specifications:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    params = {}
    params.update(parameters)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Construct parameter sets from combinations of parameters 
    # with multiple values provided ("swept parameters"):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Put all parameter values into a list format (for itertools step below):
    params = {key: ([val] if not isinstance(val, (list, np.ndarray)) else [[]] if len(val) == 0 else val) for key, val in params.items()}
    paramNames = list(params.keys())
    paramNames_swept = [key for key, val in params.items() if len(val) >  1]
    # Generate a list of the full combinatoric product set of all param value lists in params dict:
    paramSets = list(itertools.product(*list(params.values()))) 

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data structures for storing results:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if(save_partial_results):
        results_df_master = None
        caselog_df_master = None
    else:
        results_dfs = []
        caselog_dfs = []

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # RUN (REPLICATE) SIMULATIONS FOR EACH PARAMETER SET:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print(f"[ RUNNING SIMULATIONS FOR PARAMETERIZED INTERVENTIONS SCENARIO ]")

    for paramSetNum, paramSet in enumerate(paramSets):
        paramSetNum += 1 # 1-indexed instead of 0-indexed
        
        PARAM_SET = "paramSet"+str(paramSetNum)

        paramSetDict = metadata
        paramSetDict.update({'PARAM_SET': PARAM_SET })
        paramSetDict.update({paramNames[i]: paramSet[i] for i in range(len(paramSet))})

        print(f"Running simulations for parameter set {paramSetNum}/{len(paramSets)}: {str({paramNames[i]: paramSet[i] for i in range(len(paramSet)) if paramNames[i] in paramNames_swept})}")

        for rep in range(reps):
            rep += 1 # 1-indexed instead of 0-indexed

            metadata.update({'rep': rep, 'run_label': run_label})

            print(f"\tsimulation rep {rep}/{reps}...\t\r", end="")

            # Get the model to run (the model fn argument may be a pointer to a function that generates a model):
            if(callable(model)):
                model = model(paramSetDict)

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Simulate the model scenario:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            run_interventions_sim(model, 
                                    # Scenario run time params:
                                    T = paramSetDict['T'], 
                                    max_dt = paramSetDict['MAX_DT'], 
                                    default_dt = paramSetDict['DEFAULT_DT'],
                                    tau_step=paramSetDict['TAU_STEP'] if isinstance(paramSetDict['TAU_STEP'], (int, float)) and paramSetDict['TAU_STEP'] > 0 else None,
                                    terminate_at_zero_cases=paramSetDict['TERMINATE_AT_ZERO_CASES'],
                                    # Intervention timing params:
                                    cadence_dt = paramSetDict['CADENCE_DT'], 
                                    cadence_cycle_length = paramSetDict['CADENCE_CYCLE_LENGTH'],
                                    init_cadence_offset = paramSetDict['INIT_CADENCE_OFFSET'],
                                    cadence_presets = paramSetDict['CADENCE_PRESETS'],
                                    intervention_start_time = paramSetDict['INTERVENTION_START_TIME'],
                                    intervention_start_prevalence = paramSetDict['INTERVENTION_START_PREVALENCE'],
                                    prevalence_flags = paramSetDict['PREVALENCE_FLAGS'],
                                    onset_flags = paramSetDict['ONSET_FLAGS'],
                                    # Case introduction params:
                                    case_introduction_rate = paramSetDict['CASE_INTRODUCTION_RATE'],
                                    # Network params:
                                    network_active_cadences = {network: 'daily' if network!='household' else 'nightly' for network in model.networks},
                                    # Isolation params:
                                    isolation_period = paramSetDict['ISOLATION_PERIOD'],
                                    isolation_delay_onset = paramSetDict['ISOLATION_DELAY_ONSET'],
                                    isolation_delay_onset_groupmate = paramSetDict['ISOLATION_DELAY_ONSET_GROUPMATE'],
                                    isolation_delay_positive = paramSetDict['ISOLATION_DELAY_POSITIVE'],
                                    isolation_delay_positive_groupmate = paramSetDict['ISOLATION_DELAY_POSITIVE_GROUPMATE'],
                                    isolation_delay_traced = paramSetDict['ISOLATION_DELAY_TRACED'],
                                    isolation_compliance_onset = paramSetDict['ISOLATION_COMPLIANCE_ONSET'],
                                    isolation_compliance_onset_groupmate = paramSetDict['ISOLATION_COMPLIANCE_ONSET_GROUPMATE'],
                                    isolation_compliance_positive = paramSetDict['ISOLATION_COMPLIANCE_POSITIVE'],
                                    isolation_compliance_positive_groupmate = paramSetDict['ISOLATION_COMPLIANCE_POSITIVE_GROUPMATE'],
                                    isolation_compliance_traced = paramSetDict['ISOLATION_COMPLIANCE_TRACED'],
                                    isolation_exclude_compartments=paramSetDict['ISOLATION_EXCLUDE_COMPARTMENTS'],
                                    isolation_exclude_flags=paramSetDict['ISOLATION_EXCLUDE_FLAGS'],
                                    isolation_exclude_isolated=bool(paramSetDict['ISOLATION_EXCLUDE_ISOLATED']),
                                    isolation_exclude_afterNumTests=paramSetDict['ISOLATION_EXCLUDE_AFTERNUMTESTS'] if isinstance(paramSetDict['ISOLATION_EXCLUDE_AFTERNUMTESTS'], (int, float)) and paramSetDict['ISOLATION_EXCLUDE_AFTERNUMTESTS'] > 0 else None,     
                                    isolation_exclude_afterNumVaccineDoses=paramSetDict['ISOLATION_EXCLUDE_AFTERNUMVACCINEDOSES'] if isinstance(paramSetDict['ISOLATION_EXCLUDE_AFTERNUMVACCINEDOSES'], (int, float)) and paramSetDict['ISOLATION_EXCLUDE_AFTERNUMVACCINEDOSES'] > 0 else None,
                                    # Testing params:
                                    test_params=utils.load_config(paramSetDict['TEST_PARAMS_CFG']),
                                    test_type_proactive = paramSetDict['TEST_TYPE_PROACTIVE'],
                                    test_type_onset = paramSetDict['TEST_TYPE_ONSET'],
                                    test_type_traced = paramSetDict['TEST_TYPE_TRACED'],
                                    test_type_onset_groupmate = paramSetDict['TEST_TYPE_ONSET_GROUPMATE'],
                                    test_type_positive_groupmate = paramSetDict['TEST_TYPE_POSITIVE_GROUPMATE'],
                                    test_type_deisolation = paramSetDict['TEST_TYPE_DEISOLATION'],
                                    test_result_delay={'molecular': paramSetDict['TEST_RESULT_DELAY_MOLECULAR'], 'antigen': paramSetDict['TEST_RESULT_DELAY_ANTIGEN']},
                                    proactive_testing_cadence = paramSetDict['PROACTIVE_TESTING_CADENCE'],
                                    proactive_testing_synchronize = paramSetDict['PROACTIVE_TESTING_SYNCHRONIZE'],
                                    testing_capacity_max = paramSetDict['TESTING_CAPACITY_MAX'],
                                    testing_capacity_proactive = paramSetDict['TESTING_CAPACITY_PROACTIVE'],
                                    testing_accessibility_proactive = paramSetDict['TESTING_ACCESSIBILITY_PROACTIVE'],
                                    num_deisolation_tests = paramSetDict['NUM_DEISOLATION_TESTS'],
                                    testing_delay_proactive = paramSetDict['TESTING_DELAY_PROACTIVE'],
                                    testing_delay_onset = paramSetDict['TESTING_DELAY_ONSET'],
                                    testing_delay_onset_groupmate = paramSetDict['TESTING_DELAY_ONSET_GROUPMATE'],
                                    testing_delay_positive_groupmate = paramSetDict['TESTING_DELAY_POSITIVE_GROUPMATE'],
                                    testing_delay_traced = paramSetDict['TESTING_DELAY_TRACED'],
                                    testing_delay_deisolation = paramSetDict['TESTING_DELAY_DEISOLATION'],
                                    testing_compliance_proactive = paramSetDict['TESTING_COMPLIANCE_PROACTIVE'],
                                    testing_compliance_onset = paramSetDict['TESTING_COMPLIANCE_ONSET'],
                                    testing_compliance_onset_groupmate = paramSetDict['TESTING_COMPLIANCE_ONSET_GROUPMATE'],
                                    testing_compliance_positive_groupmate = paramSetDict['TESTING_COMPLIANCE_POSITIVE_GROUPMATE'],
                                    testing_compliance_traced = paramSetDict['TESTING_COMPLIANCE_TRACED'],
                                    testing_compliance_deisolation = paramSetDict['TESTING_COMPLIANCE_DEISOLATION'],
                                    testing_exclude_compartments=paramSetDict['TESTING_EXCLUDE_COMPARTMENTS'],
                                    testing_exclude_flags=paramSetDict['TESTING_EXCLUDE_FLAGS'],
                                    testing_exclude_isolated=bool(paramSetDict['TESTING_EXCLUDE_ISOLATED']),
                                    testing_exclude_afterNumTests=paramSetDict['TESTING_EXCLUDE_AFTERNUMTESTS'] if isinstance(paramSetDict['TESTING_EXCLUDE_AFTERNUMTESTS'], (int, float)) and paramSetDict['TESTING_EXCLUDE_AFTERNUMTESTS'] > 0 else None,     
                                    testing_exclude_afterNumVaccineDoses=paramSetDict['TESTING_EXCLUDE_AFTERNUMVACCINEDOSES'] if isinstance(paramSetDict['TESTING_EXCLUDE_AFTERNUMVACCINEDOSES'], (int, float)) and paramSetDict['TESTING_EXCLUDE_AFTERNUMVACCINEDOSES'] > 0 else None,     
                                    # Tracing params:                                                                       
                                    tracing_pct_contacts = paramSetDict['TRACING_PCT_CONTACTS'],
                                    tracing_delay = paramSetDict['TRACING_DELAY'],
                                    tracing_compliance = paramSetDict['TRACING_COMPLIANCE'],
                                    tracing_exclude_networks=paramSetDict['TRACING_EXCLUDE_NETWORKS'],
                                    # Misc. params:
                                    intervention_groups=model.intervention_groups if hasattr(model, 'intervention_groups') else None,
                                    print_updates=bool(paramSetDict['PRINT_UPDATES'])
                                    )

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update results data with other info:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store model/scenario parameter values with results:
            model.results.update(paramSetDict)
            # Store disease statistics:
            model.results.update(model.disease_stats)
            # Store network statistics:
            overallNetwork = union_of_networks([network for network in [network['networkx'] for network in model.networks.values()]])
            model.results.update(network_stats(networks=overallNetwork, names="overall_network", calc_connected_components=True))
            # Store simulation metadata:
            model.results.update(metadata)
        
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Add results data frames to running data set:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if(save_partial_results):
                results_df_master = results_df_master.append(model.get_results_dataframe(), ignore_index=True) if results_df_master is not None else model.get_results_dataframe()
                if(paramSetDict['TRACK_CASE_INFO']):
                    caselog_df = model.get_case_log_dataframe()
                    # Add columns with the given metadata values (including rep number)
                    for key, val in metadata.items():
                        caselog_df[key] = val if not (isinstance(val, (list, np.ndarray)) and len(val) == 0) else None
                    caselog_df_master = caselog_df_master.append(caselog_df, ignore_index=True) if caselog_df_master is not None else caselog_df
                # Save results (so far) to file:
                if(save_results):
                    utils.save_dataframe(results_df_master, file_name=outdir+'/results'+('_'+run_label if run_label is not None else ''), file_extn=output_file_extn, columns=results_columns)
                if(save_caselogs):
                    utils.save_dataframe(caselog_df_master, file_name=outdir+'/caselogs'+('_'+run_label if run_label is not None else ''), file_extn=output_file_extn, columns=caselog_columns)
            else:
                results_dfs.append(model.get_results_dataframe())
                if(paramSetDict['TRACK_CASE_INFO']):
                    caselog_df = model.get_case_log_dataframe()
                    # Add columns with the given metadata values (including rep number)
                    for key, val in metadata.items():
                        caselog_df[key] = val if not (isinstance(val, (list, np.ndarray)) and len(val) == 0) else None
                    caselog_dfs.append(caselog_df)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compile all param set and rep data into single dataframes:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if(not save_partial_results):
        results_df_master = pd.concat(results_dfs)
        caselog_df_master = pd.concat(caselog_dfs)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save final results to file:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if(save_results):
        utils.save_dataframe(results_df_master, file_name=outdir+'/results'+('_'+run_label if run_label is not None else ''), file_extn=output_file_extn, columns=results_columns)
    if(save_caselogs):
        utils.save_dataframe(caselog_df_master, file_name=outdir+'/caselogs'+('_'+run_label if run_label is not None else ''), file_extn=output_file_extn, columns=caselog_columns)

    return results_df_master, caselog_df_master


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def generate_SARSCoV2_interventions_model(parameters=None):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load in model and scenario parameter specifications:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load SEIRS+ default parameters for this scenario:
    params = utils.load_config('scenario_SARSCoV2.json')
    # Update parameters with any user provided values:
    if(parameters is not None):
        params.update(parameters)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate contact networks:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    MEAN_DEGREE = 10
    MEAN_CLUSTER_SIZE = 10
    CLUSTER_INTERCONNECTEDNESS = 0.25
    network, network_info = generate_workplace_contact_network(
                                N=params['N'],
                                num_cohorts=1,
                                num_nodes_per_cohort=params['N'],
                                num_teams_per_cohort=int(params['N'] / MEAN_CLUSTER_SIZE),
                                mean_intracohort_degree=MEAN_DEGREE,
                                farz_params={
                                    "beta": (1 - CLUSTER_INTERCONNECTEDNESS),
                                    "alpha": 5.0,
                                    "gamma": 5.0,
                                    "r": 1,
                                    "q": 0.0,
                                    "phi": 50,
                                    "b": 0,
                                    "epsilon": 1e-6,
                                    "directed": False,
                                    "weighted": False,
                                },
                            )
    networks = {"workplace": network}

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Instantiate the model:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model = SARSCoV2NetworkModel(   networks                                 = networks, 
                                    R0_mean                                  = params['R0_MEAN'],
                                    R0_cv                                    = params['R0_CV'],
                                    relative_transmissibility_presymptomatic = params['RELATIVE_TRANSMISSIBILITY_PRESYMPTOMATIC'], 
                                    relative_transmissibility_asymptomatic   = params['RELATIVE_TRANSMISSIBILITY_ASYMPTOMATIC'], 
                                    relative_susceptibility_priorexposure    = params['RELATIVE_SUSCEPTIBILITY_PRIOREXPOSURE'],    
                                    relative_susceptibility_reinfection      = params['RELATIVE_SUSCEPTIBILITY_REINFECTION'],
                                    latent_period                            = utils.gamma_dist(mean=params['LATENT_PERIOD_MEAN'], coeffvar=params['LATENT_PERIOD_CV'], N=params['N']),
                                    presymptomatic_period                    = utils.gamma_dist(mean=params['PRESYMPTOMATIC_PERIOD_MEAN'], coeffvar=params['PRESYMPTOMATIC_PERIOD_CV'], N=params['N']),
                                    symptomatic_period                       = utils.gamma_dist(mean=params['SYMPTOMATIC_PERIOD_MEAN'], coeffvar=params['SYMPTOMATIC_PERIOD_CV'], N=params['N']),
                                    pct_asymptomatic                         = params['PCT_ASYMPTOMATIC'],
                                    mixedness                                = params['MIXEDNESS'],
                                    openness                                 = params['OPENNESS'],
                                    track_case_info                          = params['TRACK_CASE_INFO'] )


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Specify other model configurations:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Tests:
    model.update_test_parameters(utils.load_config(params['TEST_PARAMS_CFG']))

    # Vaccines:
    model.add_vaccine(series='covid-vaccine', name='booster', 
                        susc_effectiveness=params['VACCINE_EFFECTIVENESS_SUSCEPTIBILITY'], 
                        transm_effectiveness=params['VACCINE_EFFECTIVENESS_TRANSMISSIBILITY'])

    # Set different asymptomatic rates for vaccinated individuals:
    pct_asymptomatic_vaccinated = utils.param_as_array(params['PCT_ASYMPTOMATIC_VACCINATED'], (1, params['N']))                                
    model.set_transition_probability('Pv', {'Iv': 1 - pct_asymptomatic_vaccinated, 'Av': pct_asymptomatic_vaccinated})

    # Particular assumption about handling susceptibility to reinfection for individuals with prior infection AND vaccination for this particular analysis:
    model.set_susceptibility(['Rv'],  to=['P', 'I', 'A', 'Pv', 'Iv', 'Av'], susceptibility=min(params['RELATIVE_SUSCEPTIBILITY_REINFECTION'],   1-params['VACCINE_EFFECTIVENESS_SUSCEPTIBILITY']))
    model.set_susceptibility(['Rpv'], to=['P', 'I', 'A', 'Pv', 'Iv', 'Av'], susceptibility=min(params['RELATIVE_SUSCEPTIBILITY_PRIOREXPOSURE'], 1-params['VACCINE_EFFECTIVENESS_SUSCEPTIBILITY']))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set up the initial state:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize a specified percentage of individuals to a prior exposure (recovered) state:
    model.set_initial_prevalence('Rp', params['INIT_PCT_PRIOR_EXPOSURE'])

    # Special version of vaccination (booster) pool for omicron workplace webapp,
    # Restricting booster vaccination to Rp individuals until booster uptake exceeds Rp prevalence
    model.vaccinate(node=(np.random.choice(model.get_individuals_by_compartment('Rp'), size=int(params['VACCINE_UPTAKE']*params['N']), replace=False)
                            if params['INIT_PCT_PRIOR_EXPOSURE'] >= params['VACCINE_UPTAKE']
                            else np.random.choice(params['N'], size=int(params['VACCINE_UPTAKE']*params['N']), replace=False)), 
                    vaccine_series='covid-vaccine')

    # Administer initial masking:
    model.mask(node=np.random.choice(range(params['N']), size=int(params['MASK_UPTAKE']*params['N']), replace=False), 
                susc_effectiveness=params['MASK_EFFECTIVENESS_SUSCEPTIBILITY'], transm_effectiveness=params['MASK_EFFECTIVENESS_TRANSMISSIBILITY'])

    # Add a 'prior_exposure' flag to each individual with a prior infection:
    # (this flag is on the Rp and Rpv compartments, but adding to individuals
    #  so they will retain this flag after leaving these compartments)
    for i in model.get_individuals_by_compartment(['Rp', 'Rpv']): 
        model.add_individual_flag(node=i, flag='prior_exposure')

    # Introduce a number of random exposures to meet the given init prevalence of removed recovereds:
    model.introduce_random_exposures(int(params['INIT_PCT_REMOVED']*params['N']), post_exposure_state='R')

    # Introduce a number of random exposures to meet the given init prevalence of active infections:
    model.introduce_random_exposures(int(params['INIT_PREVALENCE']*params['N']))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return the constructed model:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return model


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def generate_SARSCoV2_community_model(parameters=None):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load in model and scenario parameter specifications:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load SEIRS+ default parameters for this scenario:
    params = utils.load_config('scenario_SARSCoV2_community.json')
    # Update parameters with any user provided values:
    if(parameters is not None):
        params.update(parameters)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate contact networks:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    networks, clusters, households, age_groups, node_age_group_labels = generate_community_networks(params['N'])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Apply social distancing to networks:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for network_name, network_obj in networks.items():
        if(network_name != 'household'):
            apply_social_distancing(network_obj, contact_drop_prob=params['SOCIAL_DISTANCING_CONTACT_DROP_PROB'], distancing_compliance=params['SOCIAL_DISTANCING_COMPLIANCE'])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set up heterogeneous and age-stratified param distributions:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Adjust relative individual R0s for each age group:
    R0 = utils.gamma_dist(mean=params['R0_MEAN'], coeffvar=params['R0_CV'], N=params['N'])
    R0 = [ R0[i] * (     params['RELATIVE_TRANSMISSIBILITY_AGE0TO4']   if age_group == 'age0-4' and params['RELATIVE_TRANSMISSIBILITY_AGE0TO4'] is not None
                    else params['RELATIVE_TRANSMISSIBILITY_AGE5TO11']  if age_group == 'age5-11' and params['RELATIVE_TRANSMISSIBILITY_AGE5TO11'] is not None
                    else params['RELATIVE_TRANSMISSIBILITY_AGE12TO17'] if age_group == 'age12-17' and params['RELATIVE_TRANSMISSIBILITY_AGE12TO17'] is not None
                    else params['RELATIVE_TRANSMISSIBILITY_AGE18TO24'] if age_group == 'age18-24' and params['RELATIVE_TRANSMISSIBILITY_AGE18TO24'] is not None
                    else params['RELATIVE_TRANSMISSIBILITY_AGE25TO64'] if age_group in ['age25-29', 'age30-34', 'age35-39', 'age40-44', 'age45-49', 'age50-54', 'age55-59', 'age60-64'] and params['RELATIVE_TRANSMISSIBILITY_AGE25TO64'] is not None
                    else params['RELATIVE_TRANSMISSIBILITY_AGE65PLUS'] if age_group == 'age65+' and params['RELATIVE_TRANSMISSIBILITY_AGE65PLUS'] is not None
                    else 1.0)
                    for i, age_group in enumerate(node_age_group_labels) ]

    # Adjust relative susceptibilities for each age group:
    susceptibility = np.ones(params['N'])
    susceptibility = [ susceptibility[i] * (     params['RELATIVE_SUSCEPTIBILITY_AGE0TO4']   if age_group == 'age0-4' and params['RELATIVE_SUSCEPTIBILITY_AGE0TO4'] is not None
                                            else params['RELATIVE_SUSCEPTIBILITY_AGE5TO11']  if age_group == 'age5-11' and params['RELATIVE_SUSCEPTIBILITY_AGE5TO11'] is not None
                                            else params['RELATIVE_SUSCEPTIBILITY_AGE12TO17'] if age_group == 'age12-17' and params['RELATIVE_SUSCEPTIBILITY_AGE12TO17'] is not None
                                            else params['RELATIVE_SUSCEPTIBILITY_AGE18TO24'] if age_group == 'age18-24' and params['RELATIVE_SUSCEPTIBILITY_AGE18TO24'] is not None
                                            else params['RELATIVE_SUSCEPTIBILITY_AGE25TO64'] if age_group in ['age25-29', 'age30-34', 'age35-39', 'age40-44', 'age45-49', 'age50-54', 'age55-59', 'age60-64'] and params['RELATIVE_SUSCEPTIBILITY_AGE25TO64'] is not None
                                            else params['RELATIVE_SUSCEPTIBILITY_AGE65PLUS'] if age_group == 'age65+' and params['RELATIVE_SUSCEPTIBILITY_AGE65PLUS'] is not None
                                            else 1.0)
                                            for i, age_group in enumerate(node_age_group_labels) ]

    # Assign asymptomatic fractions to each age group:
    pct_asymptomatic = [     params['PCT_ASYMPTOMATIC_AGE0TO4']   if age_group == 'age0-4' and params['PCT_ASYMPTOMATIC_AGE0TO4'] is not None
                        else params['PCT_ASYMPTOMATIC_AGE5TO11']  if age_group == 'age5-11' and params['PCT_ASYMPTOMATIC_AGE5TO11'] is not None
                        else params['PCT_ASYMPTOMATIC_AGE12TO17'] if age_group == 'age12-17' and params['PCT_ASYMPTOMATIC_AGE12TO17'] is not None
                        else params['PCT_ASYMPTOMATIC_AGE18TO24'] if age_group == 'age18-24' and params['PCT_ASYMPTOMATIC_AGE18TO24'] is not None
                        else params['PCT_ASYMPTOMATIC_AGE25TO64'] if age_group in ['age25-29', 'age30-34', 'age35-39', 'age40-44', 'age45-49', 'age50-54', 'age55-59', 'age60-64'] and params['PCT_ASYMPTOMATIC_AGE25TO64'] is not None
                        else params['PCT_ASYMPTOMATIC_AGE65PLUS'] if age_group == 'age65+' and params['PCT_ASYMPTOMATIC_AGE65PLUS'] is not None
                        else params['PCT_ASYMPTOMATIC']
                        for age_group in node_age_group_labels ]

    latent_period         = utils.gamma_dist(mean=params['LATENT_PERIOD_MEAN'], coeffvar=params['LATENT_PERIOD_CV'], N=params['N'])
    presymptomatic_period = utils.gamma_dist(mean=params['PRESYMPTOMATIC_PERIOD_MEAN'], coeffvar=params['PRESYMPTOMATIC_PERIOD_CV'], N=params['N'])
    symptomatic_period    = utils.gamma_dist(mean=params['SYMPTOMATIC_PERIOD_MEAN'], coeffvar=params['SYMPTOMATIC_PERIOD_CV'], N=params['N'])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Instantiate the model:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model = SARSCoV2NetworkModel(   networks                                 = networks, 
                                    R0                                       = R0,
                                    relative_transmissibility_presymptomatic = params['RELATIVE_TRANSMISSIBILITY_PRESYMPTOMATIC'], 
                                    relative_transmissibility_asymptomatic   = params['RELATIVE_TRANSMISSIBILITY_ASYMPTOMATIC'], 
                                    susceptibility                           = susceptibility,
                                    relative_susceptibility_priorexposure    = params['RELATIVE_SUSCEPTIBILITY_PRIOREXPOSURE'],    
                                    relative_susceptibility_reinfection      = params['RELATIVE_SUSCEPTIBILITY_REINFECTION'],
                                    latent_period                            = latent_period,
                                    presymptomatic_period                    = presymptomatic_period,
                                    symptomatic_period                       = symptomatic_period,
                                    pct_asymptomatic                         = pct_asymptomatic,
                                    mixedness                                = params['MIXEDNESS'],
                                    openness                                 = params['OPENNESS'],
                                    track_case_info                          = params['TRACK_CASE_INFO'] )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Specify other model configurations:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set that individuals remain active in the 'household' network layer when in isolation:
    model.set_network_activity('household', active_isolation=True)

    # Tests:
    model.update_test_parameters(utils.load_config(params['TEST_PARAMS_CFG']))

    # Vaccines:
    model.add_vaccine(series='covid-vaccine', name='generic-dose', 
                        susc_effectiveness=params['VACCINE_EFFECTIVENESS_SUSCEPTIBILITY'], 
                        transm_effectiveness=params['VACCINE_EFFECTIVENESS_TRANSMISSIBILITY'])

    # Set different asymptomatic rates for vaccinated individuals:
    pct_asymptomatic_vaccinated = [      params['PCT_ASYMPTOMATIC_VACCINATED_AGE0TO4']   if age_group == 'age0-4' and params['PCT_ASYMPTOMATIC_VACCINATED_AGE0TO4'] is not None
                                    else params['PCT_ASYMPTOMATIC_VACCINATED_AGE5TO11']  if age_group == 'age5-11' and params['PCT_ASYMPTOMATIC_VACCINATED_AGE5TO11'] is not None
                                    else params['PCT_ASYMPTOMATIC_VACCINATED_AGE12TO17'] if age_group == 'age12-17' and params['PCT_ASYMPTOMATIC_VACCINATED_AGE12TO17'] is not None
                                    else params['PCT_ASYMPTOMATIC_VACCINATED_AGE18TO24'] if age_group == 'age18-24' and params['PCT_ASYMPTOMATIC_VACCINATED_AGE18TO24'] is not None
                                    else params['PCT_ASYMPTOMATIC_VACCINATED_AGE25TO64'] if age_group in ['age25-29', 'age30-34', 'age35-39', 'age40-44', 'age45-49', 'age50-54', 'age55-59', 'age60-64'] and params['PCT_ASYMPTOMATIC_VACCINATED_AGE25TO64'] is not None
                                    else params['PCT_ASYMPTOMATIC_VACCINATED_AGE65PLUS'] if age_group == 'age65+' and params['PCT_ASYMPTOMATIC_VACCINATED_AGE65PLUS'] is not None
                                    else params['PCT_ASYMPTOMATIC_VACCINATED']
                                    for age_group in node_age_group_labels ]
    pct_asymptomatic_vaccinated = utils.param_as_array(pct_asymptomatic_vaccinated, (1, params['N']))                                
    model.set_transition_probability('Pv', {'Iv': 1 - pct_asymptomatic_vaccinated, 'Av': pct_asymptomatic_vaccinated})

    # Explicit decision about handling susceptibility to reinfection for individuals with prior infection AND vaccination for this particular analysis:
    model.set_susceptibility(['Rv'],  to=['P', 'I', 'A', 'Pv', 'Iv', 'Av'], susceptibility=min(params['RELATIVE_SUSCEPTIBILITY_REINFECTION'],   1-params['VACCINE_EFFECTIVENESS_SUSCEPTIBILITY']))
    model.set_susceptibility(['Rpv'], to=['P', 'I', 'A', 'Pv', 'Iv', 'Av'], susceptibility=min(params['RELATIVE_SUSCEPTIBILITY_PRIOREXPOSURE'], 1-params['VACCINE_EFFECTIVENESS_SUSCEPTIBILITY']))

    # Add a node flag with each individual's age group label:
    for i in range(params['N']): 
        model.add_individual_flag(node=i, flag=node_age_group_labels[i])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set up the initial state:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize a specified percentage of individuals to a prior exposure (recovered) state:
    model.set_initial_prevalence('Rp', params['INIT_PCT_PRIOR_EXPOSURE'])

    # Administer initial vaccines:
    for age_group in age_groups:
        ageGroupIndividuals = model.get_individuals_by_flag('age'+age_group)
        ageGroupVaccineUptake = (      params['VACCINE_UPTAKE_AGE0TO4']   if 'age'+age_group == 'age0-4' and params['VACCINE_UPTAKE_AGE0TO4'] is not None
                                  else params['VACCINE_UPTAKE_AGE5TO11']  if 'age'+age_group == 'age5-11' and params['VACCINE_UPTAKE_AGE5TO11'] is not None
                                  else params['VACCINE_UPTAKE_AGE12TO17'] if 'age'+age_group == 'age12-17' and params['VACCINE_UPTAKE_AGE12TO17'] is not None
                                  else params['VACCINE_UPTAKE_AGE18TO24'] if 'age'+age_group == 'age18-24' and params['VACCINE_UPTAKE_AGE18TO24'] is not None
                                  else params['VACCINE_UPTAKE_AGE25TO64'] if 'age'+age_group in ['age25-29', 'age30-34', 'age35-39', 'age40-44', 'age45-49', 'age50-54', 'age55-59', 'age60-64'] and params['VACCINE_UPTAKE_AGE25TO64'] is not None
                                  else params['VACCINE_UPTAKE_AGE65PLUS'] if 'age'+age_group == 'age65+' and params['VACCINE_UPTAKE_AGE65PLUS'] is not None
                                  else params['VACCINE_UPTAKE'] )
        model.vaccinate(node=np.random.choice(ageGroupIndividuals, size=int(ageGroupVaccineUptake*len(ageGroupIndividuals)), replace=False), 
                        vaccine_series='covid-vaccine')

    # Administer initial masking:
    for age_group in age_groups:
        ageGroupIndividuals = model.get_individuals_by_flag('age'+age_group)
        ageGroupMaskingUptake = (      params['MASK_UPTAKE_AGE0TO4']   if 'age'+age_group == 'age0-4' and params['MASK_UPTAKE_AGE0TO4'] is not None
                                      else params['MASK_UPTAKE_AGE5TO11']  if 'age'+age_group == 'age5-11' and params['MASK_UPTAKE_AGE5TO11'] is not None
                                      else params['MASK_UPTAKE_AGE12TO17'] if 'age'+age_group == 'age12-17' and params['MASK_UPTAKE_AGE12TO17'] is not None
                                      else params['MASK_UPTAKE_AGE18TO24'] if 'age'+age_group == 'age18-24' and params['MASK_UPTAKE_AGE18TO24'] is not None
                                      else params['MASK_UPTAKE_AGE25TO64'] if 'age'+age_group in ['age25-29', 'age30-34', 'age35-39', 'age40-44', 'age45-49', 'age50-54', 'age55-59', 'age60-64'] and params['MASK_UPTAKE_AGE25TO64'] is not None
                                      else params['MASK_UPTAKE_AGE65PLUS'] if 'age'+age_group == 'age65+' and params['MASK_UPTAKE_AGE65PLUS'] is not None
                                      else params['MASK_UPTAKE'] )
        model.mask(node=np.random.choice(ageGroupIndividuals, size=int(ageGroupMaskingUptake*len(ageGroupIndividuals)), replace=False), 
                    susc_effectiveness=params['MASK_EFFECTIVENESS_SUSCEPTIBILITY'], transm_effectiveness=params['MASK_EFFECTIVENESS_TRANSMISSIBILITY'])

    # Add a 'prior_exposure' flag to each individual with a prior infection:
    # (this flag is on the Rp and Rpv compartments, but adding to individuals
    #  so they will retain this flag after leaving these compartments)
    for i in model.get_individuals_by_compartment(['Rp', 'Rpv']): 
        model.add_individual_flag(node=i, flag='prior_exposure')

    # Introduce a number of random exposures to meet the given init prevalence of removed recovereds:
    model.introduce_random_exposures(int(params['INIT_PCT_REMOVED']*params['N']), post_exposure_state='R')

    # Introduce a number of random exposures to meet the given init prevalence of active infections:
    model.introduce_random_exposures(int(params['INIT_PREVALENCE']*params['N']))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store intervention group info in the model object
    # (janky, but done so it this info can be passed to 
    # run_interventions_sim on a model by model basis 
    # when using scenario wrappers) :
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model.intervention_groups = [hh['indices'] for hh in households]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return the constructed model:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return model


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def generate_SARSCoV2_primary_school_model(parameters=None):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load in model and scenario parameter specifications:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load SEIRS+ default parameters for this scenario:
    params = utils.load_config('scenario_SARSCoV2_primaryschool.json')
    # Update parameters with any user provided values:
    if(parameters is not None):
        params.update(parameters)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate contact networks:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    NUM_GRADES                = 6
    NUM_CLASSES_PER_GRADE     = 4
    NUM_STUDENTS_PER_CLASS    = 20
    NUM_STUDENTS              = NUM_GRADES*NUM_CLASSES_PER_GRADE*NUM_STUDENTS_PER_CLASS
    NUM_TEACHERS              = NUM_GRADES*NUM_CLASSES_PER_GRADE*1
    NUM_STAFF                 = NUM_TEACHERS
    TEACHERSTAFF_DEGREE       = 8
    NUM_TEACHERSTAFF_COMMS    = 1
    NUM_STUDENT_BLOCKS        = 1

    networks, network_info = generate_primary_school_contact_network(num_grades                     = NUM_GRADES, 
                                                                     num_classes_per_grade          = NUM_CLASSES_PER_GRADE, 
                                                                     class_sizes                    = NUM_STUDENTS_PER_CLASS, 
                                                                     num_student_blocks             = NUM_STUDENT_BLOCKS, 
                                                                     block_by_household             = True, 
                                                                     connect_students_in_households = True, 
                                                                     num_staff                      = NUM_STAFF, 
                                                                     num_teacher_staff_communities  = NUM_TEACHERSTAFF_COMMS, 
                                                                     teacher_staff_degree           = TEACHERSTAFF_DEGREE)

    networks = {'school': networks['school'], 'households': networks['households']}

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    N = int(networks['school'].number_of_nodes())

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # import matplotlib.pyplot as plt
    # node_colors = ['tab:green' if label=='teacher' else 'tab:orange' if label=='staff' else 'tab:blue' for label in network_info['node_labels']]
    # networkx.draw(networks['school'], pos=networkx.spring_layout(networks['school'], weight='layout_weight'), node_size=20, node_color=node_colors, edge_color='lightgray', alpha=0.5)
    # networkx.draw(networks['school-studentsonly'], pos=networkx.spring_layout(networks['school-studentsonly'], weight='layout_weight'), node_size=20, node_color=node_colors, edge_color='lightgray', alpha=0.5)
    # networkx.draw(networks['school-teacherstaffonly'], pos=networkx.spring_layout(networks['school-teacherstaffonly'], weight='layout_weight'), node_size=20, node_color=node_colors, edge_color='lightgray', alpha=0.5)
    # networkx.draw(networks['households'], pos=networkx.spring_layout(networks['households'], weight='layout_weight'), node_size=20, node_color=node_colors, edge_color='lightgray', alpha=0.5)
    # plt.show()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set up heterogeneous and age-stratified param distributions:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Adjust relative individual R0s for each age group:
    R0 = utils.gamma_dist(mean=params['R0_MEAN'], coeffvar=params['R0_CV'], N=N)
    R0 = [ R0[i] * (     params['RELATIVE_TRANSMISSIBILITY_STUDENT'] if label == 'student' and params['RELATIVE_TRANSMISSIBILITY_STUDENT'] is not None
                    else params['RELATIVE_TRANSMISSIBILITY_ADULT']   if label in ['teacher', 'staff'] and params['RELATIVE_TRANSMISSIBILITY_ADULT'] is not None
                    else 1.0)
                    for i, label in enumerate(network_info['node_labels']) ]

    # Adjust relative susceptibilities for each age group:
    susceptibility = np.ones(N)
    susceptibility = [ susceptibility[i] * (     params['RELATIVE_SUSCEPTIBILITY_STUDENT'] if label == 'student' and params['RELATIVE_SUSCEPTIBILITY_STUDENT'] is not None
                                            else params['RELATIVE_SUSCEPTIBILITY_ADULT']   if label in ['teacher', 'staff'] and params['RELATIVE_SUSCEPTIBILITY_ADULT'] is not None
                                            else 1.0)
                                            for i, label in enumerate(network_info['node_labels']) ]

    # Assign asymptomatic fractions to each age group:
    pct_asymptomatic = [     params['PCT_ASYMPTOMATIC_STUDENT'] if label == 'student' and params['PCT_ASYMPTOMATIC_STUDENT'] is not None
                        else params['PCT_ASYMPTOMATIC_ADULT']   if label in ['teacher', 'staff'] and params['PCT_ASYMPTOMATIC_ADULT'] is not None
                        else params['PCT_ASYMPTOMATIC']
                        for label in network_info['node_labels'] ]

    latent_period         = utils.gamma_dist(mean=params['LATENT_PERIOD_MEAN'], coeffvar=params['LATENT_PERIOD_CV'], N=N)
    presymptomatic_period = utils.gamma_dist(mean=params['PRESYMPTOMATIC_PERIOD_MEAN'], coeffvar=params['PRESYMPTOMATIC_PERIOD_CV'], N=N)
    symptomatic_period    = utils.gamma_dist(mean=params['SYMPTOMATIC_PERIOD_MEAN'], coeffvar=params['SYMPTOMATIC_PERIOD_CV'], N=N)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Instantiate the model:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model = SARSCoV2NetworkModel(   networks                                 = networks, 
                                    R0                                       = R0,
                                    susceptibility                           = susceptibility,
                                    relative_transmissibility_presymptomatic = params['RELATIVE_TRANSMISSIBILITY_PRESYMPTOMATIC'], 
                                    relative_transmissibility_asymptomatic   = params['RELATIVE_TRANSMISSIBILITY_ASYMPTOMATIC'], 
                                    relative_susceptibility_priorexposure    = params['RELATIVE_SUSCEPTIBILITY_PRIOREXPOSURE'],    
                                    relative_susceptibility_reinfection      = params['RELATIVE_SUSCEPTIBILITY_REMOVED'],
                                    latent_period                            = latent_period,
                                    presymptomatic_period                    = presymptomatic_period,
                                    symptomatic_period                       = symptomatic_period,
                                    pct_asymptomatic                         = pct_asymptomatic,
                                    mixedness                                = params['MIXEDNESS'],
                                    openness                                 = params['OPENNESS'],
                                    track_case_info                          = params['TRACK_CASE_INFO'],
                                    node_groups                              = {'students': network_info['studentIDs'], 'adults': network_info['teacherIDs']+network_info['staffIDs']} 
                                )


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Specify other model configurations:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set that individuals remain active in the 'households' network layer when in isolation:
    model.set_network_activity('households', active_isolation=True)

    # Tests:
    model.update_test_parameters(utils.load_config(params['TEST_PARAMS_CFG']))

    # Vaccines:
    model.add_vaccine(series='covid-vaccine', name='booster', 
                        susc_effectiveness=params['VACCINE_EFFECTIVENESS_SUSCEPTIBILITY'], 
                        transm_effectiveness=params['VACCINE_EFFECTIVENESS_TRANSMISSIBILITY'])

    # Set different asymptomatic rates for vaccinated individuals:
    pct_asymptomatic_vaccinated = [      params['PCT_ASYMPTOMATIC_VACCINATED_STUDENT'] if label == 'student' and params['PCT_ASYMPTOMATIC_VACCINATED_STUDENT'] is not None
                                    else params['PCT_ASYMPTOMATIC_VACCINATED_ADULT']   if label in ['teacher', 'staff'] and params['PCT_ASYMPTOMATIC_VACCINATED_ADULT'] is not None
                                    else params['PCT_ASYMPTOMATIC_VACCINATED']
                                    for label in network_info['node_labels'] ]
    pct_asymptomatic_vaccinated = utils.param_as_array(pct_asymptomatic_vaccinated, (1, N))                                
    model.set_transition_probability('Pv', {'Iv': 1 - pct_asymptomatic_vaccinated, 'Av': pct_asymptomatic_vaccinated})

    # #*************************
    # # Particular assumption about handling susceptibility to reinfection for individuals with prior infection AND vaccination for this particular analysis:
    model.set_susceptibility(['Rv'],  to=['P', 'I', 'A', 'Pv', 'Iv', 'Av'], susceptibility=min(params['RELATIVE_SUSCEPTIBILITY_REMOVED'],       1-params['VACCINE_EFFECTIVENESS_SUSCEPTIBILITY']))
    model.set_susceptibility(['Rpv'], to=['P', 'I', 'A', 'Pv', 'Iv', 'Av'], susceptibility=min(params['RELATIVE_SUSCEPTIBILITY_PRIOREXPOSURE'], 1-params['VACCINE_EFFECTIVENESS_SUSCEPTIBILITY']))

    # #*************************
    # # Enforce that individuals that seek a test upon onset of symptoms are different from the individuals that self-isolate upon sympotoms:
    # isolation_compliance_onset    = utils.param_as_bool_array(params['ISOLATION_COMPLIANCE_ONSET'], n=model.pop_size, selection_mode='choice').ravel()
    # testing_compliance_onset_inds = np.random.choice(np.argwhere(isolation_compliance_onset==False).ravel(), size=int(params['TESTING_COMPLIANCE_ONSET']*model.pop_size), replace=False)
    # testing_compliance_onset      = np.array([True if i in testing_compliance_onset_inds else False for i in range(model.pop_size)], dtype=bool)

    # Allow testing compliance rates to be differest between students and adults:
    # testing_compliance_proactive_students = utils.param_as_bool_array(params['TESTING_COMPLIANCE_PROACTIVE_STUDENT'], n=len(network_info['studentIDs']), selection_mode='choice').ravel()
    # testing_compliance_proactive_adults   = utils.param_as_bool_array(params['TESTING_COMPLIANCE_PROACTIVE_ADULT'], n=len(network_info['teacherIDs']+network_info['staffIDs']), selection_mode='choice').ravel()
    # testing_compliance_proactive          = np.concatenate([testing_compliance_proactive_students, testing_compliance_proactive_adults])


    # Add a node flag with each individual's age group label:
    for i in range(N): 
        model.add_individual_flag(node=i, flag=network_info['node_labels'][i])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set up the initial state:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Introduce a different number of random prior non-omicron spike exposures for each age group:
    # instead of model.set_initial_prevalence()
    model.introduce_random_exposures(num=int(params['INIT_PCT_PRIOR_EXPOSURE_STUDENT']*len(network_info['studentIDs'])), node=network_info['studentIDs'], post_exposure_state='Rp')
    model.introduce_random_exposures(num=int(params['INIT_PCT_PRIOR_EXPOSURE_ADULT']*len(network_info['teacherIDs']+network_info['staffIDs'])), node=network_info['teacherIDs']+network_info['staffIDs'], post_exposure_state='Rp')

    # Administer initial vaccines:
    for label in np.unique(network_info['node_labels']):
        ageGroupIndividuals = model.get_individuals_by_flag(label)
        ageGroupVaccineUptake = (     params['VACCINE_UPTAKE_STUDENT'] if label == 'student' and params['VACCINE_UPTAKE_STUDENT'] is not None
                                 else params['VACCINE_UPTAKE_ADULT']   if label in ['teacher', 'staff'] and params['VACCINE_UPTAKE_ADULT'] is not None
                                 else params['VACCINE_UPTAKE'] )
        model.vaccinate(node=np.random.choice(ageGroupIndividuals, size=int(ageGroupVaccineUptake*len(ageGroupIndividuals)), replace=False), vaccine_series='covid-vaccine')

    # Administer initial masking:
    for label in np.unique(network_info['node_labels']):
        ageGroupIndividuals = model.get_individuals_by_flag(label)
        ageGroupMaskingUptake = (     params['MASK_UPTAKE_STUDENT'] if label == 'student' and params['MASK_UPTAKE_STUDENT'] is not None
                                 else params['MASK_UPTAKE_ADULT']   if label in ['teacher', 'staff'] and params['MASK_UPTAKE_ADULT'] is not None
                                 else params['MASK_UPTAKE'] )
        model.mask(node=np.random.choice(ageGroupIndividuals, size=int(ageGroupMaskingUptake*len(ageGroupIndividuals)), replace=False), 
                    susc_effectiveness=params['MASK_EFFECTIVENESS_SUSCEPTIBILITY'], transm_effectiveness=params['MASK_EFFECTIVENESS_TRANSMISSIBILITY'])

    # Add a 'prior_exposure' flag to each individual with a prior infection:
    # (this flag is on the Rp and Rpv compartments, but adding to individuals
    #  so they will retain this flag after leaving these compartments)
    for i in model.get_individuals_by_compartment(['Rp', 'Rpv']): 
        model.add_individual_flag(node=i, flag='prior_exposure')

    # Introduce a number of random exposures to meet the given init prevalence of removed recovereds:
    model.introduce_random_exposures(int(params['INIT_PCT_REMOVED']*N), post_exposure_state='R')

    # Introduce a number of random exposures to meet the given init prevalence of active infections:
    model.introduce_random_exposures(int(params['INIT_PREVALENCE']*N))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return the constructed model:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return model




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def generate_SARSCoV2_secondary_school_model(parameters=None):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load in model and scenario parameter specifications:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load SEIRS+ default parameters for this scenario:
    params = utils.load_config('scenario_SARSCoV2_secondaryschool.json')
    # Update parameters with any user provided values:
    if(parameters is not None):
        params.update(parameters)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate contact networks:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    NUM_GRADES                      = 4
    NUM_STUDENTS_PER_GRADE          = 500
    NUM_STUDENTS                    = NUM_GRADES*NUM_STUDENTS_PER_GRADE
    NUM_STUDENT_COMMS_PER_GRADE     = int(NUM_STUDENTS_PER_GRADE/10)
    STUDENT_DEGREE                  = 16
    STUDENT_PCT_CONTACTS_INTERGRADE = 0.20
    NUM_STUDENT_BLOCKS              = 1
    NUM_CLASSES_PER_STUDENT         = 6
    NUM_TEACHERS                    = 175
    NUM_STAFF                       = 75
    TEACHERSTAFF_DEGREE             = 12
    NUM_TEACHERSTAFF_COMMS          = 10

    networks, network_info = generate_secondary_school_contact_network(num_grades=NUM_GRADES, num_students_per_grade=NUM_STUDENTS_PER_GRADE, num_communities_per_grade=NUM_STUDENT_COMMS_PER_GRADE,
                                                                       student_mean_intragrade_degree=STUDENT_DEGREE, student_pct_contacts_intergrade=STUDENT_PCT_CONTACTS_INTERGRADE,
                                                                       num_student_blocks=NUM_STUDENT_BLOCKS, block_by_household=True, connect_students_in_households=True, 
                                                                       num_teachers=NUM_TEACHERS, num_staff=NUM_STAFF, num_teacher_staff_communities=NUM_TEACHERSTAFF_COMMS, teacher_staff_degree=TEACHERSTAFF_DEGREE,
                                                                       num_classes_per_student=NUM_CLASSES_PER_STUDENT, classlevel_probs = [[0.8, 0.1, 0.05, 0.05], [0.1, 0.75, 0.1, 0.05], [0.05, 0.1, 0.75, 0.1], [0.05, 0.05, 0.1, 0.8]] )

    networks = {'school': networks['school'], 'households': networks['households']}

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    N = int(networks['school'].number_of_nodes())

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # import matplotlib.pyplot as plt
    # node_colors = ['tab:green' if label=='teacher' else 'tab:orange' if label=='staff' else 'tab:blue' for label in network_info['node_labels']]
    # networkx.draw(networks['school'], pos=networkx.spring_layout(networks['school'], weight='layout_weight'), node_size=20, node_color=node_colors, edge_color='lightgray', alpha=0.5)
    # networkx.draw(networks['school-studentsonly'], pos=networkx.spring_layout(networks['school-studentsonly'], weight='layout_weight'), node_size=20, node_color=node_colors, edge_color='lightgray', alpha=0.5)
    # networkx.draw(networks['school-teacherstaffonly'], pos=networkx.spring_layout(networks['school-teacherstaffonly'], weight='layout_weight'), node_size=20, node_color=node_colors, edge_color='lightgray', alpha=0.5)
    # networkx.draw(networks['households'], pos=networkx.spring_layout(networks['households'], weight='layout_weight'), node_size=20, node_color=node_colors, edge_color='lightgray', alpha=0.5)
    # plt.show()
    # exit()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set up heterogeneous and age-stratified param distributions:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Adjust relative individual R0s for each age group:
    R0 = utils.gamma_dist(mean=params['R0_MEAN'], coeffvar=params['R0_CV'], N=N)
    R0 = [ R0[i] * (     params['RELATIVE_TRANSMISSIBILITY_STUDENT'] if label == 'student' and params['RELATIVE_TRANSMISSIBILITY_STUDENT'] is not None
                    else params['RELATIVE_TRANSMISSIBILITY_ADULT']   if label in ['teacher', 'staff'] and params['RELATIVE_TRANSMISSIBILITY_ADULT'] is not None
                    else 1.0)
                    for i, label in enumerate(network_info['node_labels']) ]

    # Adjust relative susceptibilities for each age group:
    susceptibility = np.ones(N)
    susceptibility = [ susceptibility[i] * (     params['RELATIVE_SUSCEPTIBILITY_STUDENT'] if label == 'student' and params['RELATIVE_SUSCEPTIBILITY_STUDENT'] is not None
                                            else params['RELATIVE_SUSCEPTIBILITY_ADULT']   if label in ['teacher', 'staff'] and params['RELATIVE_SUSCEPTIBILITY_ADULT'] is not None
                                            else 1.0)
                                            for i, label in enumerate(network_info['node_labels']) ]

    # Assign asymptomatic fractions to each age group:
    pct_asymptomatic = [     params['PCT_ASYMPTOMATIC_STUDENT'] if label == 'student' and params['PCT_ASYMPTOMATIC_STUDENT'] is not None
                        else params['PCT_ASYMPTOMATIC_ADULT']   if label in ['teacher', 'staff'] and params['PCT_ASYMPTOMATIC_ADULT'] is not None
                        else params['PCT_ASYMPTOMATIC']
                        for label in network_info['node_labels'] ]

    latent_period         = utils.gamma_dist(mean=params['LATENT_PERIOD_MEAN'], coeffvar=params['LATENT_PERIOD_CV'], N=N)
    presymptomatic_period = utils.gamma_dist(mean=params['PRESYMPTOMATIC_PERIOD_MEAN'], coeffvar=params['PRESYMPTOMATIC_PERIOD_CV'], N=N)
    symptomatic_period    = utils.gamma_dist(mean=params['SYMPTOMATIC_PERIOD_MEAN'], coeffvar=params['SYMPTOMATIC_PERIOD_CV'], N=N)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Instantiate the model:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model = SARSCoV2NetworkModel(   networks                                 = networks, 
                                    R0                                       = R0,
                                    susceptibility                           = susceptibility,
                                    relative_transmissibility_presymptomatic = params['RELATIVE_TRANSMISSIBILITY_PRESYMPTOMATIC'], 
                                    relative_transmissibility_asymptomatic   = params['RELATIVE_TRANSMISSIBILITY_ASYMPTOMATIC'], 
                                    relative_susceptibility_priorexposure    = params['RELATIVE_SUSCEPTIBILITY_PRIOREXPOSURE'],    
                                    relative_susceptibility_reinfection      = params['RELATIVE_SUSCEPTIBILITY_REMOVED'],
                                    latent_period                            = latent_period,
                                    presymptomatic_period                    = presymptomatic_period,
                                    symptomatic_period                       = symptomatic_period,
                                    pct_asymptomatic                         = pct_asymptomatic,
                                    mixedness                                = params['MIXEDNESS'],
                                    openness                                 = params['OPENNESS'],
                                    track_case_info                          = params['TRACK_CASE_INFO'],
                                    node_groups                              = {'students': network_info['studentIDs'], 'adults': network_info['teacherIDs']+network_info['staffIDs']} 
                                )


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Specify other model configurations:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set that individuals remain active in the 'households' network layer when in isolation:
    model.set_network_activity('households', active_isolation=True)

    # Tests:
    model.update_test_parameters(utils.load_config(params['TEST_PARAMS_CFG']))

    # Vaccines:
    model.add_vaccine(series='covid-vaccine', name='booster', 
                        susc_effectiveness=params['VACCINE_EFFECTIVENESS_SUSCEPTIBILITY'], 
                        transm_effectiveness=params['VACCINE_EFFECTIVENESS_TRANSMISSIBILITY'])

    # Set different asymptomatic rates for vaccinated individuals:
    pct_asymptomatic_vaccinated = [      params['PCT_ASYMPTOMATIC_VACCINATED_STUDENT'] if label == 'student' and params['PCT_ASYMPTOMATIC_VACCINATED_STUDENT'] is not None
                                    else params['PCT_ASYMPTOMATIC_VACCINATED_ADULT']   if label in ['teacher', 'staff'] and params['PCT_ASYMPTOMATIC_VACCINATED_ADULT'] is not None
                                    else params['PCT_ASYMPTOMATIC_VACCINATED']
                                    for label in network_info['node_labels'] ]
    pct_asymptomatic_vaccinated = utils.param_as_array(pct_asymptomatic_vaccinated, (1, N))                                
    model.set_transition_probability('Pv', {'Iv': 1 - pct_asymptomatic_vaccinated, 'Av': pct_asymptomatic_vaccinated})

    #*************************
    # Particular assumption about handling susceptibility to reinfection for individuals with prior infection AND vaccination for this particular analysis:
    model.set_susceptibility(['Rv'],  to=['P', 'I', 'A', 'Pv', 'Iv', 'Av'], susceptibility=min(params['RELATIVE_SUSCEPTIBILITY_REMOVED'],   1-params['VACCINE_EFFECTIVENESS_SUSCEPTIBILITY']))
    model.set_susceptibility(['Rpv'], to=['P', 'I', 'A', 'Pv', 'Iv', 'Av'], susceptibility=min(params['RELATIVE_SUSCEPTIBILITY_PRIOREXPOSURE'], 1-params['VACCINE_EFFECTIVENESS_SUSCEPTIBILITY']))

    #*************************
    # Enforce that individuals that seek a test upon onset of symptoms are different from the individuals that self-isolate upon sympotoms:
    # isolation_compliance_onset    = utils.param_as_bool_array(params['ISOLATION_COMPLIANCE_ONSET'], n=model.pop_size, selection_mode='choice').ravel()
    # testing_compliance_onset_inds = np.random.choice(np.argwhere(isolation_compliance_onset==False).ravel(), size=int(params['TESTING_COMPLIANCE_ONSET']*model.pop_size), replace=False)
    # testing_compliance_onset      = np.array([True if i in testing_compliance_onset_inds else False for i in range(model.pop_size)], dtype=bool)

    #*************************
    # Allow testing compliance rates to be differest between students and adults:
    # testing_compliance_proactive_students = utils.param_as_bool_array(params['TESTING_COMPLIANCE_PROACTIVE_STUDENT'], n=len(network_info['studentIDs']), selection_mode='choice').ravel()
    # testing_compliance_proactive_adults   = utils.param_as_bool_array(params['TESTING_COMPLIANCE_PROACTIVE_ADULT'], n=len(network_info['teacherIDs']+network_info['staffIDs']), selection_mode='choice').ravel()
    # testing_compliance_proactive          = np.concatenate([testing_compliance_proactive_students, testing_compliance_proactive_adults])

    # Add a node flag with each individual's age group label:
    for i in range(N): 
        model.add_individual_flag(node=i, flag=network_info['node_labels'][i])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set up the initial state:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # SPECIAL TO THIS PROJECT:
    #*************************
    # Introduce a different number of random prior non-omicron spike exposures for each age group:
    # instead of model.set_initial_prevalence()
    model.introduce_random_exposures(num=int(params['INIT_PCT_PRIOR_EXPOSURE_STUDENT']*len(network_info['studentIDs'])), node=network_info['studentIDs'], post_exposure_state='Rp')
    model.introduce_random_exposures(num=int(params['INIT_PCT_PRIOR_EXPOSURE_ADULT']*len(network_info['teacherIDs']+network_info['staffIDs'])), node=network_info['teacherIDs']+network_info['staffIDs'], post_exposure_state='Rp')

    # Administer initial vaccines:
    for label in np.unique(network_info['node_labels']):
        ageGroupIndividuals = model.get_individuals_by_flag(label)
        ageGroupVaccineUptake = (     params['VACCINE_UPTAKE_STUDENT'] if label == 'student' and params['VACCINE_UPTAKE_STUDENT'] is not None
                                 else params['VACCINE_UPTAKE_ADULT']   if label in ['teacher', 'staff'] and params['VACCINE_UPTAKE_ADULT'] is not None
                                 else params['VACCINE_UPTAKE'] )
        model.vaccinate(node=np.random.choice(ageGroupIndividuals, size=int(ageGroupVaccineUptake*len(ageGroupIndividuals)), replace=False), vaccine_series='covid-vaccine')

    # Administer initial masking:
    for label in np.unique(network_info['node_labels']):
        ageGroupIndividuals = model.get_individuals_by_flag(label)
        ageGroupMaskingUptake = (     params['MASK_UPTAKE_STUDENT'] if label == 'student' and params['MASK_UPTAKE_STUDENT'] is not None
                                 else params['MASK_UPTAKE_ADULT']   if label in ['teacher', 'staff'] and params['MASK_UPTAKE_ADULT'] is not None
                                 else params['MASK_UPTAKE'] )
        model.mask(node=np.random.choice(ageGroupIndividuals, size=int(ageGroupMaskingUptake*len(ageGroupIndividuals)), replace=False), 
                    susc_effectiveness=params['MASK_EFFECTIVENESS_SUSCEPTIBILITY'], transm_effectiveness=params['MASK_EFFECTIVENESS_TRANSMISSIBILITY'])

    # Add a 'prior_exposure' flag to each individual with a prior infection:
    # (this flag is on the Rp and Rpv compartments, but adding to individuals
    #  so they will retain this flag after leaving these compartments)
    for i in model.get_individuals_by_compartment(['Rp', 'Rpv']): 
        model.add_individual_flag(node=i, flag='prior_exposure')

    # Introduce a number of random exposures to meet the given init prevalence of removed recovereds:
    model.introduce_random_exposures(int(params['INIT_PCT_REMOVED']*N), post_exposure_state='R')

    # Introduce a number of random exposures to meet the given init prevalence of active infections:
    model.introduce_random_exposures(int(params['INIT_PREVALENCE']*N))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return the constructed model:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return model





