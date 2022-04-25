# External libraries
import numpy as np
import itertools
# seirsplus libraries
from seirsplus.models.preconfig_disease_models import *
from seirsplus.networks import *
from seirsplus import utils
from seirsplus.sim_loops import *

#------------------------------

# Generic wrapper function for passing in a CompartmentNetworkModel (or compartment config and network) and sim/intervention params dict

# SARSCoV2_generic_population_scenario
# SARSCoV2_primary_school_scenario
# SARSCoV2_secondary_school_scenario
# SARSCoV2_community_scenario

# ^^^^ each of the above call the aforementioned generic wrapper?


# def xxx(model, (or compartment_cfg and networks), intervention_parameters={some default/template json}, sim_reps, ...)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def SARSCoV2_community_scenario(sim_reps=1, parameters=None, metadata={}, outdir='./', run_label=None, save_results=True, save_caselogs=False, save_partial_results=True, results_columns=None, caselog_columns=None, output_file_extn='.csv'):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load in model and scenario parameter specifications:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load SEIRS+ default parameters for this scenario:
    params = utils.load_config('scenario_params_community.json')
    # Update parameters with any user provided values:
    if(parameters is not None):
        params.update(parameters)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Construct parameter sets from combinations of parameters 
    # with multiple values provided ("swept parameters"):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Put all parameter values into a list format (for itertools step below):
    params = {key: ([val] if not isinstance(val, (list, np.ndarray)) else val) for key, val in params.items()}
    paramNames = list(params.keys())
    paramNames_swept = [key for key, val in params.items() if len(val) >  1]
    # Generate a list of the full combinatoric product set of all param value lists in params dict:
    paramSets = list(itertools.product(*params.values())) 

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

    print(f"[ RUNNING SIMULATIONS FOR PARAMETERIZED COMMUNITY SCENARIO ]")

    for paramSetNum, paramSet in enumerate(paramSets):
        paramSetNum += 1 # 1-indexed instead of 0-indexed
        
        PARAM_SET = "paramSet"+str(paramSetNum)

        paramSetDict = metadata
        paramSetDict.update({'PARAM_SET': PARAM_SET })
        paramSetDict.update({paramNames[i]: paramSet[i] for i in range(len(paramSet))})

        print(f"Running simulations for parameter set {paramSetNum}/{len(paramSets)}: {str({paramNames[i]: paramSet[i] for i in range(len(paramSet)) if paramNames[i] in paramNames_swept})}")

        for rep in range(sim_reps):
            rep += 1 # 1-indexed instead of 0-indexed

            metadata.update({'rep': rep, 'run_label': run_label})

            print(f"\tsimulation rep {rep}/{sim_reps}...\t\r", end="")

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Generate contact networks:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            networks, clusters, households, age_groups, node_age_group_labels = generate_community_networks(paramSetDict['N'])

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Apply social distancing to networks:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for network_name, network_obj in networks.items():
                if(network_name != 'household'):
                    apply_social_distancing(network_obj, contact_drop_prob=paramSetDict['SOCIAL_DISTANCING_CONTACT_DROP_PROB'], distancing_compliance=paramSetDict['SOCIAL_DISTANCING_COMPLIANCE'])

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set up heterogeneous and age-stratified param distributions:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Adjust relative individual R0s for each age group:
            R0 = utils.gamma_dist(mean=paramSetDict['R0_MEAN'], coeffvar=paramSetDict['R0_CV'], N=paramSetDict['N'])
            R0 = [ R0[i] * (     paramSetDict['RELATIVE_TRANSMISSIBILITY_AGE0TO4']   if age_group == 'age0-4' and paramSetDict['RELATIVE_TRANSMISSIBILITY_AGE0TO4'] is not None
                            else paramSetDict['RELATIVE_TRANSMISSIBILITY_AGE5TO11']  if age_group == 'age5-11' and paramSetDict['RELATIVE_TRANSMISSIBILITY_AGE5TO11'] is not None
                            else paramSetDict['RELATIVE_TRANSMISSIBILITY_AGE12TO17'] if age_group == 'age12-17' and paramSetDict['RELATIVE_TRANSMISSIBILITY_AGE12TO17'] is not None
                            else paramSetDict['RELATIVE_TRANSMISSIBILITY_AGE18TO24'] if age_group == 'age18-24' and paramSetDict['RELATIVE_TRANSMISSIBILITY_AGE18TO24'] is not None
                            else paramSetDict['RELATIVE_TRANSMISSIBILITY_AGE25TO64'] if age_group in ['age25-29', 'age30-34', 'age35-39', 'age40-44', 'age45-49', 'age50-54', 'age55-59', 'age60-64'] and paramSetDict['RELATIVE_TRANSMISSIBILITY_AGE25TO64'] is not None
                            else paramSetDict['RELATIVE_TRANSMISSIBILITY_AGE65PLUS'] if age_group == 'age65+' and paramSetDict['RELATIVE_TRANSMISSIBILITY_AGE65PLUS'] is not None
                            else 1.0)
                            for i, age_group in enumerate(node_age_group_labels) ]

            # Adjust relative susceptibilities for each age group:
            susceptibility = np.ones(paramSetDict['N'])
            susceptibility = [ susceptibility[i] * (     paramSetDict['RELATIVE_SUSCEPTIBILITY_AGE0TO4']   if age_group == 'age0-4' and paramSetDict['RELATIVE_SUSCEPTIBILITY_AGE0TO4'] is not None
                                                    else paramSetDict['RELATIVE_SUSCEPTIBILITY_AGE5TO11']  if age_group == 'age5-11' and paramSetDict['RELATIVE_SUSCEPTIBILITY_AGE5TO11'] is not None
                                                    else paramSetDict['RELATIVE_SUSCEPTIBILITY_AGE12TO17'] if age_group == 'age12-17' and paramSetDict['RELATIVE_SUSCEPTIBILITY_AGE12TO17'] is not None
                                                    else paramSetDict['RELATIVE_SUSCEPTIBILITY_AGE18TO24'] if age_group == 'age18-24' and paramSetDict['RELATIVE_SUSCEPTIBILITY_AGE18TO24'] is not None
                                                    else paramSetDict['RELATIVE_SUSCEPTIBILITY_AGE25TO64'] if age_group in ['age25-29', 'age30-34', 'age35-39', 'age40-44', 'age45-49', 'age50-54', 'age55-59', 'age60-64'] and paramSetDict['RELATIVE_SUSCEPTIBILITY_AGE25TO64'] is not None
                                                    else paramSetDict['RELATIVE_SUSCEPTIBILITY_AGE65PLUS'] if age_group == 'age65+' and paramSetDict['RELATIVE_SUSCEPTIBILITY_AGE65PLUS'] is not None
                                                    else 1.0)
                                                    for i, age_group in enumerate(node_age_group_labels) ]

            # Assign asymptomatic fractions to each age group:
            pct_asymptomatic = [     paramSetDict['PCT_ASYMPTOMATIC_AGE0TO4']   if age_group == 'age0-4' and paramSetDict['PCT_ASYMPTOMATIC_AGE0TO4'] is not None
                                else paramSetDict['PCT_ASYMPTOMATIC_AGE5TO11']  if age_group == 'age5-11' and paramSetDict['PCT_ASYMPTOMATIC_AGE5TO11'] is not None
                                else paramSetDict['PCT_ASYMPTOMATIC_AGE12TO17'] if age_group == 'age12-17' and paramSetDict['PCT_ASYMPTOMATIC_AGE12TO17'] is not None
                                else paramSetDict['PCT_ASYMPTOMATIC_AGE18TO24'] if age_group == 'age18-24' and paramSetDict['PCT_ASYMPTOMATIC_AGE18TO24'] is not None
                                else paramSetDict['PCT_ASYMPTOMATIC_AGE25TO64'] if age_group in ['age25-29', 'age30-34', 'age35-39', 'age40-44', 'age45-49', 'age50-54', 'age55-59', 'age60-64'] and paramSetDict['PCT_ASYMPTOMATIC_AGE25TO64'] is not None
                                else paramSetDict['PCT_ASYMPTOMATIC_AGE65PLUS'] if age_group == 'age65+' and paramSetDict['PCT_ASYMPTOMATIC_AGE65PLUS'] is not None
                                else paramSetDict['PCT_ASYMPTOMATIC']
                                for age_group in node_age_group_labels ]

            latent_period         = utils.gamma_dist(mean=paramSetDict['LATENT_PERIOD_MEAN'], coeffvar=paramSetDict['LATENT_PERIOD_CV'], N=paramSetDict['N'])
            presymptomatic_period = utils.gamma_dist(mean=paramSetDict['PRESYMPTOMATIC_PERIOD_MEAN'], coeffvar=paramSetDict['PRESYMPTOMATIC_PERIOD_CV'], N=paramSetDict['N'])
            symptomatic_period    = utils.gamma_dist(mean=paramSetDict['SYMPTOMATIC_PERIOD_MEAN'], coeffvar=paramSetDict['SYMPTOMATIC_PERIOD_CV'], N=paramSetDict['N'])

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Instantiate the model:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            model = SARSCoV2NetworkModel(   networks                                 = networks, 
                                            R0                                       = R0,
                                            relative_transmissibility_presymptomatic = paramSetDict['RELATIVE_TRANSMISSIBILITY_PRESYMPTOMATIC'], 
                                            relative_transmissibility_asymptomatic   = paramSetDict['RELATIVE_TRANSMISSIBILITY_ASYMPTOMATIC'], 
                                            susceptibility                           = susceptibility,
                                            relative_susceptibility_priorexposure    = paramSetDict['RELATIVE_SUSCEPTIBILITY_PRIOREXPOSURE'],    
                                            relative_susceptibility_reinfection      = paramSetDict['RELATIVE_SUSCEPTIBILITY_REINFECTION'],
                                            latent_period                            = latent_period,
                                            presymptomatic_period                    = presymptomatic_period,
                                            symptomatic_period                       = symptomatic_period,
                                            pct_asymptomatic                         = pct_asymptomatic,
                                            mixedness                                = paramSetDict['MIXEDNESS'],
                                            openness                                 = paramSetDict['OPENNESS'],
                                            track_case_info                          = paramSetDict['TRACK_CASE_INFO'] )

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Specify other model configurations:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set that individuals remain active in the 'household' network layer when in isolation:
            model.set_network_activity('household', active_isolation=True)

            # Tests:
            model.update_test_parameters(utils.load_config(paramSetDict['TEST_PARAMS_CFG']))

            # Vaccines:
            model.add_vaccine(series='covid-vaccine', name='generic-dose', 
                                susc_effectiveness=paramSetDict['VACCINE_EFFECTIVENESS_SUSCEPTIBILITY'], 
                                transm_effectiveness=paramSetDict['VACCINE_EFFECTIVENESS_TRANSMISSIBILITY'])

            # Set different asymptomatic rates for vaccinated individuals:
            pct_asymptomatic_vaccinated = [      paramSetDict['PCT_ASYMPTOMATIC_VACCINATED_AGE0TO4']   if age_group == 'age0-4' and paramSetDict['PCT_ASYMPTOMATIC_VACCINATED_AGE0TO4'] is not None
                                            else paramSetDict['PCT_ASYMPTOMATIC_VACCINATED_AGE5TO11']  if age_group == 'age5-11' and paramSetDict['PCT_ASYMPTOMATIC_VACCINATED_AGE5TO11'] is not None
                                            else paramSetDict['PCT_ASYMPTOMATIC_VACCINATED_AGE12TO17'] if age_group == 'age12-17' and paramSetDict['PCT_ASYMPTOMATIC_VACCINATED_AGE12TO17'] is not None
                                            else paramSetDict['PCT_ASYMPTOMATIC_VACCINATED_AGE18TO24'] if age_group == 'age18-24' and paramSetDict['PCT_ASYMPTOMATIC_VACCINATED_AGE18TO24'] is not None
                                            else paramSetDict['PCT_ASYMPTOMATIC_VACCINATED_AGE25TO64'] if age_group in ['age25-29', 'age30-34', 'age35-39', 'age40-44', 'age45-49', 'age50-54', 'age55-59', 'age60-64'] and paramSetDict['PCT_ASYMPTOMATIC_VACCINATED_AGE25TO64'] is not None
                                            else paramSetDict['PCT_ASYMPTOMATIC_VACCINATED_AGE65PLUS'] if age_group == 'age65+' and paramSetDict['PCT_ASYMPTOMATIC_VACCINATED_AGE65PLUS'] is not None
                                            else paramSetDict['PCT_ASYMPTOMATIC_VACCINATED']
                                            for age_group in node_age_group_labels ]
            pct_asymptomatic_vaccinated = utils.param_as_array(pct_asymptomatic_vaccinated, (1, paramSetDict['N']))                                
            model.set_transition_probability('Pv', {'Iv': 1 - pct_asymptomatic_vaccinated, 'Av': pct_asymptomatic_vaccinated})

            # Explicit decision about handling susceptibility to reinfection for individuals with prior infection AND vaccination for this particular analysis:
            model.set_susceptibility(['Rv'],  to=['P', 'I', 'A', 'Pv', 'Iv', 'Av'], susceptibility=min(paramSetDict['RELATIVE_SUSCEPTIBILITY_REINFECTION'],   1-paramSetDict['VACCINE_EFFECTIVENESS_SUSCEPTIBILITY']))
            model.set_susceptibility(['Rpv'], to=['P', 'I', 'A', 'Pv', 'Iv', 'Av'], susceptibility=min(paramSetDict['RELATIVE_SUSCEPTIBILITY_PRIOREXPOSURE'], 1-paramSetDict['VACCINE_EFFECTIVENESS_SUSCEPTIBILITY']))

            # Add a node flag with each individual's age group label:
            for i in range(paramSetDict['N']): 
                model.add_individual_flag(node=i, flag=node_age_group_labels[i])

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set up the initial state:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize a specified percentage of individuals to a prior exposure (recovered) state:
            model.set_initial_prevalence('Rp', paramSetDict['INIT_PCT_PRIOR_EXPOSURE'])

            # Administer initial vaccines:
            for age_group in age_groups:
                ageGroupIndividuals = model.get_individuals_by_flag('age'+age_group)
                ageGroupVaccineUptake = (      paramSetDict['VACCINE_UPTAKE_AGE0TO4']   if 'age'+age_group == 'age0-4' and paramSetDict['VACCINE_UPTAKE_AGE0TO4'] is not None
                                          else paramSetDict['VACCINE_UPTAKE_AGE5TO11']  if 'age'+age_group == 'age5-11' and paramSetDict['VACCINE_UPTAKE_AGE5TO11'] is not None
                                          else paramSetDict['VACCINE_UPTAKE_AGE12TO17'] if 'age'+age_group == 'age12-17' and paramSetDict['VACCINE_UPTAKE_AGE12TO17'] is not None
                                          else paramSetDict['VACCINE_UPTAKE_AGE18TO24'] if 'age'+age_group == 'age18-24' and paramSetDict['VACCINE_UPTAKE_AGE18TO24'] is not None
                                          else paramSetDict['VACCINE_UPTAKE_AGE25TO64'] if 'age'+age_group in ['age25-29', 'age30-34', 'age35-39', 'age40-44', 'age45-49', 'age50-54', 'age55-59', 'age60-64'] and paramSetDict['VACCINE_UPTAKE_AGE25TO64'] is not None
                                          else paramSetDict['VACCINE_UPTAKE_AGE65PLUS'] if 'age'+age_group == 'age65+' and paramSetDict['VACCINE_UPTAKE_AGE65PLUS'] is not None
                                          else paramSetDict['VACCINE_UPTAKE'] )
                model.vaccinate(node=np.random.choice(ageGroupIndividuals, size=int(ageGroupVaccineUptake*len(ageGroupIndividuals)), replace=False), 
                                vaccine_series='covid-vaccine')

            # Administer initial masking:
            for age_group in age_groups:
                ageGroupIndividuals = model.get_individuals_by_flag('age'+age_group)
                ageGroupMaskingUptake = (      paramSetDict['MASK_UPTAKE_AGE0TO4']   if 'age'+age_group == 'age0-4' and paramSetDict['MASK_UPTAKE_AGE0TO4'] is not None
                                              else paramSetDict['MASK_UPTAKE_AGE5TO11']  if 'age'+age_group == 'age5-11' and paramSetDict['MASK_UPTAKE_AGE5TO11'] is not None
                                              else paramSetDict['MASK_UPTAKE_AGE12TO17'] if 'age'+age_group == 'age12-17' and paramSetDict['MASK_UPTAKE_AGE12TO17'] is not None
                                              else paramSetDict['MASK_UPTAKE_AGE18TO24'] if 'age'+age_group == 'age18-24' and paramSetDict['MASK_UPTAKE_AGE18TO24'] is not None
                                              else paramSetDict['MASK_UPTAKE_AGE25TO64'] if 'age'+age_group in ['age25-29', 'age30-34', 'age35-39', 'age40-44', 'age45-49', 'age50-54', 'age55-59', 'age60-64'] and paramSetDict['MASK_UPTAKE_AGE25TO64'] is not None
                                              else paramSetDict['MASK_UPTAKE_AGE65PLUS'] if 'age'+age_group == 'age65+' and paramSetDict['MASK_UPTAKE_AGE65PLUS'] is not None
                                              else paramSetDict['MASK_UPTAKE'] )
                model.mask(node=np.random.choice(ageGroupIndividuals, size=int(ageGroupMaskingUptake*len(ageGroupIndividuals)), replace=False), 
                            susc_effectiveness=paramSetDict['MASK_EFFECTIVENESS_SUSCEPTIBILITY'], transm_effectiveness=paramSetDict['MASK_EFFECTIVENESS_TRANSMISSIBILITY'])

            # Add a 'prior_exposure' flag to each individual with a prior infection:
            # (this flag is on the Rp and Rpv compartments, but adding to individuals
            #  so they will retain this flag after leaving these compartments)
            for i in model.get_individuals_by_compartment(['Rp', 'Rpv']): 
                model.add_individual_flag(node=i, flag='prior_exposure')

            # Introduce a number of random exposures to meet the given init prevalence of removed recovereds:
            model.introduce_random_exposures(int(paramSetDict['INIT_PCT_REMOVED']*paramSetDict['N']), post_exposure_state='R')

            # Introduce a number of random exposures to meet the given init prevalence of active infections:
            model.introduce_random_exposures(int(paramSetDict['INIT_PREVALENCE']*paramSetDict['N']))

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Simulate the model scenario:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            run_interventions_sim(model, 
                                    # Scenario run time params:
                                    T = paramSetDict['T'], 
                                    max_dt = 0.25, 
                                    default_dt = 0.1,
                                    tau_step=paramSetDict['TAU_STEP'] if isinstance(paramSetDict['TAU_STEP'], (int, float)) and paramSetDict['TAU_STEP'] > 0 else None,
                                    terminate_at_zero_cases=paramSetDict['TERMINATE_AT_ZERO_CASES'],
                                    # Intervention timing params:
                                    cadence_dt = paramSetDict['CADENCE_DT'], 
                                    cadence_cycle_length = 28,
                                    init_cadence_offset = 'random',
                                    cadence_presets = 'default',
                                    intervention_start_time = 0,
                                    intervention_start_prevalence = 0,
                                    prevalence_flags = ['active_infection'],
                                    onset_flags = ['symptomatic'], 
                                    # Case introduction params:
                                    case_introduction_rate = paramSetDict['CASE_INTRODUCTION_RATE'],
                                    # Network params:
                                    network_active_cadences = {network: 'daily' if network!='household' else 'nightly' for network in networks},
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
                                    isolation_exclude_compartments=[],          
                                    isolation_exclude_flags=[],      
                                    isolation_exclude_isolated=False,           
                                    isolation_exclude_afterNumTests=None,       
                                    isolation_exclude_afterNumVaccineDoses=None,
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
                                    testing_exclude_compartments=[],
                                    testing_exclude_flags=[],
                                    testing_exclude_isolated=True,
                                    testing_exclude_afterNumTests=None,
                                    testing_exclude_afterNumVaccineDoses=None,
                                    # Tracing params:                                                                       
                                    tracing_pct_contacts = paramSetDict['TRACING_PCT_CONTACTS'],
                                    tracing_delay = paramSetDict['TRACING_DELAY'],
                                    tracing_compliance = paramSetDict['TRACING_COMPLIANCE'],
                                    tracing_exclude_networks=['household'],
                                    # Misc. params:
                                    intervention_groups=[hh['indices'] for hh in households],
                                    print_updates=False
                                    )

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update results data with other info:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store model/scenario parameter values with results:
            model.results.update(paramSetDict)
            # Store disease statistics:
            model.results.update(model.disease_stats)
            # Store network statistics:
            overallNetwork = union_of_networks([network for network in networks.values()])
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
                        caselog_df[key] = val
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
                        caselog_df[key] = val
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


# Example calls:

# results, caselogs = SARSCoV2_community_scenario(sim_reps=3, parameters={'T': [10, 20], 'N':[1001, 1002]}, metadata={'lol': 'lmao'}, save_caselogs=True)

# SARSCoV2_community_scenario()



