# External Libraries
import numpy as np
import scipy
import networkx

# Internal Libraries
import seirsplus.FARZ as FARZ
from seirsplus.utils import *



def generate_workplace_contact_network(N, num_cohorts=1, num_nodes_per_cohort=100, num_teams_per_cohort=10,
                                        mean_intracohort_degree=6, pct_contacts_intercohort=0.2,
                                        farz_params={'alpha':5.0, 'gamma':5.0, 'beta':0.5, 'r':1, 'q':0.0, 'phi':10, 
                                                     'b':0, 'epsilon':1e-6, 'directed': False, 'weighted': False},
                                        distancing_scales=[]):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate FARZ networks of intra-cohort contacts:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    cohortNetworks = []

    teams_indices = {}

    for i in range(num_cohorts):

        numNodes            = num_nodes_per_cohort[i] if isinstance(num_nodes_per_cohort, list) else num_nodes_per_cohort
        numTeams            = num_teams_per_cohort[i] if isinstance(num_teams_per_cohort, list) else num_teams_per_cohort
        cohortMeanDegree    = mean_intracohort_degree[i] if isinstance(mean_intracohort_degree, list) else mean_intracohort_degree

        farz_params.update({'n':numNodes, 'k':numTeams, 'm':cohortMeanDegree})

        cohortNetwork, cohortTeamLabels = FARZ.generate(farz_params={'n':N, 
                                                    'm':10, 
                                                    'k':100,
                                                    'beta':0.75, 
                                                    'alpha':5.0, 
                                                    'gamma':5.0, 
                                                    'r':1, 
                                                    'q':0.0, 
                                                    'phi':50, 
                                                    'b':0, 'epsilon':1e-6, 'directed': False, 'weighted': False})

        cohortNetworks.append(cohortNetwork)

        for node, teams in cohortTeamLabels.items():
            for team in teams:
                try:
                    teams_indices['c'+str(i)+'-t'+str(team)].append(node)
                except KeyError:
                    teams_indices['c'+str(i)+'-t'+str(team)] = [node]    

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Establish inter-cohort contacts:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    cohortsAdjMatrices = [networkx.adj_matrix(cohortNetwork) for cohortNetwork in cohortNetworks]

    workplaceAdjMatrix = scipy.sparse.block_diag(cohortsAdjMatrices)
    workplaceNetwork   = networkx.from_scipy_sparse_matrix(workplaceAdjMatrix)

    N = workplaceNetwork.number_of_nodes()

    cohorts_indices = {}
    cohortStartIdx  = -1
    cohortFinalIdx  = -1
    for c, cohortNetwork in enumerate(cohortNetworks):

        cohortStartIdx = cohortFinalIdx + 1
        cohortFinalIdx = cohortStartIdx + cohortNetwork.number_of_nodes() - 1
        cohorts_indices['c'+str(c)] = list(range(cohortStartIdx, cohortFinalIdx))

        for team, indices in teams_indices.items():
            if('c'+str(c) in team):
                teams_indices[team] = [idx+cohortStartIdx for idx in indices]

        for i in list(range(cohortNetwork.number_of_nodes())):
            i_intraCohortDegree = cohortNetwork.degree[i]
            i_interCohortDegree = int( ((1/(1-pct_contacts_intercohort))*i_intraCohortDegree)-i_intraCohortDegree )
            # Add intercohort edges:
            if(len(cohortNetworks) > 1):
                for d in list(range(i_interCohortDegree)):
                    j = np.random.choice(list(range(0, cohortStartIdx))+list(range(cohortFinalIdx+1, N)))
                    workplaceNetwork.add_edge(i, j)

    network_info = { 'cohorts_indices': cohorts_indices,
                     'teams_indices':   teams_indices    }

    return workplaceNetwork, network_info


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



def generate_community_networks(
        N, 
        age_brackets_pcts='default',  
        age_brackets_minor='default', 
        age_brackets_adult='default', 
        age_brackets_elder='default',
        hh_size_distn='default', 
        hh_stats='default',
        hh_mixing_matrix='default',
        nonhh_mixing_matrix='default',
        age_brackets_meanOutOfHHDegree='default',
        mean_degree_tolerance=0.1
        ):

    networks   = {}
    clusters   = {}
    age_groups = {}

    print("Generating random community contact network for N=" +str(N))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate the populations' household structure
    # (this determines size of each age group):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #----------------------------------------
    # Load demographic information
    #----------------------------------------
    
    # TODO: Store country data in dataframe(s) and load/read from dataframe here instead of the following dicts.
    country_data = {'default': { # USA data
                        'age_brackets_pcts': {
                            # Data source: 2019 Census, American Community Survey - https://data.census.gov/cedsci/table?q=S01&d=ACS%201-Year%20Estimates%20Subject%20Tables&tid=ACSST1Y2019.S0101
                            '0-4':   0.05911791,
                            '5-11':  0.08609514,
                            '12-17': 0.07708737,
                            '18-24': 0.09253356,
                            # '25-64': 0.52042648,
                            '25-29': 0.07078154,
                            '30-34': 0.06807582,
                            '35-39': 0.06619635,
                            '40-44': 0.06149956,
                            '45-49': 0.06214433,
                            '50-54': 0.06234740,
                            '55-59': 0.06545239,
                            '60-64': 0.06392909,
                            '65+':   0.16473954,
                            },
                        'hh_size_distn': { 
                            # Data source: 2021 Census Bureau Data - https://www.census.gov/data/tables/time-series/demo/families/households.html
                            1: 0.284551031,
                            2: 0.350301314,
                            3: 0.150256675,
                            4: 0.12389653,
                            5: 0.058315567,
                            6: 0.020279995,
                            7: 0.012398889
                            },

                        'hh_stats': {
                            # Data source: UN Database on Household Size and Composition, 2015 IPUMS Data - https://population.un.org/Household/index.html#/countries/840
                            'pct_with_minor': 0.3144,
                            'pct_with_elder': 0.2809,
                            'pct_with_minorandelder': 0.0195,
                            'mean_num_minors_givenAtLeastOneMinor': 1.91,
                            # Data source: 2019 Census, American Community Survey - https://data.census.gov/cedsci/table?q=Household%20Size%20and%20Type&tid=ACSST1Y2019.S2501
                            'pct_with_elder_givenSingleOccupant': 0.4024,
                            },
                        }
                    }
    age_distn     = country_data[age_brackets_pcts]['age_brackets_pcts'] if age_brackets_pcts in country_data else age_brackets_pcts
    hh_size_distn = country_data[hh_size_distn]['hh_size_distn'] if hh_size_distn in country_data else hh_size_distn
    hh_stats      = country_data[hh_stats]['hh_stats'] if hh_stats in country_data else hh_stats
    hh_mixmat     = load_config('mixingmatrix_household_US.csv') if hh_mixing_matrix == 'default' else hh_mixing_matrix
    nonhh_mixmat  = load_config('mixingmatrix_outofhousehold_US.csv') if nonhh_mixing_matrix == 'default' else nonhh_mixing_matrix

    #----------------------------------------
    # Pre-process demographic statistics:
    #----------------------------------------
    meanHHSize = np.average(list(hh_size_distn.keys()), weights=list(hh_size_distn.values()))
    # Calculate the distribution of household sizes given that the household has multiple occupants:
    hh_size_distn_multis = {key: value / (1 - hh_size_distn[1]) for key, value in hh_size_distn.items()}
    hh_size_distn_multis[1] = 0

    #----------------------------------------
    # Define major age groups (under 20, between 20-60, over 60),
    # and calculate age distributions conditional on belonging (or not) to one of these groups:
    #----------------------------------------

    ageBrackets_minor    = ['0-4', '5-11', '12-17'] if age_brackets_minor == 'default' else age_brackets_minor
    ageBrackets_adult    = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64'] if age_brackets_adult == 'default' else age_brackets_adult
    ageBrackets_elder    = ['65+'] if age_brackets_elder == 'default' else age_brackets_elder
    ageBrackets_NOTminor = ageBrackets_adult + ageBrackets_elder
    ageBrackets_NOTelder = ageBrackets_minor + ageBrackets_adult

    totalPct_minor   = np.sum([age_distn[bracket] for bracket in ageBrackets_minor])
    age_distn_minors = {bracket: pct/totalPct_minor for bracket, pct in age_distn.items() if bracket in ageBrackets_minor}
    hhContactProbs_minors = {bracket: hh_mixmat[b][:len(ageBrackets_minor)]/np.sum(hh_mixmat[b][0:len(ageBrackets_minor)]) for b, bracket in enumerate(list(age_distn.keys())) }

    totalPct_adult   = np.sum([age_distn[bracket] for bracket in ageBrackets_adult])
    age_distn_adults = {bracket: pct/totalPct_adult for bracket, pct in age_distn.items() if bracket in ageBrackets_adult}
    hhContactProbs_adults = {bracket: hh_mixmat[b][len(ageBrackets_minor):-len(ageBrackets_elder)]/np.sum(hh_mixmat[b][len(ageBrackets_minor):-len(ageBrackets_elder)]) for b, bracket in enumerate(list(age_distn.keys())) }

    totalPct_elder   = np.sum([age_distn[bracket] for bracket in ageBrackets_elder])
    age_distn_elders = {bracket: pct/totalPct_elder for bracket, pct in age_distn.items() if bracket in ageBrackets_elder}
    hhContactProbs_elders = {bracket: hh_mixmat[b][-len(ageBrackets_elder):]/np.sum(hh_mixmat[b][-len(ageBrackets_elder):]) for b, bracket in enumerate(list(age_distn.keys())) }

    totalPct_NOTminor   = np.sum([age_distn[bracket] for bracket in ageBrackets_NOTminor])
    age_distn_NOTminors = {bracket: pct/totalPct_NOTminor for bracket, pct in age_distn.items() if bracket in ageBrackets_NOTminor}
    hhContactProbs_NOTminors = {bracket: hh_mixmat[b][len(ageBrackets_minor):]/np.sum(hh_mixmat[b][len(ageBrackets_minor):]) for b, bracket in enumerate(list(age_distn.keys())) }

    totalPct_NOTelder   = np.sum([age_distn[bracket] for bracket in ageBrackets_NOTelder])
    age_distn_NOTelders = {bracket: pct/totalPct_NOTelder for bracket, pct in age_distn.items() if bracket in ageBrackets_NOTelder}
    hhContactProbs_NOTelders = {bracket: hh_mixmat[b][:-len(ageBrackets_elder)]/np.sum(hh_mixmat[b][:-len(ageBrackets_elder)]) for b, bracket in enumerate(list(age_distn.keys())) }

    #----------------------------------------
    # Calculate the probabilities of household contexts 
    # based on age groups and single/multi-occupancy:
    #----------------------------------------
    
    prob_minorHH  = hh_stats['pct_with_minor']
    prob_elderHH  = hh_stats['pct_with_elder']
    prob_singleHH = hh_size_distn[1]
    prob_multiHH  =  1 - prob_singleHH

    hhContexts_probs = {}
    hhContexts_probs['minor_elder_single']       = 0  # can't have both a minor and an elder in a household with only 1 occupant
    hhContexts_probs['minor_NOTelder_single']    = 0  # assume no minors live on their own (data suggests <1% actually do)
    hhContexts_probs['NOTminor_elder_single']    = (hh_stats['pct_with_elder_givenSingleOccupant'] * prob_singleHH)
    hhContexts_probs['NOTminor_NOTelder_single'] = (1 - hh_stats['pct_with_elder_givenSingleOccupant']) * prob_singleHH
    hhContexts_probs['minor_elder_multi']        = hh_stats['pct_with_minorandelder']
    hhContexts_probs['minor_NOTelder_multi']     = (prob_minorHH - hhContexts_probs['minor_elder_multi'] - hhContexts_probs['minor_NOTelder_single'] - hhContexts_probs['minor_elder_single'])
    hhContexts_probs['NOTminor_elder_multi']     = (prob_elderHH - hhContexts_probs['minor_elder_multi'] - hhContexts_probs['NOTminor_elder_single'] - hhContexts_probs['minor_elder_single'])
    hhContexts_probs['NOTminor_NOTelder_multi']  = (prob_multiHH - hhContexts_probs['minor_elder_multi'] - hhContexts_probs['NOTminor_elder_multi'] - hhContexts_probs['minor_NOTelder_multi'])
    assert (np.sum(list(hhContexts_probs.values())) == 1.0), "Household context probabilities do not sum to 1"

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Randomly construct households following the size and age distributions defined above:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("\tPopulating age groups and constructing households...")

    households     = []  # List of dicts storing household data structures and age_groups
    numHouseless   = N   # Number of individuals still to place in households
    curMemberIndex = 0
    while(numHouseless > 0):

        household = {}

        household['context'] = np.random.choice(list(hhContexts_probs.keys()), p=list(hhContexts_probs.values()))

        household['age_brackets'] = []

        if household['context'] == "NOTminor_elder_single":
            # Household size is definitely 1
            household['size'] = 1
            # There is only 1 occupant in this household, and they are ELDER; add them:
            household['age_brackets'].append( np.random.choice(list(age_distn_elders.keys()), p=list(age_distn_elders.values())) )
        #--------------------
        elif household['context'] == "NOTminor_NOTelder_single":
            # Household size is definitely 1
            household['size'] = 1
            # There is only 1 occupant in this household, and they are ADULT; add them:
            household['age_brackets'].append( np.random.choice(list(age_distn_adults.keys()), p=list(age_distn_adults.values())) )
        

        #--------------------
        elif household['context'] == "minor_elder_multi":
            # There are at least two occupants, draw the household size:
            household['size'] = min( numHouseless, max(2, np.random.choice(list(hh_size_distn_multis), p=list(hh_size_distn_multis.values()))) )
            # There's definitely at least one MINOR in this household, add one from an appropriate age bracket:
            household['age_brackets'].append( np.random.choice(list(age_distn_minors.keys()), p=list(age_distn_minors.values())) )
            # Figure out how many additional minors to add given there is at least one minor; add them:
            # > Must leave room for at least one elder (see minmax terms)
            numAdditionalMinors_givenAtLeastOneMinor = min( max(0, np.random.poisson(hh_stats['mean_num_minors_givenAtLeastOneMinor']-1)), household['size']-len(household['age_brackets']) - 1 )
            for k in range(numAdditionalMinors_givenAtLeastOneMinor):
                household['age_brackets'].append( np.random.choice(ageBrackets_minor, p=hhContactProbs_minors[household['age_brackets'][0]]) )
            # There's definitely at least one ELDER in this household, add one from an appropriate age bracket:
            household['age_brackets'].append( np.random.choice(ageBrackets_elder, p=hhContactProbs_elders[household['age_brackets'][0]]) )
            # Any remaining occupants can be any age EXCLUDING minors (all minors already added):
            for m in range(household['size'] - len(household['age_brackets'])):
                household['age_brackets'].append( np.random.choice(ageBrackets_NOTminor, p=hhContactProbs_NOTminors[household['age_brackets'][0]]) )
        

        #--------------------
        elif household['context'] == "minor_NOTelder_multi":
            # There are at least two occupants, draw the household size:
            household['size'] = min( numHouseless, max(2, np.random.choice(list(hh_size_distn_multis), p=list(hh_size_distn_multis.values()))) )
            # There's definitely at least one MINOR in this household, add one from an appropriate age bracket:
            household['age_brackets'].append( np.random.choice(list(age_distn_minors.keys()), p=list(age_distn_minors.values())) )
            # Figure out how many additional minors to add given there is at least one minor; add them:
            # > NOT CURRENTLY ASSUMING that there must be at least one non-minor occupant in every household (doing so makes total % minor in households too low)
            numAdditionalMinors_givenAtLeastOneMinor = min( max(0, np.random.poisson(hh_stats['mean_num_minors_givenAtLeastOneMinor']-1)), household['size']-len(household['age_brackets']) )
            for k in range(numAdditionalMinors_givenAtLeastOneMinor):
                household['age_brackets'].append( np.random.choice(ageBrackets_minor, p=hhContactProbs_minors[household['age_brackets'][0]]) )
            # There are no ELDERs in this household.
            # Remaining occupants can be any age EXCLUDING ELDER and EXCLUDING MINOR (all minors already added):
            for m in range(household['size'] - len(household['age_brackets'])):
                household['age_brackets'].append( np.random.choice(ageBrackets_adult, p=hhContactProbs_adults[household['age_brackets'][0]]) )
        

        #--------------------
        elif household['context'] == "NOTminor_elder_multi":
            # There are at least two occupants, draw the household size:
            household['size'] = min( numHouseless, max(2, np.random.choice(list(hh_size_distn_multis), p=list(hh_size_distn_multis.values()))) )
            # There are no MINORs in this household.
            # There's definitely at least one ELDER in this household, add one from an appropriate age bracket:
            household['age_brackets'].append( np.random.choice(list(age_distn_elders.keys()), p=list(age_distn_elders.values())) )
            # Any remaining occupants can be any age EXCLUDING MINOR:
            for m in range(household['size'] - len(household['age_brackets'])):
                household['age_brackets'].append( np.random.choice(ageBrackets_NOTminor, p=hhContactProbs_NOTminors[household['age_brackets'][0]]) )
        

        #--------------------
        elif household['context'] == "NOTminor_NOTelder_multi":
            # There are at least two occupants, draw the household size:
            household['size'] = min( numHouseless, max(2, np.random.choice(list(hh_size_distn_multis), p=list(hh_size_distn_multis.values()))) )
            # There are no MINORs in this household.
            # There are no ELDERs in this household.
            # Draw a first ADULT occupant:
            household['age_brackets'].append( np.random.choice(list(age_distn_adults.keys()), p=list(age_distn_adults.values())) )
            # Remaining household occupants can be any ADULT age, add as many as needed to meet the household size:
            for m in range(household['size'] - len(household['age_brackets'])):
                household['age_brackets'].append( np.random.choice(ageBrackets_adult, p=hhContactProbs_adults[household['age_brackets'][0]]) )
        

        #--------------------
        elif(household['context'] == 'minor_NOTelder_single'):
           pass # impossible by assumption
        elif(household['context'] == 'minor_elder_single'):
           pass # impossible by definition

        # Advance household placement loop:
        if len(household['age_brackets']) == household['size']:
            numHouseless -= household['size']
            households.append(household)
        else:
            print("Household size does not match number of age brackets assigned. ("+household['context']+")")

    numHouseholds = len(households)

    #----------------------------------------

    # Check the frequencies of constructed households against the target distributions:
    # print("Generated overall age distribution:")
    # for ageBracket in sorted(age_distn):
    #     age_freq = (np.sum([len([age for age in household['age_brackets'] if age == ageBracket]) for household in households])/N)
    #     print(str(ageBracket) + ": %.4f\t(%.4f from target)" % (age_freq, (age_freq - age_distn[ageBracket])))
    # print()
    # print("Generated household size distribution:")
    # for size in sorted(hh_size_distn):
    #     size_freq = (np.sum([1 for household in households if household['size'] == size])/numHouseholds)
    #     print(str(size) + ": %.4f\t(%.4f from target)" % (size_freq, (size_freq - hh_size_distn[size])))
    # print("Num households: " + str(numHouseholds))
    # print("mean household size: " + str(meanHHSize))
    # print()
    # if True:
    #     print("Generated percent households with at least one minor:")
    #     checkval = (len([household for household in households if not set(household['age_brackets']).isdisjoint(ageBrackets_minor)])/numHouseholds)
    #     target = hh_stats['pct_with_minor']
    #     print("%.4f\t\t(%.4f from target)" % (checkval, checkval - target))
    #     print("Generated percent households with at least one elder")
    #     checkval = (len([household for household in households if not set(household['age_brackets']).isdisjoint(ageBrackets_elder)])/numHouseholds)
    #     target = hh_stats['pct_with_elder']
    #     print("%.4f\t\t(%.4f from target)" % (checkval, checkval - target))
    #     print("Generated percent households with at least one minor AND elder")
    #     checkval = (len([household for household in households if not set(household['age_brackets']).isdisjoint(ageBrackets_elder) and not set(household['age_brackets']).isdisjoint(ageBrackets_minor)])/numHouseholds)
    #     target = hh_stats['pct_with_minorandelder']
    #     print("%.4f\t\t(%.4f from target)" % (checkval, checkval - target))
    #     print("Generated percent households with single occupant who is elder")
    #     checkval = (np.sum([1 for household in households if household['size'] == 1 and not set(household['age_brackets']).isdisjoint(ageBrackets_elder)])/numHouseholds)
    #     target = hh_stats['pct_with_elder_givenSingleOccupant'] * prob_singleHH
    #     print("%.4f\t\t(%.4f from target)" % (checkval, checkval - target))
    #     print("Generated mean num minor occupants given at least one occupant is minor")
    #     checkval = np.mean([np.in1d(household['age_brackets'], ageBrackets_minor).sum() for household in households if not set(household['age_brackets']).isdisjoint(ageBrackets_minor)])
    #     target = hh_stats['mean_num_minors_givenAtLeastOneMinor']
    #     print("%.4f\t\t(%.4f from target)" % (checkval, checkval - target))


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Update age_groups about the population of each age group:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    curidx = 0
    node_labels = [] 

    for bracket in age_distn.keys():

        age_groups[bracket] = {}

        # Count the number of individuals in each age bracket in the generated households:
        age_groups[bracket]['size'] = np.sum([len([age for age in household['age_brackets'] if age == bracket]) for household in households]) 

        # Store and label the node indices that are part of this age group layer:
        age_groups[bracket]['indices'] = list(range(curidx, curidx + age_groups[bracket]['size']))
        curidx += age_groups[bracket]['size']
        # Store label of age assigned to each node:
        if(len(age_groups[bracket]['indices']) > 0):
            node_labels[min(age_groups[bracket]['indices']) : max(age_groups[bracket]['indices'])] = ['age'+bracket] * age_groups[bracket]['size']


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate a contact network layer representing
    # out-of-household contacts for each stratified age group:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if(age_brackets_meanOutOfHHDegree == 'default'):
        age_brackets_meanOutOfHHDegree = {# Data source: See [SEIRS+ model description paper]
                                            '0-4':   5.7,
                                            '5-11':  13.0,
                                            '12-17': 14.0,
                                            '18-24': 10.2,
                                            '25-29': 9.6,
                                            '30-34': 9.9,
                                            '35-39': 9.8,
                                            '40-44': 10.5,
                                            '45-49': 8.3,
                                            '50-54': 7.5,
                                            '55-59': 6.8,
                                            '60-64': 6.2,
                                            '65+':   4.5
                                        }

    for bracket, targetMeanDegree in age_brackets_meanOutOfHHDegree.items():
        print("\tGenerating out-of-household contact network layer for", bracket, "age group...")

        graph_generated       = False
        graph_gen_attempts    = 0
        tolerance_relaxations = 0

        targetMeanDegreeRange = (targetMeanDegree - mean_degree_tolerance, targetMeanDegree + mean_degree_tolerance)

        degree_param = targetMeanDegree

        while not graph_generated:

                subnetwork_layer, clusts = FARZ.generate(farz_params={
                                                    "n":        age_groups[bracket]['size'],
                                                    "m":        int(degree_param),
                                                    "k":        max(1, int(age_groups[bracket]['size']/(1*targetMeanDegree))),  # num clusters; arbitrarily assume mean community size 3*meanDegree
                                                    "alpha":    5.0,    # cluster coeff param
                                                    "gamma":    5.0,    # assortativity param
                                                    "r":        2,      # max num clusters node can be part of
                                                    "q":        0.5,    # probability of multi-community membership
                                                    "beta":     0.8,   # prob within community edges
                                                    "phi":      50,     # higher value makes clusters more balanced in size, 1 results in power law size distribution
                                                    "b":        0,
                                                    "epsilon":  1e-6,
                                                    "directed": False,
                                                    "weighted": False
                                                    }
                                                )

                # Compute the mean degree of the generated FARZ (sub)network layer:
                nodeDegrees = [d[1] for d in subnetwork_layer.degree()]
                meanDegree  = np.mean(nodeDegrees)        
        
                # Enforce that the generated graph has mean degree within the given tolerance:
                if (meanDegree >= targetMeanDegreeRange[0] and meanDegree <= targetMeanDegreeRange[1]):
                    # Successful (sub)network layer generation with mean degree in target range.
                    #----------------------------------------
                    # The generated (sub)network has num nodes equal to size of age group,
                    # but needs to be embedded in a network layer with N (pop size) nodes:
                    #----------------------------------------
                    # Get the adjacency matrix for the (sub)network:
                    subnetwork_layer_adj = networkx.adj_matrix(subnetwork_layer)
                    # Create a full N-sized adjacency matrix by placing this bracket's 
                    #adjacency block alongside empty blocks for the other age brackets:
                    bracket_blocks    = []
                    for other_bracket in age_groups:
                        if(other_bracket == bracket):
                            bracket_blocks.append(subnetwork_layer_adj)
                        else:
                            bracket_blocks.append( scipy.sparse.csr_matrix((age_groups[other_bracket]['size'], age_groups[other_bracket]['size'])) ) # creates empty sparse matrix with shape (n,n)
                    network_layer_adj = scipy.sparse.block_diag(bracket_blocks)
                    # Create a new networkx obj for the full N-sized network layer:
                    network_layer     = networkx.from_scipy_sparse_matrix(network_layer_adj)
                    # Store the networkx obj for this layer (for return):
                    networks[bracket] = network_layer
                    
                    #----------------------------------------
                    # Store the mean degree for this layer (for return):
                    # This should maybe be done later after mixing edges are added in: age_groups[bracket]['mean_degree'] = meanDegree
                    #----------------------------------------
                    # Reformat and store community memberships for this layer (for return):
                    clusters[bracket]  = {}
                    for individual, cluster_ids in clusts.items():
                        for cluster_id in cluster_ids:
                            if(cluster_id in clusters[bracket]):
                                clusters[bracket][cluster_id]['indices'].append(individual)
                                clusters[bracket][cluster_id]['size'] += 1
                            else:
                                clusters[bracket][cluster_id] = {'indices':[individual], 'size':1}
                    # Exit while loop upon successful network_layer generation
                    graph_generated = True
                else:
                    graph_gen_attempts += 1
                    # print("attempt", graph_gen_attempts, ":", "relaxation", tolerance_relaxations)
                    # print("\tTry again... (mean degree = %.2f is outside the target range for mean degree (%.2f, %.2f)"
                    #         % ( meanDegree, targetMeanDegreeRange[0], targetMeanDegreeRange[1] ) )
                    if(graph_gen_attempts % 5 == 0):
                        # Relax the tolerance after every 5 failed attempts
                        tolerance_relaxations += 1
                        try:
                            targetMeanDegreeRange = (targetMeanDegree - mean_degree_tolerance*2**tolerance_relaxations, 
                                                     targetMeanDegree + mean_degree_tolerance*2**tolerance_relaxations)
                        except OverflowError:
                            targetMeanDegreeRange = (0, N)
                        degree_param = targetMeanDegree
                    else:
                        if(meanDegree < targetMeanDegreeRange[0]):
                            degree_param += 1 
                        elif(meanDegree > targetMeanDegreeRange[1]):
                            degree_param -= 1 

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate a network layer representing
    # out-of-household contacts *between* age groups:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("\tGenerating contact network layer for out-of-household age group mixing...")

    # Instantiate an empty graph on which to construct the mixing network layer:
    mixing_network_layer = networkx.empty_graph(N)

    ageBrackets = list(age_groups.keys())
    for b, bracket in enumerate(ageBrackets):
        # Get the frequencies with which individuals of this age group 
        # contact individuals of other age groups out of the household:
        mixing_probs = nonhh_mixmat[b]/np.sum(nonhh_mixmat[b])
        # Get the within-age-group network layer for this age bracket:
        bracket_network = networks[bracket]
        #----------------------------------------
        # For each edge in this bracket's network layer;
        # Draw what age group this individual "should" be making contact with given the mixing probs. 
        # If the mixing-appropriate contact is in another age group, drop the within-bracket contact 
        # and create a new contact in the mixing network layer.
        #----------------------------------------
        for node_i, contact_j in bracket_network.edges:
            node_ageGroup = node_labels[node_i]#.split('_')[-1] # labels are in age_X-Y format
            mixed_contact_ageGroup = np.random.choice(ageBrackets, p=mixing_probs)
            if('age'+mixed_contact_ageGroup != node_ageGroup):
                bracket_network.remove_edge(node_i, contact_j)
                mixing_network_layer.add_edge(node_i, np.random.choice(age_groups[mixed_contact_ageGroup]['indices']))

    networks['mixing'] = mixing_network_layer

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate a network layer representing
    # within household contacts for each household:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("\tGenerating within-household contact network layer...")

    # Create a copy of the list of node indices for each age group (graph layer) to draw from:
    sel_indices = {}
    for bracket, layer_data in age_groups.items(): 
        sel_indices[bracket] = list(np.random.permutation( age_groups[bracket]['indices'] )) # np.random.permutation randomizes order

    individualAgeBracketLabels = [None] * N

    # Instantiate an empty graph on which to construct the households network layer:
    hh_network_layer = networkx.empty_graph(N)

    # Go through each household, look up what the age brackets of the members should be,
    # and randomly select nodes from corresponding age groups (graph layers) to place in the given household.
    # Strongly connect the nodes selected for each household by adding edges to the adjacency matrix.
    for household in households:
        
        household['indices'] = []
        
        for hhAgeBracket in household['age_brackets']:
            ageBracketIndices = sel_indices[hhAgeBracket]
            memberIndex       = ageBracketIndices.pop()
            household['indices'].append(memberIndex)
            individualAgeBracketLabels[memberIndex] = hhAgeBracket

        for memberIdx in household['indices']:
            nonselfIndices = [i for i in household['indices'] if memberIdx != i]
            for housemateIdx in nonselfIndices:
                # Add edges to households network layer:
                hh_network_layer.add_edge(memberIdx, housemateIdx)

    networks['household'] = hh_network_layer

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    return networks, clusters, households, age_groups, node_labels






# N = 50000
# networks, clusters, households, age_groups, node_labels = generate_community_networks(N)

# # print(networks)
# # print()
# # print(clusters)
# # print()
# # print(households)
# # print()
# # print(age_groups)
# # print()
# # print(node_labels)


# for net_id, network in networks.items():
#     try:
#         print(net_id, "mean_degree =", np.sum([d[1] for d in network.degree()])/age_groups[net_id]['size'] + 2.2)
#     except:
#         pass

# overallNetwork = networkx.empty_graph(N)
# for net_id, network in networks.items():
#     overallNetwork = networkx.compose(overallNetwork, network)
# print("overall mean_degree =", np.mean([d[1] for d in overallNetwork.degree()]))




# network_stats(overallNetwork,  plot=True)


# # exit()  


# ageBrackets = list(age_groups.keys())

# ageBracket_contactMatrix = np.zeros( (len(ageBrackets), len(ageBrackets)) )
# for b_i, bracket in enumerate(ageBrackets):
#     print(''+bracket)
#     if('+' in bracket):
#         continue
#     n = age_groups[bracket]['size']
#     # bracket_network = networks[bracket]
#     for i in age_groups[bracket]['indices']:
#         # Get the contacts of individual i:
#         # print(i)
#         contacts = list(networks['household'].neighbors(i))
#         # print(contacts)

#         contacts_ages = list(np.array(node_labels)[contacts])
#         # print(contacts_ages)
        
#         # Count the occurrences of each age group among 
#         for b_j, other_bracket in enumerate(ageBrackets):
#             if('+' in other_bracket):
#                 continue
#             ageBracket_contactMatrix[b_i, b_j] += contacts_ages.count('age_'+other_bracket)/n
#             # print('\t', other_bracket, contacts_ages.count('age_'+other_bracket))

# import matplotlib.pyplot as plt
# plt.imshow(ageBracket_contactMatrix, cmap='Blues')
# plt.show()



# ageBracket_contactMatrix = np.zeros( (len(ageBrackets), len(ageBrackets)) )
# for b_i, bracket in enumerate(ageBrackets):
#     print(''+bracket)
#     if('+' in bracket):
#         continue
#     n = age_groups[bracket]['size']
#     # bracket_network = networks[bracket]
#     for i in age_groups[bracket]['indices']:
#         # Get the contacts of individual i:
#         # print(i)
#         contacts = list(networks['mixing'].neighbors(i))
#         # print(contacts)

#         contacts_ages = list(np.array(node_labels)[contacts])
#         # print(contacts_ages)
        
#         # Count the occurrences of each age group among 
#         for b_j, other_bracket in enumerate(ageBrackets):
#             if('+' in other_bracket):
#                 continue
#             ageBracket_contactMatrix[b_i, b_j] += contacts_ages.count('age_'+other_bracket)/n
#             # print('\t', other_bracket, contacts_ages.count('age_'+other_bracket))

# import matplotlib.pyplot as plt
# plt.imshow(ageBracket_contactMatrix, cmap='Blues')
# plt.show()



# ageBracket_contactMatrix = np.zeros( (len(ageBrackets), len(ageBrackets)) )
# for b_i, bracket in enumerate(ageBrackets):
#     print(''+bracket)
#     if('+' in bracket):
#         continue
#     n = age_groups[bracket]['size']
#     # bracket_network = networks[bracket]
#     for i in age_groups[bracket]['indices']:
#         # Get the contacts of individual i:
#         # print(i)
#         contacts = list(overallNetwork.neighbors(i))
#         # print(contacts)

#         contacts_ages = list(np.array(node_labels)[contacts])
#         # print(contacts_ages)
        
#         # Count the occurrences of each age group among 
#         for b_j, other_bracket in enumerate(ageBrackets):
#             if('+' in other_bracket):
#                 continue
#             ageBracket_contactMatrix[b_i, b_j] += contacts_ages.count('age_'+other_bracket)/n
#             # print('\t', other_bracket, contacts_ages.count('age_'+other_bracket))

# import matplotlib.pyplot as plt
# plt.imshow(ageBracket_contactMatrix, cmap='Blues')
# plt.show()


def apply_social_distancing(network, contact_drop_prob, distancing_compliance=True):
    # Initialize social distancing compliances:
    distancing_compliance = param_as_bool_array(distancing_compliance, (1, network.number_of_nodes())).flatten()
    # Store compliances as node attributes in the model object (e.g., for case logging purposes)
    # model.set_node_attribute(node=list(range(network.number_of_nodes())), attribute_name='distancing_compliance', attribute_value=distancing_compliance)
    # Get lists of pre-distancing edges for all nodes:
    edges_orig = [list(network.edges(node)) for node in network.nodes()]
    # Go through individuals and randomly drop the designated proportion of their edges (if compliant):
    for i in range(network.number_of_nodes()):
        if(distancing_compliance[i]==True):
            for edge in edges_orig[i]:    
                if(np.random.rand() < contact_drop_prob):
                    try:
                        network.remove_edge(edge[0], edge[1])
                    except networkx.NetworkXError: 
                        # this error type is raised when the edge passed to remove_edge does not exist
                        pass
        








