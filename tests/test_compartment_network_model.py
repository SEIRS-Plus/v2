from hypothesis.strategies._internal.core import sampled_from
from hypothesis.strategies._internal.numbers import integers
from numpy.core.fromnumeric import size
#from seirsplus.models.compartment_model_builder import CompartmentModelBuilder
from seirsplus.models.compartment_network_model import CompartmentNetworkModel
from seirsplus.models.sarscov2_network_model import SARSCoV2NetworkModel
from seirsplus.networks import *
from seirsplus.utils import distributions
import numpy as np
from hypothesis import given, strategies as st, extra

# ------------------------

# Instantiate a FARZ network
N = 200
MEAN_DEGREE = 10
MEAN_CLUSTER_SIZE = 10
CLUSTER_INTERCONNECTEDNESS = 0.25
network, network_info = generate_workplace_contact_network(
    num_cohorts=1,
    num_nodes_per_cohort=N,
    num_teams_per_cohort=int(N / MEAN_CLUSTER_SIZE),
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
networks = {"network": network}


# Instantiate the model:
model = CompartmentNetworkModel(
    compartments="./tests/testsim_scripts/compartments_SARSCoV2_workplacenet.json",
    networks=networks,
    transition_mode="time_in_state",
)

@given(prevalence=st.floats(min_value=0.0, max_value=1.0))
def test_set_initial_prevalence(prevalence):
    model.set_initial_prevalence('E', prevalence)
    assert(model.compartments['E']["initial_prevalence"] == prevalence)

@given(node=st.lists(st.integers(min_value=0, max_value=N-1)), state=st.characters(whitelist_categories='L', whitelist_characters=['S', 'E', 'P', 'I', 'A', 'R']))
def test_set_state(node, state):
    model.set_state(node, state)
    for i in node:
        assert(model.X[i] == state)

@given(rate=st.lists(elements=st.floats(min_value=0.0, max_value=1.0), min_size=N, max_size=N))
def test_set_transition_rate(rate):
    model.set_transition_rate('E', 'S', np.array(rate))
    trans_dict = model.compartments['E']['transitions']
    assert(trans_dict['S']['rate'] == rate)
    assert(trans_dict['S']['time'] == 1/rate)

@given(susceptibility=st.lists(elements=st.floats(min_value=0.0, max_value=1.0), min_size=N, max_size=N))
def test_set_susceptibility(susceptibility):
    model.set_susceptibility('E', 'S', np.array(susceptibility))


@given(time=st.lists(elements=st.integers(min_value=0, max_value=300), min_size=N, max_size=N))
def test_set_transition_time(time):
    model.set_transition_time('S', 'E', np.array(time))


def test_update_data_series():
    pass

def test_increase_data_series_length():
    pass

def test_finalize_data_series():
    pass

@given(
        N=st.integers(min_value=1000, max_value=1000),
        #isolation_period=st.integers(min_value=1, max_value=100),
        T=st.integers(min_value=100, max_value=100),
        max_dt=st.floats(min_value=0.1, max_value=0.1),
        default_dt=st.floats(min_value=0.1, max_value=0.1),
        #cadence_dt=st.floats(min_value=0.1),
        #cadence_cycle_length=st.integers(min_value=1, max_value=28)
)
def test_run_with_interventions(N, T, max_dt, default_dt):
    networks, clusters, households, age_groups, node_labels = generate_community_networks(N)   
    # Instantiate the model:
    intervention_model = CompartmentNetworkModel(
                compartments=load_config("compartments_SARSCoV2_default.json"),
                networks=networks,
                transition_mode="time_in_state",
                isolation_period=10
            )

    # Specify other model configurations:
    intervention_model.set_network_activity('household', active_isolation=True)

    # Set up the initial state:
    intervention_model.set_initial_prevalence("E", 0.01) 
    # Run the model
    model.run_with_interventions(T=T, max_dt=max_dt, default_dt=default_dt,
                                    # Intervention timing params:
                                    cadence_dt=0.5, 
                                    cadence_cycle_length=28,
                                    init_cadence_offset='random',
                                    cadence_presets='default',
                                    intervention_start_time=0,
                                    intervention_start_prevalence=0,
                                    prevalence_flags=['infected'],
                                    onset_flags=['symptomatic'], 
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
                                    isolation_compliance_traced=True,
                                    isolation_exclude_compartments=[],          
                                    isolation_exclude_flags=[],      
                                    isolation_exclude_isolated=False,           
                                    isolation_exclude_afterNumTests=None,       
                                    isolation_exclude_afterNumVaccineDoses=None,
                                    # Testing params:
                                    test_params=load_config("tests_SARSCoV2_default.json"), 
                                    test_type_proactive='pcr',
                                    test_type_onset='pcr',
                                    test_type_traced='pcr', 
                                    proactive_testing_cadence='weekly',
                                    testing_capacity_max=1.0,
                                    testing_capacity_proactive=0.1,
                                    testing_delay_proactive=0,
                                    testing_delay_onset=1,
                                    testing_delay_onset_groupmate=1,
                                    testing_delay_positive_groupmate=1,
                                    testing_delay_traced=1,                                    
                                    testing_compliance_proactive=True,
                                    testing_compliance_onset=True, 
                                    testing_compliance_onset_groupmate=False,
                                    testing_compliance_positive_groupmate=False,
                                    testing_compliance_traced=True,
                                    testing_exclude_compartments=[],
                                    testing_exclude_flags=[],
                                    testing_exclude_isolated=True,
                                    testing_exclude_afterNumTests=None,
                                    testing_exclude_afterNumVaccineDoses=None,
                                    # Tracing params:                                                                       
                                    tracing_pct_contacts=0.5,
                                    tracing_delay=3,
                                    tracing_compliance=True,
                                    # Misc. params:
                                    intervention_groups=None)