from hypothesis.strategies._internal.core import sampled_from
from numpy.core.fromnumeric import size
from seirsplus.models.compartment_model_builder import CompartmentModelBuilder
from seirsplus.models.compartment_network_model import CompartmentNetworkModel
from seirsplus.models.sarscov2_network_model import SARSCoV2NetworkModel
from seirsplus import networks
from seirsplus.utils import distributions
import numpy as np
from hypothesis import given, strategies as st, extra

# ------------------------

# Instantiate a FARZ network
N = 200
MEAN_DEGREE = 10
MEAN_CLUSTER_SIZE = 10
CLUSTER_INTERCONNECTEDNESS = 0.25
network, network_info = networks.generate_workplace_contact_network(
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

@given(rate=st.lists(elements=st.floats(min_value=0.0, max_value=1.0), min_size=N, max_size=N))
def test_set_transition_rate(rate):
    model.set_transition_rate('E', 'S', np.array(rate))

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
