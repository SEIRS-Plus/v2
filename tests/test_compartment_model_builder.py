from seirsplus.models.compartment_model_builder import *
from seirsplus import networks
from seirsplus.utils import distributions
from seirsplus.utils.distributions import gamma_dist
from hypothesis.extra import numpy
from hypothesis import given, strategies as st

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
compartmentModel = CompartmentModelBuilder()

latent_period = distributions.gamma_dist(mean=3.0, coeffvar=0.6, N=N)
presymptomatic_period = distributions.gamma_dist(mean=2.2, coeffvar=0.5, N=N)
symptomatic_period = distributions.gamma_dist(mean=4.0, coeffvar=0.4, N=N)
infectious_period = presymptomatic_period + symptomatic_period
pct_asymptomatic = 0.3

R0_mean = 3.0
R0_cv = 2.0
R0 = distributions.gamma_dist(R0_mean, R0_cv, N)
transmissibility = 1 / infectious_period * R0

def test_compartment_model_builder(test_compartment=compartmentModel):
    assert(isinstance(test_compartment.compartments, dict))

def test_add_compartments():
    compartments_list = ["S", "E", "P", "I", "A", "R"]
    compartmentModel.add_compartments(compartments_list)
    assert(len(compartmentModel.compartments) == 6)
    for key in compartmentModel.compartments:
        assert(key in compartments_list)

@given(m=st.floats(0.01, 10), co=st.floats(0.01, 10))
def test_gamma_dist(m, co):
    dist = distributions.gamma_dist(m, co, N)
    assert(len(dist) == N)

@given(s=st.floats(0.01))
def test_set_susceptibility(s):
    test_compartment = CompartmentModelBuilder()
    test_compartment.add_compartments(["S", "E", "P", "I", "A", "R"])
    test_compartment.set_susceptibility("S", to=["P", "I", "A"], susceptibility=s)

@given(t=st.floats(0.01))
def test_set_transmissibility(t):
    test_compartment = CompartmentModelBuilder()
    test_compartment.add_compartments(["S", "E", "P", "I", "A", "R"])
    test_compartment.set_susceptibility("S", to=["P", "I", "A"], susceptibility=1.0)
    test_compartment.set_transmissibility(["P", "I", "A"], "network", transmissibility=t)

@given(p=st.floats(0.01), latent_period_test=st.floats(0.01), pres_per=st.floats(0.01), pct_asympt=st.floats(0.01,1.0), symp_per=st.floats(0.01))
def test_add_transition(p, latent_period_test, pres_per, pct_asympt, symp_per):
    test_compartment = CompartmentModelBuilder()
    test_compartment.add_compartments(["S", "E", "P", "I", "A", "R"])
    test_compartment.set_susceptibility("S", to=["P", "I", "A"], susceptibility=1.0)
    test_compartment.set_transmissibility(["P", "I", "A"], "network", transmissibility=0.0)

    test_compartment.add_transition(
    "S", to="E", upon_exposure_to=["P", "I", "A"], prob=p
    )
    test_compartment.add_transition("E", to="P", time=latent_period_test, prob=p)
    test_compartment.add_transition(
        "P", to="I", time=pres_per, prob=1 - pct_asympt
    )
    test_compartment.add_transition(
        "P", to="A", time=pres_per, prob=pct_asympt
    )
    test_compartment.add_transition("I", to="R", time=symp_per, prob=p)
    test_compartment.add_transition("A", to="R", time=symp_per, prob=p)

