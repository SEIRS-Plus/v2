import mlflow
mlflow.set_tracking_uri(uri='databricks')

from seirsplus.models.compartment_model_builder import CompartmentModelBuilder
from seirsplus.models.compartment_network_model import CompartmentNetworkModel
from seirsplus.networks import *
from seirsplus.utils import distributions
from seirsplus.utils.io import *
from seirsplus.scenarios import *
import matplotlib.pyplot as plt
import numpy as np

# ------------------------

# Create MLflow Client
client = mlflow.tracking.MlflowClient()
# experiments = client.list_experiments()

# experiment = client.create_experiment(name='/Shared/proactive-testing')


testing_capacities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
tracing_coverage   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


with mlflow.start_run(experiment_id='2741365028337245') as run:
    for testcap in testing_capacities:
        for tracecov in tracing_coverage:


            submit a job to hyak that has 1000  reps



            with mlflow.start_run(nested=True, experiment_id='2741365028337245') as nested_run:
                mlflow.log_param('testing_capacity', testcap)
                mlflow.log_param('tracing_coverage', tracecov)
                mlflow.set_tags({
                    'type': 'network',
                    'problem': 'hospitals overrun',
                })

    mlflow.log_param('best_run', 3)


exit()

N = 1000

networks, clusters, households, age_groups, node_labels = generate_community_networks(N)


# Instantiate the model:
model = CompartmentNetworkModel(
            compartments=load_config("compartments_SARSCoV2_default.json"),
            networks=networks,
            transition_mode="time_in_state",
            isolation_period=10
        )

# Specify other model configurations:
model.set_network_activity('household', active_isolation=True)

# Set up the initial state:
model.set_initial_prevalence("E", 0.01)

# Run the model
# model.run_with_interventions(T=100, max_dt=0.1, default_dt=0.1,
run_interventions_scenario(model, T=100, max_dt=0.1, default_dt=0.1,
                                # Intervention timing params:
                                cadence_dt=0.5, 
                                cadence_cycle_length=28,
                                init_cadence_offset='random',
                                cadence_presets='default',
                                intervention_start_time=0,
                                intervention_start_prevalence=0,
                                prevalence_flags=['infected'],
                                onset_flags=['symptomatic'], 
                                case_introduction_rate=1/7,
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
                                test_type_proactive='molecular',
                                test_type_onset='molecular',
                                test_type_traced='molecular', 
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

# Plot results
# fig, ax = plt.subplots()
# ax.fill_between(
#     model.tseries,
#     model.counts["E"] + model.counts["P"] + model.counts["I"] + model.counts["A"],
#     np.zeros_like(model.tseries),
#     label="A",
#     color="pink",
# )
# ax.fill_between(
#     model.tseries,
#     model.counts["E"] + model.counts["P"] + model.counts["I"],
#     np.zeros_like(model.tseries),
#     label="I",
#     color="crimson",
# )
# ax.fill_between(
#     model.tseries,
#     model.counts["E"] + model.counts["P"],
#     np.zeros_like(model.tseries),
#     label="P",
#     color="orange",
# )
# ax.fill_between(
#     model.tseries,
#     model.counts["E"],
#     np.zeros_like(model.tseries),
#     label="E",
#     color="gold",
# )
# ax.legend()
# plt.show()
