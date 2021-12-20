# External libraries
import numpy as np
import matplotlib.pyplot as plt
# seirsplus libraries
from seirsplus.models.preconfig_disease_models import *
from seirsplus.networks import *
from seirsplus.utils import *
from seirsplus.scenarios import *

# ------------------------

# Set population size:
N = 100

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generate contact networks:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
networks, clusters, households, age_groups, node_labels = generate_community_networks(N)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Instantiate the model:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
model = SARSCoV2NetworkModel(networks=networks, mixedness=0.2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Specify other model configurations:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
model.set_network_activity('household', active_isolation=True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set model metadata:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
for i in range(N): 
    model.add_individual_flag(node=i, flag=node_labels[i])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up the initial state:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
model.introduce_random_exposures((1/N)*N)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run the model scenario:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                                # Network params:
                                network_active_cadences={network: 'daily' if network!='household' else 'nightly' for network in networks},
                                # Isolation params:
                                isolation_delay_onset=0,
                                isolation_delay_onset_groupmate=0,
                                isolation_delay_positive=0,
                                isolation_delay_positive_groupmate=0,
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
                                test_type_proactive='antigen',
                                test_type_onset='pcr',
                                test_type_traced='pcr', 
                                test_result_delay={'pcr': 1, 'antigen': 0},
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Update results data with other info:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# timestamp
# connected components

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Save results to file:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
results = model.get_results_dataframe()
results.to_csv('./results.csv')

cases = model.get_case_log_dataframe()
cases.to_csv('./case_logs.csv')




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
