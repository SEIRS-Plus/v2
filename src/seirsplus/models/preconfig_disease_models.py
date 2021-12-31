'''
Pre-configured disease models
'''
# External Libraries
import numpy as np
# Internal Libraries
from seirsplus import utils
from seirsplus.models.compartment_network_model import CompartmentNetworkModel


####################################################
####################################################


class SARSCoV2NetworkModel(CompartmentNetworkModel):
    def __init__(self,
                    networks,
                    R0_mean='default',
                    R0_cv='default',
                    transmissibility='default',  # overrides R0 params if given
                    susceptibility='default',
                    latent_period='default',
                    presymptomatic_period='default',
                    symptomatic_period='default',
                    pct_asymptomatic='default',
                    isolation_period='default',
                    # Other parent class args:
                    mixedness=0.0,
                    openness=0.0,
                    transition_mode='time_in_state',
                    local_trans_denom_mode='all_contacts',
                    track_case_info=False,
                    store_Xseries=False,
                    node_groups=None,
                    seed=None
                ):

        # Set the compartments specification for this model:
        compartments = utils.load_config('compartments_SARSCoV2.json') 

        # Instantiate the base model, passing parent class args:
        super().__init__(
                            compartments=compartments,
                            networks=networks,
                            mixedness=mixedness,
                            openness=openness,
                            transition_mode=transition_mode,
                            local_trans_denom_mode=local_trans_denom_mode,
                            track_case_info=track_case_info,
                            store_Xseries=store_Xseries,
                            node_groups=node_groups,
                            seed=seed
                        )

        self.disease_stats = {}

        # Initialize disease-specific parameter distributions:

        # Disease progression parameter distributions:
        latent_period           = (utils.gamma_dist(mean=3.0, coeffvar=0.6, N=self.pop_size) 
                                    if latent_period == 'default' 
                                    else utils.param_as_array(latent_period, (1,self.pop_size)))

        presymptomatic_period   = (utils.gamma_dist(mean=2.2, coeffvar=0.5, N=self.pop_size) 
                                    if presymptomatic_period == 'default' 
                                    else utils.param_as_array(presymptomatic_period, (1,self.pop_size)))

        symptomatic_period      = (utils.gamma_dist(mean=4.0, coeffvar=0.4, N=self.pop_size) 
                                    if symptomatic_period == 'default' 
                                    else utils.param_as_array(symptomatic_period, (1,self.pop_size)))

        incubation_period = latent_period + presymptomatic_period
        infectious_period = presymptomatic_period + symptomatic_period

        self.disease_stats.update(utils.dist_stats([latent_period, presymptomatic_period, symptomatic_period, incubation_period, infectious_period], 
                                                    ["latent_period", "presymptomatic_period", "symptomatic_period", "incubation_period", "infectious_period"]))

        if(self.transition_mode == 'time_in_state'):
            self.set_transition_time('E', to='P', time=latent_period)
            self.set_transition_time('P', to=['I', 'A'], time=presymptomatic_period)
            self.set_transition_time(['I', 'A'], to='R', time=symptomatic_period)
        elif(self.transition_mode == 'exponential_rates'):
            self.set_transition_rate('E', to='P', rate=1/latent_period)
            self.set_transition_rate('P', to=['I', 'A'], rate=1/presymptomatic_period)
            self.set_transition_rate(['I', 'A'], to='R', rate=1/symptomatic_period)

        pct_asymptomatic = 0.3 if pct_asymptomatic == 'default' else pct_asymptomatic
        self.set_transition_probability('P', {'I': 1 - pct_asymptomatic, 'A': pct_asymptomatic})

        # Susceptibility parameter distributions:
        if susceptibility != 'default':
            susceptibility = utils.param_as_array(susceptibility, (1,self.pop_size))
        else:
            susceptibility = np.ones(shape=(1, self.pop_size))
        self.set_susceptibility('S', to=['P', 'I', 'A'], susceptibility=susceptibility)

        # Transmissibility parameter distributions:
        if transmissibility != 'default':
            transmissibility = utils.param_as_array(transmissibility, (1,self.pop_size))
        else:
            R0_mean = 3.0 if R0_mean == 'default' else R0_mean
            R0_cv = 2.0 if R0_cv == 'default' else R0_cv
            R0 = utils.gamma_dist(mean=R0_mean, coeffvar=R0_cv, N=self.pop_size)
            transmissibility = 1/infectious_period * R0
        self.set_transmissibility(['P', 'I', 'A'], ['local'], transmissibility=transmissibility)

        self.mixedness = 0.2
        self.openness = 0.0

        self.isolation_period = 10 if isolation_period == 'default' else isolation_period


####################################################
####################################################


class SARSCoV2NetworkModel_reinfection(CompartmentNetworkModel):
    def __init__(self,
                    networks,
                    R0_mean='default',
                    R0_cv='default',
                    transmissibility='default',  # overrides R0 params if given
                    susceptibility='default',
                    relative_transmissibility_reinfection='default',  
                    relative_susceptibility_reinfection='default',
                    latent_period='default',
                    presymptomatic_period='default',
                    symptomatic_period='default',
                    pct_asymptomatic='default',
                    isolation_period='default',
                    # Other parent class args:
                    mixedness=0.0,
                    openness=0.0,
                    transition_mode='time_in_state',
                    local_trans_denom_mode='all_contacts',
                    track_case_info=False,
                    store_Xseries=False,
                    node_groups=None,
                    seed=None
                ):

        # Set the compartments specification for this model:
        compartments = utils.load_config('compartments_SARSCoV2_reinfection.json') 

        # Instantiate the base model, passing parent class args:
        super().__init__(
                            compartments=compartments,
                            networks=networks,
                            mixedness=mixedness,
                            openness=openness,
                            transition_mode=transition_mode,
                            local_trans_denom_mode=local_trans_denom_mode,
                            track_case_info=track_case_info,
                            store_Xseries=store_Xseries,
                            node_groups=node_groups,
                            seed=seed
                        )

        self.disease_stats = {}

        # Initialize disease-specific parameter distributions:

        # Disease progression parameter distributions:
        latent_period           = (utils.gamma_dist(mean=3.0, coeffvar=0.6, N=self.pop_size) 
                                    if latent_period == 'default' 
                                    else utils.param_as_array(latent_period, (1,self.pop_size)))

        presymptomatic_period   = (utils.gamma_dist(mean=2.2, coeffvar=0.5, N=self.pop_size) 
                                    if presymptomatic_period == 'default' 
                                    else utils.param_as_array(presymptomatic_period, (1,self.pop_size)))

        symptomatic_period      = (utils.gamma_dist(mean=4.0, coeffvar=0.4, N=self.pop_size) 
                                    if symptomatic_period == 'default' 
                                    else utils.param_as_array(symptomatic_period, (1,self.pop_size)))

        incubation_period = latent_period + presymptomatic_period
        infectious_period = presymptomatic_period + symptomatic_period

        self.disease_stats.update(utils.dist_stats([latent_period, presymptomatic_period, symptomatic_period, incubation_period, infectious_period], 
                                                    ["latent_period", "presymptomatic_period", "symptomatic_period", "incubation_period", "infectious_period"]))

        if(self.transition_mode == 'time_in_state'):
            self.set_transition_time('E', to='P', time=latent_period)
            self.set_transition_time('P', to=['I', 'A'], time=presymptomatic_period)
            self.set_transition_time(['I', 'A'], to='R', time=symptomatic_period)
            self.set_transition_time('Er', to='Pr', time=latent_period)
            self.set_transition_time('Pr', to=['Ir', 'Ar'], time=presymptomatic_period)
            self.set_transition_time(['Ir', 'Ar'], to='Rr', time=symptomatic_period)
        elif(self.transition_mode == 'exponential_rates'):
            self.set_transition_rate('E', to='P', rate=1/latent_period)
            self.set_transition_rate('P', to=['I', 'A'], rate=1/presymptomatic_period)
            self.set_transition_rate(['I', 'A'], to='R', rate=1/symptomatic_period)
            self.set_transition_rate('Er', to='Pr', rate=1/latent_period)
            self.set_transition_rate('Pr', to=['Ir', 'Ar'], rate=1/presymptomatic_period)
            self.set_transition_rate(['Ir', 'Ar'], to='Rr', rate=1/symptomatic_period)

        pct_asymptomatic = 0.3 if pct_asymptomatic == 'default' else pct_asymptomatic
        self.set_transition_probability('P', {'I': 1 - pct_asymptomatic, 'A': pct_asymptomatic})
        self.set_transition_probability('Pr', {'Ir': 1 - pct_asymptomatic, 'Ar': pct_asymptomatic})

        # Susceptibility parameter distributions:
        if susceptibility != 'default':
            susceptibility = utils.param_as_array(susceptibility, (1,self.pop_size))
        else:
            susceptibility = np.ones(shape=(1, self.pop_size))
        self.set_susceptibility('S', to=['P', 'I', 'A'], susceptibility=susceptibility)

        # Transmissibility parameter distributions:
        if transmissibility != 'default':
            transmissibility = utils.param_as_array(transmissibility, (1,self.pop_size))
        else:
            R0_mean = 3.0 if R0_mean == 'default' else R0_mean
            R0_cv = 2.0 if R0_cv == 'default' else R0_cv
            R0 = utils.gamma_dist(mean=R0_mean, coeffvar=R0_cv, N=self.pop_size)
            transmissibility = 1/infectious_period * R0
        self.set_transmissibility(['P', 'I', 'A', 'Pr', 'Ir', 'Ar'], ['local'], transmissibility=transmissibility)

        # Susceptibility to reinfection parameter distributions:
        if relative_susceptibility_reinfection != 'default':
            relative_susceptibility_reinfection = utils.param_as_array(relative_susceptibility_reinfection, (1,self.pop_size))
        else:
            relative_susceptibility_reinfection = np.zeros(shape=(1, self.pop_size))
        self.set_susceptibility('R',  to=['P', 'I', 'A', 'Pr', 'Ir', 'Ar'], susceptibility=np.multiply(susceptibility, relative_susceptibility_reinfection))
        self.set_susceptibility('Rr', to=['P', 'I', 'A', 'Pr', 'Ir', 'Ar'], susceptibility=np.multiply(susceptibility, relative_susceptibility_reinfection))

        # Transmissibility conditional on reinefction parameter distributions:
        if relative_transmissibility_reinfection != 'default':
            relative_transmissibility_reinfection = utils.param_as_array(relative_transmissibility_reinfection, (1,self.pop_size))
        else:
            relative_susceptibility_reinfection = np.ones(shape=(1, self.pop_size))
        self.set_transmissibility(['Pr', 'Ir', 'Ar'], ['local'], transmissibility=np.multiply(transmissibility, relative_transmissibility_reinfection))

        self.mixedness = 0.2
        self.openness = 0.0

        self.isolation_period = 10 if isolation_period == 'default' else isolation_period


