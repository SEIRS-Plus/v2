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
                    R0='default',                                       # overrides R0 mean and CV params if given
                    R0_mean='default',
                    R0_cv='default',
                    transmissibility='default',                         # overrides R0 params if given
                    transmissibility_presymptomatic='default',          # overrides relative_transmissibility_presymptomatic if given
                    transmissibility_asymptomatic='default',            # overrides relative_transmissibility_asymptomatic if given
                    relative_transmissibility_presymptomatic='default', 
                    relative_transmissibility_asymptomatic='default',   
                    susceptibility='default',
                    susceptibility_priorexposure='default',             # overrides relative_susceptibility_priorexposure if given
                    susceptibility_reinfection='default',               # overrides relative_susceptibility_reinfection if given
                    relative_susceptibility_priorexposure='default',    
                    relative_susceptibility_reinfection='default',      
                    latent_period='default',
                    presymptomatic_period='default',
                    symptomatic_period='default',
                    pct_asymptomatic='default',
                    # Other parent class args:
                    mixedness=0.2,
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
                                    if isinstance(latent_period, str) and latent_period == 'default' 
                                    else utils.param_as_array(latent_period, (1,self.pop_size)))
        presymptomatic_period   = (utils.gamma_dist(mean=2.0, coeffvar=0.5, N=self.pop_size) 
                                    if isinstance(presymptomatic_period, str) and presymptomatic_period == 'default' 
                                    else utils.param_as_array(presymptomatic_period, (1,self.pop_size)))
        symptomatic_period      = (utils.gamma_dist(mean=4.0, coeffvar=0.4, N=self.pop_size) 
                                    if isinstance(symptomatic_period, str) and symptomatic_period == 'default' 
                                    else utils.param_as_array(symptomatic_period, (1,self.pop_size)))

        incubation_period = latent_period + presymptomatic_period
        infectious_period = presymptomatic_period + symptomatic_period

        self.disease_stats.update(utils.dist_stats([latent_period, presymptomatic_period, symptomatic_period, incubation_period, infectious_period], 
                                                    ["latent_period", "presymptomatic_period", "symptomatic_period", "incubation_period", "infectious_period"]))

        self.set_node_attribute(node='all', attribute_name='latent_period', attribute_value=latent_period)
        self.set_node_attribute(node='all', attribute_name='presymptomatic_period', attribute_value=presymptomatic_period)
        self.set_node_attribute(node='all', attribute_name='symptomatic_period', attribute_value=symptomatic_period)
        self.set_node_attribute(node='all', attribute_name='incubation_period', attribute_value=incubation_period)
        self.set_node_attribute(node='all', attribute_name='infectious_period', attribute_value=infectious_period)

        if(isinstance(self.transition_mode, str) and self.transition_mode == 'time_in_state'):
            self.set_transition_time('E', to='P', time=latent_period)
            self.set_transition_time('P', to=['I', 'A'], time=presymptomatic_period)
            self.set_transition_time(['I', 'A'], to='R', time=symptomatic_period)
        elif(isinstance(self.transition_mode, str) and self.transition_mode == 'exponential_rates'):
            self.set_transition_rate('E', to='P', rate=1/latent_period)
            self.set_transition_rate('P', to=['I', 'A'], rate=1/presymptomatic_period)
            self.set_transition_rate(['I', 'A'], to='R', rate=1/symptomatic_period)

        if(isinstance(pct_asymptomatic, str) and pct_asymptomatic != 'default'):
            pct_asymptomatic = utils.param_as_array(pct_asymptomatic, (1,self.pop_size))
        else:
            pct_asymptomatic = utils.param_as_array(0.3, (1,self.pop_size))
        self.set_transition_probability('P', {'I': 1 - pct_asymptomatic, 'A': pct_asymptomatic})

        # Susceptibility parameter distributions:
        if(isinstance(susceptibility, str) and susceptibility != 'default'):
            susceptibility = utils.param_as_array(susceptibility, (1,self.pop_size))
        else:
            susceptibility = np.ones(shape=(1, self.pop_size))
        self.set_susceptibility('S', to=['P', 'I', 'A'], susceptibility=susceptibility)

        # Susceptibility parameter distributions:
        # Susceptibility of the immunologically naive:
        if(isinstance(susceptibility, str) and susceptibility != 'default'):
            susceptibility = utils.param_as_array(susceptibility, (1,self.pop_size))
        else:
            susceptibility = np.ones(shape=(1, self.pop_size))
        self.set_susceptibility('S',  to=['P', 'I', 'A'], susceptibility=susceptibility)
        
        # Susceptibility of those with prior exposure:
        if(isinstance(susceptibility_priorexposure , str) and susceptibility_priorexposure  == 'default'):
            if(isinstance(relative_susceptibility_priorexposure , str) and relative_susceptibility_priorexposure  == 'default'):
                susceptibility_priorexposure = 0.0 * susceptibility
            else:
                relative_susceptibility_priorexposure = utils.param_as_array(relative_susceptibility_priorexposure, (1,self.pop_size))
                susceptibility_priorexposure = relative_susceptibility_priorexposure * susceptibility
        else:
            susceptibility_priorexposure = utils.param_as_array(susceptibility_priorexposure, (1,self.pop_size))
        self.set_susceptibility('Rp', to=['P', 'I', 'A'], susceptibility=susceptibility_priorexposure)
        
        # Susceptibility of recovereds to reinfection:
        if(isinstance(susceptibility_reinfection , str) and susceptibility_reinfection  == 'default'):
            if(isinstance(relative_susceptibility_reinfection , str) and relative_susceptibility_reinfection  == 'default'):
                susceptibility_reinfection = 0.0 * susceptibility
            else:
                relative_susceptibility_reinfection = utils.param_as_array(relative_susceptibility_reinfection, (1,self.pop_size))
                susceptibility_reinfection = relative_susceptibility_reinfection * susceptibility
        else:
            susceptibility_reinfection = utils.param_as_array(susceptibility_reinfection, (1,self.pop_size))
        self.set_susceptibility('R',  to=['P', 'I', 'A'], susceptibility=susceptibility_reinfection)
        
        # Transmissibility parameter distributions:
        # Baseline infectious transmissibility:
        if(isinstance(transmissibility , str) and transmissibility  != 'default'):
            transmissibility = utils.param_as_array(transmissibility, (1,self.pop_size))
        elif(R0 !='default'):
            R0 = utils.param_as_array(R0, (1,self.pop_size))
            transmissibility = 1/infectious_period * R0
        else:
            R0_mean = 3.0 if isinstance(R0_mean , str) and R0_mean  == 'default' else R0_mean
            R0_cv = 2.0 if isinstance(R0_cv , str) and R0_cv  == 'default' else R0_cv
            R0 = utils.gamma_dist(mean=R0_mean, coeffvar=R0_cv, N=self.pop_size)
            transmissibility = 1/infectious_period * R0
        self.set_transmissibility('I', ['local'], transmissibility=transmissibility)
        # Transmissibility of presymptomatic individuals:
        if(isinstance(transmissibility_presymptomatic, str) and transmissibility_presymptomatic != 'default'):
            transmissibility_presymptomatic = utils.param_as_array(transmissibility_presymptomatic, (1,self.pop_size))
        elif(isinstance(relative_transmissibility_presymptomatic , str) and relative_transmissibility_presymptomatic  != 'default'):
            transmissibility_presymptomatic = relative_transmissibility_presymptomatic * transmissibility
        else:
            transmissibility_presymptomatic = 1.0 * transmissibility
        self.set_transmissibility('P', ['local'], transmissibility=transmissibility_presymptomatic)
        # Transmissibility of asymptomatic individuals:
        if(isinstance(transmissibility_asymptomatic, str) and transmissibility_asymptomatic != 'default'):
            transmissibility_asymptomatic = utils.param_as_array(transmissibility_asymptomatic, (1,self.pop_size))
        elif(isinstance(relative_transmissibility_asymptomatic , str) and relative_transmissibility_asymptomatic  != 'default'):
            transmissibility_asymptomatic = relative_transmissibility_asymptomatic * transmissibility
        else:
            transmissibility_asymptomatic = 1.0 * transmissibility
        self.set_transmissibility('A', ['local'], transmissibility=transmissibility_asymptomatic)
        

####################################################
####################################################
