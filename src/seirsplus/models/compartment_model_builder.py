"""
Programmatically building up compartment models
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Libraries
import json

# External Libraries
import numpy as np


class CompartmentModelBuilder:
    """
    The CompartmentModelBuilder class gives helper functions for defining
    a new compartment model from scratch within a python script. Initializes
    an empty dictionary, compartments, that new compartments can be added to.
    """

    def __init__(self):
        self.compartments = {}

    def add_compartment(
        self,
        name,
        transitions=None,
        transmissibilities=None,
        susceptibilities=None,
        initial_prevalence=0.0,
        exogenous_prevalence=0.0,
        flags=None,
        default_state=None,
        exclude_from_eff_pop=False,
    ):

        """
        Function to build an individual compartment for a compartment model

        Args:
            name (string): name of the compartment
            transitions:
            susceptibilities:
            initial_prevalence (float):
            exogenous_prevalence (float):
            flags:
            default_state:
            exclude_from_eff_pop:

        """
        self.compartments[name] = {
            "transitions": transitions if transitions is not None else {},
            "transmissibilities": transmissibilities
            if transmissibilities is not None
            else {},
            "susceptibilities": susceptibilities
            if susceptibilities is not None
            else {},
            "initial_prevalence": initial_prevalence,
            "exogenous_prevalence": exogenous_prevalence,
            "flags": flags if flags is not None else [],
            "default_state": default_state if default_state is not None else False,
            "exclude_from_eff_pop": exclude_from_eff_pop,
        }
        if default_state is None and not any(
            [self.compartments[c]["default_state"] for c in self.compartments]
        ):
            # There is no default state set so far, make this new compartment the default
            self.compartments[name]["default_state"] = True

    def add_compartments(self, names):
        """Function to compartments to a compartment model using the add_compartment function

        Args:
            names (list): list of compartment names to add to the compartment model
        """
        for name in names:
            self.add_compartment(name)

    def add_transition(
        self, compartment, to, upon_exposure_to=None, rate=None, time=None, prob=None
    ):
        """function to add transition for one compartment and destination state at a time

        Args:
            compartment (string): name of compartment
            to (string):
            upon_exposure_to (list): list of compartments that can cause a transition
            rate:
            time (float): how long it takes for transition to occur
            prob (float): likelihood of the transition occuring
        """
        infectiousStates = (
            [upon_exposure_to]
            if (
                not isinstance(upon_exposure_to, (list, np.ndarray))
                and upon_exposure_to is not None
            )
            else upon_exposure_to
        )
        if upon_exposure_to is None:  # temporal transition
            transn_config = {}
            if time is not None:
                transn_config.update({"time": time, "rate": 1 / time})
            if rate is not None:
                transn_config.update({"rate": rate, "time": 1 / rate})
            if prob is not None:
                transn_config.update({"prob": prob})
            self.compartments[compartment]["transitions"].update({to: transn_config})
        else:  # transmission-induced transition
            for infectiousState in infectiousStates:
                transn_config = {}
                if prob is not None:
                    # transmission-induced transition do not have rates/times
                    transn_config.update({"prob": prob})
                if (
                    infectiousState
                    in self.compartments[compartment]["susceptibilities"]
                ):
                    self.compartments[compartment]["susceptibilities"][infectiousState][
                        "transitions"
                    ].update({to: transn_config})
                else:
                    self.compartments[compartment]["susceptibilities"].update(
                        {infectiousState: {"transitions": {to: transn_config}}}
                    )

    def set_transition_rate(self, compartment, to, rate):
        # Note that it only makes sense to set a rate for temporal transitions.
        compartments = (
            [compartment]
            if not isinstance(compartment, (list, np.ndarray))
            else compartment
        )
        destStates = [to] if not isinstance(to, (list, np.ndarray)) else to
        for compartment in compartments:
            transn_dict = self.compartments[compartment]["transitions"]
            for destState in destStates:
                try:
                    transn_dict[destState]["rate"] = rate
                    transn_dict[destState]["time"] = 1 / rate
                except KeyError:
                    transn_dict[destState] = {"rate": rate}
                    transn_dict[destState] = {"time": 1 / rate}

    def set_transition_time(self, compartment, to, time):
        # Note that it only makes sense to set a time for temporal transitions.
        compartments = (
            [compartment]
            if not isinstance(compartment, (list, np.ndarray))
            else compartment
        )
        destStates = [to] if not isinstance(to, (list, np.ndarray)) else to
        for compartment in compartments:
            transn_dict = self.compartments[compartment]["transitions"]
            for destState in destStates:
                try:
                    transn_dict[destState]["time"] = time
                    transn_dict[destState]["rate"] = 1 / time
                except KeyError:
                    transn_dict[destState] = {"time": time}
                    transn_dict[destState] = {"rate": 1 / time}

    def set_transition_probability(
        self, compartment, probs_dict, upon_exposure_to=None
    ):
        compartments = (
            [compartment]
            if not isinstance(compartment, (list, np.ndarray))
            else compartment
        )
        infectiousStates = (
            [upon_exposure_to]
            if (
                not isinstance(upon_exposure_to, (list, np.ndarray))
                and upon_exposure_to is not None
            )
            else upon_exposure_to
        )
        for compartment in compartments:
            if upon_exposure_to is None:
                transn_dict = self.compartments[compartment]["transitions"]
                for destState in probs_dict:
                    try:
                        transn_dict[destState]["prob"] = probs_dict[destState]
                    except KeyError:
                        transn_dict[destState] = {"prob": probs_dict[destState]}
            else:
                for infectiousState in infectiousStates:
                    transn_dict = self.compartments[compartment]["susceptibilities"][
                        infectiousState
                    ]["transitions"]
                    for destState in probs_dict:
                        try:
                            transn_dict[destState]["prob"] = probs_dict[destState]
                        except KeyError:
                            transn_dict[destState] = {"prob": probs_dict[destState]}

    def set_susceptibility(self, compartment, to, susceptibility=1.0, transitions={}):
        compartments = (
            [compartment]
            if not isinstance(compartment, (list, np.ndarray))
            else compartment
        )
        infectiousStates = [to] if not isinstance(to, (list, np.ndarray)) else to
        for compartment in compartments:
            for infectiousState in infectiousStates:
                self.compartments[compartment]["susceptibilities"].update(
                    {
                        infectiousState: {
                            "susceptibility": susceptibility,
                            "transitions": transitions,
                        }
                    }
                )

    def set_transmissibility(self, compartment, transm_mode, transmissibility=0.0):
        compartments = (
            [compartment]
            if not isinstance(compartment, (list, np.ndarray))
            else compartment
        )
        transmModes = (
            [transm_mode]
            if not isinstance(transm_mode, (list, np.ndarray))
            else transm_mode
        )
        for compartment in compartments:
            transm_dict = self.compartments[compartment]["transmissibilities"]
            for transmMode in transmModes:
                transm_dict = self.compartments[compartment][
                    "transmissibilities"
                ].update({transmMode: transmissibility})

    def set_initial_prevalence(self, compartment, prevalence=0.0):
        compartments = (
            [compartment]
            if not isinstance(compartment, (list, np.ndarray))
            else compartment
        )
        for compartment in compartments:
            self.compartments[compartment]["initial_prevalence"] = prevalence

    def set_exogenous_prevalence(self, compartment, prevalence=0.0):
        compartments = (
            [compartment]
            if not isinstance(compartment, (list, np.ndarray))
            else compartment
        )
        for compartment in compartments:
            self.compartments[compartment]["exogenous_prevalence"] = prevalence

    def set_default_state(self, compartment):
        for c in self.compartments:
            self.compartments[c]["default_state"] = c == compartment

    def set_exclude_from_eff_pop(self, compartment, exclude=True):
        compartments = (
            [compartment]
            if not isinstance(compartment, (list, np.ndarray))
            else compartment
        )
        for compartment in compartments:
            self.compartments[compartment]["exclude_from_eff_pop"] = exclude

    def add_compartment_flag(self, compartment, flag):
        compartments = (
            list(range(self.pop_size))
            if compartment == "all"
            else [compartment]
            if not isinstance(compartment, (list, np.ndarray))
            else compartment
        )
        flags = [flag] if not isinstance(flag, (list, np.ndarray)) else flag
        for compartment in compartments:
            for flag in flags:
                self.compartments[compartment]["flags"].append(flag)

    def remove_compartment_flag(self, compartment, flag):
        compartments = (
            list(range(self.pop_size))
            if compartment == "all"
            else [compartment]
            if not isinstance(compartment, (list, np.ndarray))
            else compartment
        )
        flags = [flag] if not isinstance(flag, (list, np.ndarray)) else flag
        for compartment in compartments:
            for flag in flags:
                self.compartments[compartment]["flags"] = [
                    f for f in self.compartments[compartment]["flags"] if f != flag
                ]  # remove all occurrences of flag

    def save_json(self, filename):
        """
        Function to save a compartment model as a JSON
        """
        with open(filename, "w") as outfile:
            json.dump(self.compartments, outfile, indent=6)
