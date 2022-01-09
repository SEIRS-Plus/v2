import numpy as np
from networkx import from_numpy_array
from numpy.testing import assert_equal, assert_almost_equal

from seirsplus.models.compartment_network_model import CompartmentNetworkModel

# ------------------------

class TestCompartmentNetworkModelInitialization:

    def setup(self):
        # Instantiate a toy FARZ network
        adjacency_array = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
        network = from_numpy_array(adjacency_array)
        networks = {"simple_network": network}

        initial_prevalence = 0.1

        # Instantiate the model:
        self.model = CompartmentNetworkModel(
            compartments="./tests/three_compartment_model.json",
            networks=networks,
            transition_mode="time_in_state",
        )

        self.model.set_initial_prevalence('E', initial_prevalence)


    def test_process_local_transmissibility_2d_array(self):

        two_d_transmissibility_array = np.array([[0.1, 0.2, 0], [0.3, 0.4, 0], [0, 0, 0.5]])
        transm_dict = {"simple_network": two_d_transmissibility_array, "exogenous": 0.0}

        self.model.process_local_transmissibility(transm_dict, "simple_network")
        assert_equal(transm_dict["simple_network"], two_d_transmissibility_array)

        # TODO: We should check that transmissibility is zero for pairs not connected in the network. If we add that check,
        #  test it here.


    def test_process_local_transmissibility_1d_array(self):
        one_d_transmissibility_vector = [0.3, 0.6, 0.9]

        # By infected
        transm_dict = {"simple_network": one_d_transmissibility_vector, "exogenous": 0.0, "pairwise_mode": "infected"}
        self.model.process_local_transmissibility(transm_dict, "simple_network")

        expected = np.array([[0.3, 0.6, 0], [0.3, 0.6, 0], [0, 0, 0.9]])
        assert_equal(transm_dict["simple_network"].todense(), expected)

        # By infectee
        transm_dict = {"simple_network": one_d_transmissibility_vector, "exogenous": 0.0, "pairwise_mode": "infectee"}
        self.model.process_local_transmissibility(transm_dict, "simple_network")

        expected = np.array([[0.3, 0.3, 0], [0.6, 0.6, 0], [0, 0, 0.9]])
        assert_equal(transm_dict["simple_network"].todense(), expected)

        # By min
        transm_dict = {"simple_network": one_d_transmissibility_vector, "exogenous": 0.0, "pairwise_mode": "min"}
        self.model.process_local_transmissibility(transm_dict, "simple_network")

        expected = np.array([[0.3, 0.3, 0], [0.3, 0.6, 0], [0, 0, 0.9]])
        assert_equal(transm_dict["simple_network"].todense(), expected)

        # By max
        transm_dict = {"simple_network": one_d_transmissibility_vector, "exogenous": 0.0, "pairwise_mode": "max"}
        self.model.process_local_transmissibility(transm_dict, "simple_network")

        expected = np.array([[0.3, 0.6, 0], [0.6, 0.6, 0], [0, 0, 0.9]])
        assert_equal(transm_dict["simple_network"].todense(), expected)

        # By mean
        transm_dict = {"simple_network": one_d_transmissibility_vector, "exogenous": 0.0, "pairwise_mode": "mean"}
        self.model.process_local_transmissibility(transm_dict, "simple_network")

        expected = np.array([[0.3, 0.45, 0], [0.45, 0.6, 0], [0, 0, 0.9]])
        assert_almost_equal(transm_dict["simple_network"].todense(), expected, decimal=5)


    def test_process_local_transmissibility_single_number(self):

        transmissibility = 0.5

        for pairwise_mode in ["infected", "infectee", "min", "max", "mean"]:
            transm_dict = {"simple_network": transmissibility, "exogenous": 0.0, "pairwise_mode": pairwise_mode}
            self.model.process_local_transmissibility(transm_dict, "simple_network")

            expected = np.array([[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 0.5]])
            assert_equal(transm_dict["simple_network"].todense(), expected)


    def test_process_transition_times_time_in_state_mode(self):

        self.model.transition_mode = "time_in_state"

        # TODO: Understand how a scalar in the json dictionary becomes an array here
        transitions_dict = {"E": {"prob": 1.0, "time": np.array([4, 4, 4])}}
        self.model.process_transition_times(transitions_dict)

        expected = np.array([4, 4, 4]).reshape(3, 1)
        assert_equal(transitions_dict["E"]["time"], expected)

        transitions_dict = {"E": {"prob": 1.0, "rate": np.array([4, 4, 4])}}
        self.model.process_transition_times(transitions_dict)

        expected = 1 / np.array([4, 4, 4]).reshape(3, 1)
        assert_almost_equal(transitions_dict["E"]["time"], expected, 5)


    def test_process_transition_times_rate_mode(self):

        self.model.transition_mode = "exponential_rates"

        # TODO: Understand how a scalar in the json dictionary becomes an array here
        transitions_dict = {"E": {"prob": 1.0, "time": np.array([4, 4, 4])}}
        self.model.process_transition_times(transitions_dict)

        expected = 1 / np.array([4, 4, 4]).reshape(3, 1)
        assert_equal(transitions_dict["E"]["rate"], expected)

        transitions_dict = {"E": {"prob": 1.0, "rate": np.array([4, 4, 4])}}
        self.model.process_transition_times(transitions_dict)

        expected = np.array([4, 4, 4]).reshape(3, 1)
        assert_almost_equal(transitions_dict["E"]["rate"], expected, 5)


    def test_process_transition_probs(self):

        # TODO: Rewrite the method under test to allow injection of a deterministic random generator.

        # Unrealistic to go straight from S to I, but whatever
        transitions_dict = {
            "E": {"prob": 0.4, "time": np.array([4, 4, 4])},
            "I": {"prob": 0.6, "time": np.array([4, 4, 4])}
        }

        self.model.process_transition_probs(transitions_dict)

        assert_equal(transitions_dict["E"]["prob"], np.array([0.4, 0.4, 0.4]).reshape(3, 1))
        assert_equal(transitions_dict["I"]["prob"], np.array([0.6, 0.6, 0.6]).reshape(3, 1))

        transitions_dict = {
            "E": {"prob": [0.4, 0.4, 0.4], "time": np.array([4, 4, 4])},
            "I": {"prob": [0.6, 0.6, 0.6], "time": np.array([4, 4, 4])}
        }

        self.model.process_transition_probs(transitions_dict)

        assert_equal(transitions_dict["E"]["prob"], np.array([0.4, 0.4, 0.4]).reshape(3, 1))
        assert_equal(transitions_dict["I"]["prob"], np.array([0.6, 0.6, 0.6]).reshape(3, 1))


    def test_process_local_transm_offsets(self):

        two_d_offset_array = np.array([[0.1, 0.2, 0], [0.3, 0.4, 0], [0, 0, 0.5]])
        transm_dict = {"offsets": {"simple_network": two_d_offset_array}, "exogenous": 0.0}

        self.model.process_local_transm_offsets(transm_dict, "simple_network")
        assert_equal(transm_dict["offsets"]["simple_network"], two_d_offset_array)



