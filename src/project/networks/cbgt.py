from nengo import Connection, Node, Network
from nengo.config import Config
from nengo.networks import BasalGanglia, Thalamus

import numpy as np

from dataclasses import dataclass, field, fields

from project.networks.cortex import CortexParams, Cortex

#======================================================================================#
# ------------------------------ Parameter Dataclasses ------------------------------- #
#======================================================================================#

@dataclass(frozen = True)
class BGParams:
    """
    Dataclass containing the parameters for Nengo's implementation of the basal ganglia. The parameters are the same  as in the Nengo documentation [1]. 
    The default values are modified to better fit the model used in this project. The dimension is fixed at 2.

    Parameters
    ----------
    n_neurons_per_ensemble: int, optional
        Number of neurons in each ensemble in the network. Default is 300
    output_weight: float, optional
        Scaling factor from the output of the basal ganglia, i.e. from the GPi to the thalamus. A defualt value of -10 has been chosen to ensure that
        the thalamus is completely inhibited when no input is presented
    input_bias: float, optional
        A bias added to the input of the basal ganglia, affecting both inputs equally. An input bias is important to ensure that the basal ganglia
        stays in the correct operating regime. A default value of 0.4 has been chosen, to ensure a proper operating regime without saturating the system.
    ampa_config: Config, optional
        Configuration for AMPA synapses in the basal ganglia. If None, a lowpass synapse with a time constant of 2 ms will be used.
    gaba_config: Config, optional
        Configuration for GABA synapses in the basal ganglia. If None, a lowpass synapse with a time constant of 8 ms will be used.

    References
    ------------
    .. [1] Applied Brain Research. Nengo API Reference.
    https://www.nengo.ai/nengo/v4.0.0/networks.html#nengo.networks.BasalGanglia. 
    Accessed 20 May 2026
    """
    n_neurons_per_ensemble: int = 300
    output_weight: float = -10
    input_bias: float = 0.4
    ampa_config: Config = None
    gaba_config: Config = None


@dataclass(frozen = True)
class ThalamusParams:
    """
    Dataclass containing the parameters for Nengo's implementation of the thalamus. The parameters are the same as in
    the Nengo documentation [1]. The default values are modified to better fit the simulated model. The dimension is fixed at 2.

    Parameters
    -----------

    n_neurons_per_ensemble : int, optional
        Number of neurons in each ensemble in the network.
    mutual_inhib : float, optional
        Inhibitory strength between the two thalamic channels. The default is 1.
    threshold : float, optional
        Threshold below which values will not be represented. A default of 0.3 is used to ensure that the thalamus is not activated by noise,
        and that the threshold is not too close to the default decision threshold of the simulation, which is at 0.7.

    References
    ------------
    .. [1] Applied Brain Research. Nengo API Reference.
    https://www.nengo.ai/nengo/v4.0.0/networks.html#nengo.networks.Thalamus. 
    Accessed 20 May 2026
    """
    n_neurons_per_ensemble: int = 150
    mutual_inhib: float = 1.0
    threshold: float = 0.3

@dataclass(frozen = True)
class ModelParams:
    """
    Dataclass containing the parameters for the entire CBGT model. As input, ModelParams takes dataclasses containing the parameters for each network, as
    well as general parameters for the entire model. See the descriptions of each dataclass for more details. This structure is convenient for when
    functions to run trials and experiments are introduced.

    Parameters
    -------------
    cx_params: CortexParams, optional
        - Dataclass containing the parameters for the cortex. If None, default parameters will be used. See CortexParameters for more details.
    bg_params: BGParams, optional
        - Dataclass containing the parameters for the basal ganglia. If None, default parameters will be used. See BGParams for more details.
    th_params: ThalamusParams, optional
        - Dataclass containing the parameters for the thalamus. If None, default parameters will be used. See ThalamusParams for more details.
    stn_bias: float, optional
        - A bias input to the STN. Slows down the decision-making process, leading to more cautious and deliberate decisions. This parameter has
        not been utilized in the project. A default value of 0.0 is used, meaning that the STN receives no bias input.
    tau: float, optional
        - Synaptic time constant for connections between sub-networks. Default is 0.01

    """
    cx_params: CortexParams = field(default_factory=CortexParams)
    bg_params: BGParams = field(default_factory=BGParams)
    th_params: ThalamusParams = field(default_factory=ThalamusParams)
    stn_bias: float = 0.0
    tau: float = 0.01


#======================================================================================#
# ----------------------------------- Network ---------------------------------------- #
#======================================================================================#

class CBGT(Network):
    """
    Cortico-basal ganglia-thalamic model. This network consists of three sub-networks: the cortex, the basal ganglia, and the thalamus.
    The network contains two action channels, representing the left and right actions, respectively.
    The cortex responds selectively to input stimuli, dots moving either left or right. The cortex creates competition between the two stimulus directions,
    biasing action selection and enhancing coherence. The cortex sends the input into the respective action channels in the basal ganglia.
    The basal ganglia implements a winner-take-all network, and is responsible for action selection. The basal ganglia sends the output to the thalamus 
    through the GPi, which inhibits the thalamus.
    The thalamus is responsible for initiating the selected action. An action is initiated when the GPi releases its inhibitory influence on the thalamus.
    The thalamus will represent the value 0 for unselected actions and approximately 1 for selected actions. The two actions inhibit each other, such that
    only one action is selected at a time.

    Parameters
    ---------------
    params: ModelParams, optional
        - Dataclass containing the parameters for the entire model. See ModelParams for more details. If None, default parameters will be used.
    model_seed: int, optional
        - Seed for the entire model to ensure reproducibility and consistency across trials and experiments. Default is None, meaning that random seeds
        will be generated. A seed of 1 is used for most simulations
    **kwargs: keyword arguments
        - Additional keyword arguments passed to the Network constructor
    

    Attributes
    ---------------
    cortex: Network
        - The cortical network, consisting of the L, R, and I populations. L and R represent the left and right action channels, respectively.
        I represents the inhibitory population
    bg: Network
        - The basal ganglia network, containing two channels responding to left an right stimulus, respectively. The basal ganglia network consists of two
        dimensions, 0 and 1, representing the left and right action channels, respectively. Each action channel receives input from the corresponding 
        channel in the cortex, and sends output to the thalamus.
    thalamus: Network
        - The thalamus network, containing two channels responding to left an right stimulus, respectively. As with the basal ganglia network, the thalamus
        consists of two dimensions, 0 for left and 1 for right. The thalamus receives input from the GPi of the basal ganglia.
    input: Node 
        - Accepting input externally. The input is a 2D vector, where the first and second dimension represents the left and right channels, respectively.
        Each dimension of the dots stimulus must be fed into separate nodes, which should then be connected to the corresponding cortical channel.
        I.e. dots_node1 -> input[0] and dots_node2 -> input[1]. This is because a Node cannot accept an array of Processes [1] as input.
    output: Node
        - Output node for the network, containing the decision. The output for the network is the same as the output of the thalamus. which in this case
        is a 2D vector, whose first and seconds dimensions corresponds to the left and right decision, respectively. The output will be approximately 0
        for unselected actions and approximately 1 for selected actions.

    References
    ------------
    .. [1] Applied Brain Research. Nengo API Reference.
    https://www.nengo.ai/nengo/v4.0.0/frontend-api.html#module-nengo.processes. 
    Accessed 20 May 2026
    """

    def __init__(
            self,
            params: ModelParams | None = None,
            model_seed: int = None,
            **kwargs,
        ):
        # Set Parameters
        params = params if params is not None else ModelParams()
        cx_params = params.cx_params
        bg_params = params.bg_params
        th_params = params.th_params
        tau = params.tau
        stn_bias = params.stn_bias

        # Convert parameters to dictionaries for easier unpacking when creating sub-networks
        bg_dict = {f.name: getattr(bg_params, f.name) for f in fields(bg_params)}
        th_dict = {f.name: getattr(th_params, f.name) for f in fields(th_params)}


        kwargs.setdefault('label', 'CBGT')
        super().__init__(seed = model_seed, **kwargs)

        with self:
            #====== Create sub-networks ======#
            self.cortex = Cortex(params=cx_params)
            self.bg = BasalGanglia(dimensions=2, **bg_dict)
            self.thalamus = Thalamus(dimensions=2, **th_dict)
            self.output = self.thalamus.output

            #====== Create Input ======#
            # Define inputs
            self.input = Node(size_in=2, label='Input')
            stn_bias_input = Node(np.ones(2)*stn_bias, label='STN Bias')

            #====== Connect sub-networks ======#
            # Connect cortex to basal ganglia
            Connection(self.cortex.output, self.bg.input, synapse=tau)
            # Connect basal ganglia to thalamus
            Connection(self.bg.output, self.thalamus.input, synapse=tau)
            # Connect input
            Connection(self.input, self.cortex.input, synapse=None)
            # Connect STN bias input
            Connection(stn_bias_input, self.bg.stn.input, synapse=None)
