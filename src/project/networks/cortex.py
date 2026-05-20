from nengo import Ensemble, Connection, Node, Network
from nengo import config
from nengo.neurons import LIF, NeuronType
from nengo.dists import Choice, Uniform
from nengo.config import Config
from nengo.networks.ensemblearray import EnsembleArray
import numpy as np

from dataclasses import dataclass

@dataclass(frozen = True)
class CortexParams:
    """
    Dataclass containing the parameters for the Cortex class.
    The default parameters are chosen to ensure the following:
    - The channel populations stay bounded within a reasonable range (< 1) and are not too sensitive to the input.
    - The channel populations compete with each other
    - If the value of one population is sufficiently high, the other population will be completely repressed if its value is sufficiently low.
        i.e. the systems expresses mild winner-take-all dynamics if the difference in activation between the two channels is large enough.

    Paramters
    ---------------
    n_neurons_per_ensemble: int, optional
        - Number of neurons in each ensemble, default is 100
    tau_C: float, optional
        - Time constant for the R and L channels, default is 0.02
    tau_I: float, optional
        - Time constant for the I ensemble, default is 0.005
    phi: float, optional
        - Recurrent connection strength, default is 0.75
    alpha_E: float, optional
        - Strength of excitatory connections between L and R, default is 0.3
    alpha_I: float, optional
        - Strength of excitatory connections from L and R to I, default is 1
    beta: float, optional
        - Strength of inhibitory connections from I to L and R, default is 1
    threshold_C: float, optional
        - Threshold for neurons in L and R, default is 0. Determines the intercepts for the tuning curves of the neurons. The intercepts are
            uniformly distributed between threshold_C and 1.0, meaning that the population will start responding when the input exceeds threshold_C.
    threshold_I: float, optional
        - Threshold for neurons in I, default is 0.
    neuron_model: NeuronType, optional
        - Neuron type for all ensembles, default is LIF.
    output_gain: float, optional
        - Gain for the output connection, default is 1. This parameter has not been used. 
    """

    n_neurons_per_ensemble: int = 300
    tau_C: float = 0.02
    tau_I: float = 0.005
    phi: float = 1
    alpha_E: float = 0.3
    alpha_I: float = 1
    beta: float = 1
    threshold_C: float = 0.0
    threshold_I: float = 0.0
    neuron_model: NeuronType = LIF()
    output_gain: float = 1.0



class Cortex(Network):
    """
    The overall structure of the cortex is adapted Wang [1], who proposed a recurrent network model of decision making in cortical circuits. Whereas
    the model by Wang is based on biophysical parameters, the model used here is designed using the Neural Engineering Framewrok (NEF). The cortical
    model presented is designed as an LTI control system.
    The cortex consists of 3 ensembles. Two ensembles, L and R, responding to left and right stimulus, respectively. Both are connected to a shared 
    inhibitory ensemble I, leading to lateral inhibition between the two action channels. The L and R populations sends excitatory connections to I with
    strength alpha_I, whereas the inhibitory population sends inhibitory connections to L and R with strength beta.
    L and R are recurrently connected with strength phi. Furthermore, the two ensembles are connected with excitatory connections to
    one another with strength alpha_E. L and R send stimulus to the striatal populations for their respective action channels.
    For simplicity, the synapses are all standard lowpass filters, essentially making the L and R ensembles leaky integrators with lateral inhibition. 

    The dynamics of the network are described by the following equations:

    tau_C * dx_R/dt = (phi - 1)*x_R + u_R + alpha_E*x_L - beta*x_I
    tau_C * dx_L/dt = (phi - 1)*x_L + u_L + alpha_E*x_R - beta*x_I
    tau_I * dx_I/dt = -x_I + (x_L + x_R)*alpha_I

    
    Parameters
    ---------------
    params: CortexParams, optional
        - Dataclass containing the parameters for the Cortex class. See CortexParams for more details. If not provided, the default parameters
        described in CortexParams will be used
    **kwargs: keyword arguments
        - Additional keyword arguments passed to the Network constructor
    

    Attributes
    ---------------
    L: Ensemble
        - Ensemble representing the left action channel
    R: Ensemble
        - Ensemble representing the right action channel
    I: Ensemble
        - Ensemble representing the inhibitory population
    input: Node
        - Accepting input externally. The input is a 2D vector, where the first and second dimension represents the left and right channels, respectively.
        Each dimension of the dots stimulus must be fed into seperate nodes, which should then be connected to the corresponding cortical channel.
        I.e. dots_node1 -> input[0] and dots_node2 -> input[1]. This is because a Node cannot accept an array of Processes as input.
    output: Node
        - Output signal

            
    References
    ..  [1] Xiao-Jing Wang. “Probabilistic decision making by slow reverberation in cortical circuits”. 
        In: Neuron 36 (2002), pp. 955–968. doi: 10.1016/S0896-6273(02)01092-9.

    """

    def __init__(
            self, 
            params: CortexParams | None = None, 
            **kwargs):
        
        self.params = params if params is not None else CortexParams()

        n_neurons_per_ensemble = self.params.n_neurons_per_ensemble
        tau_C = self.params.tau_C
        tau_I = self.params.tau_I
        phi = self.params.phi
        alpha_E = self.params.alpha_E
        alpha_I = self.params.alpha_I
        beta = self.params.beta
        threshold_C = self.params.threshold_C
        threshold_I = self.params.threshold_I
        neuron_model = self.params.neuron_model
        output_gain = self.params.output_gain


        kwargs.setdefault('label', 'Cortex')
        super().__init__(**kwargs)



        # Configurations that affect all ensembles unless overwritten
        config = Config(Ensemble)
        config[Ensemble].encoders = Choice([[1]]) # Positive encoders, ensuring that firing rate will only increase with increasing input
        config[Ensemble].radius = 1.5 # Maximum value that the ensemble can represent. Input will lie between 0 and 1. Radius is slightly
                                    # larger to accommodate for noise and recurrence.
        config[Ensemble].neuron_type = neuron_model

        # Parameters for the EnsembleArray, which is used to create the L and R ensembles.
        # EnsembleArray is used to create multiple ensembles with the same parameters. They are
        # particularly convenient for representing different channels. The ensemblearray is essentially a way to create a 2D ensemble,
        # split up into two 1D ensembles. The first dimension will represent the right channel, whereas the second dimension will represent the left channel.
        ea_params_C = {'n_ensembles': 2, 'n_neurons': n_neurons_per_ensemble, 'intercepts': Uniform(threshold_C, 1.0), 'label': 'LR'} 

        # Parameters of inhibitory ensemble
        ea_params_I = {'n_neurons': n_neurons_per_ensemble, 'dimensions': 1, 'intercepts': Uniform(threshold_I, 1.0), 'label': 'I'}

        with self, config:
            # ===== Create ensembles ===== #
            LR = EnsembleArray(
                **ea_params_C)
            self.I = Ensemble(
                **ea_params_I)
            
            # Split the ensemble array into two separate ensembles for the left and right channels for easier reference
            self.L = LR.ensembles[0]
            self.R = LR.ensembles[1]
            self.L.label = 'L'
            self.R.label = 'R'


            # ===== Create Input and Output ===== #
            self.input = Node(
                size_in=2, 
                label='Input',)
            self.output = Node(
                size_in=2,
                label='Output')

            # ===== Create connections ===== #
            # Input connection
            Connection(
                self.input, 
                LR.input, 
                synapse=tau_C)

            # Recurrent connections
            Connection(
                LR.output,
                LR.input, 
                transform= phi*np.eye(2),
                synapse=tau_C)
            
            # Excitatory connections between L and R
            Connection(
                LR.output, 
                LR.input, 
                transform= alpha_E*np.fliplr(np.eye(2)), 
                synapse=tau_C)
            
            # Output Connection
            Connection(
                LR.output, 
                self.output, 
                transform=output_gain*np.eye(2),
                synapse=None)
            
            # - Lateral inhibition - 
            # Connections from L and R to I
            Connection(
                LR.output, 
                self.I, 
                transform=alpha_I*np.ones((1,2)), 
                synapse=tau_I)
            # Connections from I to L and R
            Connection(
                self.I, 
                LR.input, 
                transform= -beta*np.ones((2,1)), 
                synapse=tau_C)
            