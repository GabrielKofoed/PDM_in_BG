from nengo.processes import FilteredNoise
from nengo.dists import Gaussian
from nengo import Lowpass, Probe, Connection, Simulator

import pandas as pd
import numpy as np
from numpy.random import SeedSequence, default_rng
from dataclasses import dataclass, asdict, replace
from itertools import product
from collections import deque
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import t


from project.networks.cbgt import ModelParams, CBGT
from project.utilities.statistics import bootstrap_ci, left_right_accuracy_diff_ci
from nengo import Node




# ============================================= #
# --------------- Cortical Input -------------- # 
# ============================================= #

def dots(direction, strength = 1, coherence = 1, noise_std = 0.125, tau = 0.01, task_seed = None, swap_seeds = False, print_val = True):
    """
    Input stimulus to the cortex. Visualized as dots on a screen, moving either left or right. Difficulty can be adjusted through three parameters:
        - Coherence, representing the fraction of evidence favoring the correct decision. Visualized as the fraction of dots moving in the same
        direction. When coherence is 1, all dots move in the same direction. When coherence is 0, equal amounts of dots move in both directions.
        - Strength, representing the overall strength of the stimulus. Can be visualized as the brightness of the screen of moving dots, or perhaps the
        screen is blurry. When strength is 1, the stimulus is at maximum strength. When strength is 0, there is no stimulus, only noise.
        - Noise, representing the amount of noise in the stimulus. Noise can stem from the apparatus in itself or from the environment. 

    The stimulus is generated as white noise from a Gaussian distribution, with the mean determined by the direction and strength of the stimulus.
    The white noise is then filtered through a low-pass filter. The input is adapted from Neuronal Dynamics python exercises [1].

    The output is a tuple of two Nodes, u_L and u_R, representing the left and right stimulus, respectively. Each must be connected to their
    respective channel in the cortex.

    Parameters
    -----------
    direction: str
        - Direction of the coherently moving dots, either 'left' or 'right'
    strength: float
        - Strength of the stimulus, visualized as how many dots are on the screen. Should be between 0 (no stimulus) and 1 (max stimulus). Default is 1.
        Set to 0 to have no stimulus, only noise. This is useful for testing the behavior of the model in absence of stimulus. In the absence of stimulus,
        the model should preferably not make a decision.
    coherence: float
        - Coherence, i.e. the fraction of dots moving in the same direction. Should be between 0 (no coherence) and 1 (full coherence). Default is 1.
        If set to 0, there is no coherence, only noise, and the model should preferably not make a decision. Setting coherence = 0 is different from
        setting strength = 0. When coherence = 0, there can still be a strong stimulus, but it is indiscernible, as the dots are moving in random directions.
    noise_std: float
        - Standard deviation of the Gaussian noise. Default is 0.125. Cannot be below of equal to 0 due to how the stimulus is generated. 
    tau: float
        - Time constant of the low-pass filter. Default is 0.01, i.e. 10 ms.
    task_seed: int or SeedSequence, optional
        - Seed for white noise generator. Default is None. If the task_seed is an int, use it to make a SeedSequence, such that the left and right stimuli
        get different seeds. If task_seed is a SeedSequence, reuse it to generate to different seeds for the left and right stimulus. If seed is None, 
        generate two random seeds for left and right stimulus.
        The user should only manually input an int. A SeedSequence is given as an input by other functions, such as run_task, run_trials, and run_experiments,
        to ensure reproducibility across trials and experiments. 
    swap_seeds: bool, optional
        - If True, the task seed will be swapped. When running a single task to test for bias, this ensures that the difference in decision is due to the 
        direction of the stimulus, not the specific noise realization. This is not used in normal operation. Default is False.
    print_val: bool, optional
        - Whether to print the mean values and coherence. Default is True.
    
        
    Returns
    -------------
    u_L: Node
        - Node outputting the left stimulus. Must be connected to the L population in the cortex
    u_R: Node
        - Node outputting the right stimulus. Must be connected to the R population in the cortex

    References
    .. [1] Wulfram Gerstner et al. Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition. 2014.
    Perceptual Decision Making Exercise (Python)
    url: https://neuronaldynamics-exercises.readthedocs.io/en/latest/exercises/perceptual-decision-making.html
    Accessed 6 April 2026

    """
    # Setting different seeds for left and right input, to ensure they are not identical.
    # If a SeedSequence is given, reuse that. If not, generate a new SeedSequence.
    ss = task_seed if isinstance(task_seed, SeedSequence) else SeedSequence(task_seed)
    left_ss, right_ss = ss.spawn(2)
    left_seed = int(left_ss.generate_state(1)[0])
    right_seed = int(right_ss.generate_state(1)[0])
    if swap_seeds:
        left_seed, right_seed = right_seed, left_seed

    if direction not in ['left', 'right']:
        raise ValueError("Direction must be either 'left' or 'right'")

    # Mathematically, the direction of the stimulus is defined by the sign of the coherence. If the direction is 'left', coherence becomes negative
    coherence*=-1 if direction == 'left' else 1

    # Means of the Gaussian distributions for the left and right inputs
    mu_L = strength * (0.5 - 0.5 * coherence)
    mu_R = strength * (0.5 + 0.5 * coherence)

    # Making Processes
    P_L = FilteredNoise(synapse = Lowpass(tau), dist = Gaussian(mu_L, noise_std), seed = left_seed, scale = False)
    P_R = FilteredNoise(synapse = Lowpass(tau), dist = Gaussian(mu_R, noise_std), seed = right_seed, scale = False)
    
    # Create Nodes to output the stimulus
    u_L = Node(P_L, size_out=1)
    u_R = Node(P_R, size_out=1)



    if print_val:
        print('Input Parameters\n----------------')
        print('mu_L: ', round(mu_L,3))
        print('mu_R: ', round(mu_R,3))
        print('Coherence:', round(abs(coherence),3))
        print('Left Seed:', left_seed)
        print('Right Seed:', right_seed)
        print('')
        

    return u_L, u_R


# =============================================#
# ---------------- Experiments ----------------# 
# =============================================#

"""
For differentiating and grouping relevant parameters:
    - Model Parameters: Parameters for each instance of a model. This should be constant between trials and parameters
    - Dots Parameters: Parameters for each task, i.e. the input stimulus. Should vary across trials and experiments.
    - Trial Parameters: Parameters for ensuring reproducibility across trials, and parameters for how the simulation is run.

The word 'trial' will refer to a single run of the model. However, the function that runs a single trial is called 'run_task', as to
not confuse it with the function 'run_trials', which runs multiple trials. The data returned by the 'run_task' function is named TaskResults, and the data
returned by 'run_trials' is called TrialResults, for the same reason. The word 'experiment' will refer to a collection of trials, where some 
parameters can be varied across trials. 

Right now, when the 'run_trials' function runs several experiements, it builds the model from scratch each time, albeit using the same seed for consistency.
This keeps implementation simple, but may be computationally inefficient. This works for now, but the process could be optimized in the
future, such that the model is only built once, then reset between trials.
"""


# ---------- Dataclasses for task parameters ---------- #



@dataclass(frozen = True)
class DotsParams:
    """
    Parameters for 'run_task' that serves as inputs to 'dots' function. Default values are the same as the 'dots' function. The following parameter
    descriptions are the same as the 'dots' function. However, print_vals is set to False in the following functions, as it is not desired to print
    the parameters. task_seed is also not included, as it will be input into the run_task function, which will then pass it to the dots function.

    This dataclass behaves slightly differently based on whether it is passed to 'run_task' or 'run_trials'. When passed to 'run_trials', 
    the direction is not used, as the function runs an equal amount of trials for left and right stimulus.

    
    Parameters
    -----------
    direction: str | None
        - Direction of the coherently moving dots, either 'left' or 'right' when passed to 'run_task'. Should only be set to None when passed to 'run_trials',
        as the function runs an equal amount of left and right trials. If specified as either 'left' or 'right' and passed to 'run_trials', the direction
        will be ignored.
    strength: float
        - Strength of the stimulus, visualized as how many dots are on the screen. Should be between 0 (no stimulus) and 1 (max stimulus). Default is 1.
        Set to 0 to have no stimulus, only noise. This is useful for testing the behavior of the model in absence of stimulus. In the absence of stimulus,
        the model should preferably not make a decision.
    coherence: float
        - Coherence, i.e. the fraction of dots moving in the same direction. Should be between 0 (no coherence) and 1 (full coherence). Default is 1.
        If set to 0, there is no coherence, only noise, and the model should preferably not make a decision. Setting coherence = 0 is different from
        setting strength = 0. When coherence = 0, there can still be a strong stimulus, but the dots are moving in random directions.
    noise_std: float
        - Standard deviation of the Gaussian noise. Default is 0.125.
    tau: float
        - Time constant of the low-pass filter. Default is 0.01, i.e. 10 ms.
    """
    direction : str | None
    coherence: float = 1
    strength: float = 1
    noise_std: float = 0.125
    tau: float = 0.01

    
        
@dataclass(frozen = True)
class TrialParams:
    """
    Simulation and trial specific parameters. This dataclass contains parameters that ensure reproducibility across trials, and parameters that
    define how the simulation is run. This dataclass is passed to both 'run_task' and 'run_trials', but behaves slightly differently in each.
    See the parameter descriptions for more details. These parameters are held constant across several trials to ensure consistency.

    Parameters
    ---------------------------
    model_seed: int or None
        - Seed for the CBGT model, ensuring reproducibility and consistency across trials. If None, the behavior of this parameter differs between
        'run_task' and 'run_trials'. If None and passed to 'run_task', a random model seed will be used. If None and passed to 'run_trials', a random
        model seed will still be used. However, when a model seed is specified, the same random seed will be used every time the functions calls 'run_task', 
        ensuring consistency across trials.
    task_seed: int or SeedSequence or None
        - Seed for the input stimulus. If None, use random seed. The user should only specify an int or None. Seedsequence is used internally. 
        This parameter behaves slightly differently whether passed to 'run_task' or 'run_trials'. If passed to 'run_task', the seed will be passed to
        the 'dots' function. If passed to 'run_trials', the seed will be used to create a SeedSequence, which will be used to generate different seeds
        for each trial, ensuring reproducibility across trials.
    decision_threshold: float
        - Decision threshold for the model, i.e. the value of the Thalamus when a decision is made. Default is 0.7. The trial is terminated when
        the Thalamus reaches this threshold, where reaction time and accuracy are recorded. To avoid noise being mistaken for a decision, the value of the
        thalamus is averaged across the last few time steps, defined in the 'decision_window' parameter. The trial is terminated once this mean value
        reaches the threshold
    decision_window: float
        - Time window (s) for averaging the thalamus output to detect a decision. Default is 10 ms.
        Note that even for higher values, this parameter was not found to have a significant effect on the results for larger amounts of noise.
    PD_window: float
        - Post-decision window (s) for determining how stable a decision is. After a decision, the model will run for this amount of time, and the
        standard deviation of the thalamus output for the winning channel will be calculated. Default is 1 second.
    terminate_decision: bool
        - Whether or not to terminate the trial once a decision is made. Default is True. If False, the trial will continue until max_time.
        Useful for probing the neural activity. Can only be False if passed to 'run_task'. If False, the decision will not be recorded.
    t_warmup: float, optional
        - Time (s) for the warmup period, during which the gate function will output 0, effectively blocking the stimulus from reaching the model.
        A warmup period is necessary to ensure the internal dynamics have had time to stabilize due to the bias input to the basal ganglia. 
        Default is 0.15 seconds    
    dt: float
        - Timestep for the simulation. Default is 0.001, i.e. 1 ms. It is generally not necessary to change this parameter, and was only included
        for completeness
    max_time: float
        - Maximum time for the simulation. Default is 1.0 second. If the simulation time reaches this value, the trial is terminated whether or not
        an action has been made. The circuit responds within a few hundred milliseconds, so 1 second is more than enough time for the model to make a decision.
        Any decisions made after a few hundred milliseconds are due to noise, and are thus likely not meaningful.
    """
    model_seed: int | None = None
    task_seed: int | SeedSequence | None = None
    decision_threshold: float = 0.7
    decision_window: float = 10e-3
    PD_window: float = 1
    terminate_decision: bool = True
    t_warmup: float = 0.15
    max_time: float = 1.0
    dt: float = 0.001

# ---------- Dataclasses for results ---------- #

@dataclass(frozen = True)
class TaskResults:
    """
    Dataclass used to contain the results of the 'run_task' function. The results are stored in three dictionaries and one optional dictionary

    Attributes
    -------------
    stim_info: dict
        - Contains the parameters for the stimulus, i.e. the DotsParams.
    decision_info: dict
        - Contains the results of the trial. The following keys are included:
                - 'Decision': An integer value representing the decision made by the model. 0 for left, 1 for right, None for no decision.
                - 'Accurate': A boolean value representing whether the decision was accurate or not. Will be False if no decision was made in the presence
                of a stimulus. Will be true if decision = None and direction = None, as the model correctly did not make a decision in the absence of stimulus.
                - 'RT (ms)': Integer value representing the reaction time in milliseconds, i.e. the time it took to make the decision.
                None for no decision within max_time.
                - 'Outcome': A string value representing the outcome of the trial. 'Correct Decision', 'Wrong Decision', 'No Decision', or 'Premature'.
                This variable is slightly different from 'Accurate', as it differentiates between which kind of decision was made, not just whether the 
                decision was accurate or not. This variable becomes especially useful when differentiating between mistakes and indecisiveness. Another 
                important distinction between "Accurate" and "Correct Decision" is that if direction = None and decision = None, "Accurate" will be true, but
                "Outcome" will be "No Decision". 
                - 'PD Variability': The post-decision variability, calculated as the standard deviation of the thalamus output for the winning channel
                after a decision has been made. This variable is only calculated if a decision is made, and is None for no decision.
                Low values indicate the decision is stable. High values indicate the decision is unstable.
    trial_info: dict
        - Contains trial-specific parameters. The following keys are included:
                - 'model_seed': The seed used for the model.
                - 'task_seed': The seed used for the input stimulus
                - 'decision_threshold': Decision threshold
    probes: dict | None 
        - Optional, included if return_probes is True in the 'run_task' function. Contains the neural data recorded from the probes. Each probe is two-dimensional,
        with the first and second dimensions corresponding to the left and right action channels, respectively. The key names are:
            - 'time': The time data. This is the same for all probes
            - 'input': The values of the input nodes, i.e. the stimulus.
            - 'output': The values of the output nodes
            - 'cortex': The values of the LR ensembles in the cortex
            - 'dSPN': The values of the dSPN ensembles, i.e. the striatal D1 populations
            - 'iSPN': The values of the iSPN ensembles, i.e. the striatal D2 populations
            - 'STN': The values of the STN ensembles
            - 'GPi': The values of the GPi ensembles
            - 'GPe': The values of the GPe ensembles

    """
    stim_info: dict
    decision_info: dict
    trial_info: dict
    probes: dict | None = None

@dataclass(frozen = True)
class TrialResults:
    """
    Dataclass used to contain the results of the 'run_trials' function. The data from each trial is stored in a pandas dataframe, where each row
    corresponds to a single trial. The summary statistics are stored in a dictionary, which are calculated across all trials.

    Attributes
    ---------------------
    trial_data: pd.DataFrame
        - Dataframe containing the data from each trial. Each row corresponds to a single trial. The columns contain all data included in the TaskResults
        dataclass, except for the probes, which are never included. 'master_seed' contains the master seed used to generate the seeds for the input stimulus
        across trials.
    summary_stats: dict
        - Dictionary containing summary statistics calculated across all trials. The following keys are included:
                # - Accuracy statistics -
                # Accuracies
                - 'Accuracy (%)': The average accuracy across trials, i.e. the fraction of accurate trials
                - 'Accuracy Difference (%)': The difference in accuracy between left and right trials, i.e. accuracy for left trials minus accuracy for 
                right trials.
                # Confidence Intervals for accuracies
                - 'Accuracy CI Low (%)': The lower bound of the 95% confidence interval for accuracy
                - 'Accuracy CI High (%)': The upper bound of the 95% confidence interval for accuracy
                - 'Accuracy Diff CI Low (%)': The lower bound of the 95% confidence interval for the accuracy difference between left and right trials
                - 'Accuracy Diff CI High (%)': The upper bound of the 95% confidence interval for the accuracy difference between left and right trials

                # - Reaction time statistics -
                - 'Mean Correct RT (ms)': The average reaction time across trials, calculated for correct trials.
                - 'Mean Correct RT CI Margin (ms)': The margin of the 95% confidence interval for the mean correct reaction time.
                - 'Correct RT std (ms)': The standard deviation of the reaction time across trials, calculated for correct trials.
                - 'Correct RT std CI Low (ms)': The lower bound of the 95% confidence interval for the standard deviation of the correct reaction time
                - 'Correct RT std CI High (ms)': The upper bound of the 95% confidence interval for the standard deviation of the correct reaction time
                - 'Inverse Efficiency': The speed-accuracy tradeoff, calculated as the mean correct RT divided by accuracy.
                
                # - Outcomes -
                - 'Correct Decision (%)': The percentage of trials that resulted in a correct decision.
                - 'Wrong Decision (%)': The percentage of trials that resulted in a wrong decision.
                - 'Premature (%)': The percentage of trials that were terminated due to a premature decision.
                - 'No Decision (%)': The percentage of trials that were terminated due to no decision being made within max_time.
                # Confidence interval for wrong decisions
                - 'Wrong Decision CI Low (%)': The lower bound of the 95% confidence interval for wrong decisions
                - 'Wrong Decision CI High (%)': The upper bound of the 95% confidence interval for wrong decisions

                # Post-decision variability
                - 'Mean PD Variability': The average post-decision variability across trials, calculated for correct and wrong decisions.
                - 'Mean PD Variability CI Margin': The margin of the 95% confidence interval for the mean post-decision variability.
    """
    trial_data: pd.DataFrame
    summary_stats: dict


@dataclass(frozen = True)
class ExperimentResults:
    """
    Dataclass used to contain the results of the 'run_experiments' function. The function will always return a dataframe containing the summary statistics
    across varying parameters. Optionally, the results for each individual trial is returned as well.

    Attributes
    -------------------------
    summary_stats: pd.DataFrame
        - Dataframe containing the summary statistics calculated across trials for each parameter combination. The columns are the same as the 'summary_stats'
        keys in the TrialResults dataclass, where each row corresponds to a different parameter combination.
    trial_data: pd.DataFrame | None
        - Optional dataframe containing the data from each trial. The columns are the same as the 'trial_data' dataframe in TrialResults.
        
    """
    summary_stats: pd.DataFrame
    trial_data: pd.DataFrame | None = None



# -------------- Functions for running tasks, trials, and experiments -------------- #

def add_pathway_bias(source, bg, pathway_bias, synapse=None):
    """
    Helper function for run_task. Adds a direct/indirect pathway correction. This is to simulate which effect learning may have had
    on the system.

    This function essentially copies the input from the cortex to the direct and indirect pathway. Since inputs are superimposed, this is equal to
    changing the relative contribution of the direct and indirect pathway. The default input transformations to the SPN in Nengo are as follows

        dSPN_default = 1.2*x
        iSPN_default = 0.8*x

    With the correction, the SPN transformations become
        dSPN_corrected = (1.2 + pathway_bias)*x
        iSPN_corrected = (0.8 - pathway_bias)*x
    
    """
    if pathway_bias is None or pathway_bias == 0:
        return

    if not -1.2 < pathway_bias < 0.8:
        raise ValueError(
            "Pathway_bias should be between -1.2 and 0.8 to avoid "
            "strongly suppressing or inverting one pathway."
        )

    n = 2
    I = np.eye(n)

    Connection(
        source,
        bg.strD1.input,
        transform=pathway_bias * I,
        synapse=synapse,
    )

    Connection(
        source,
        bg.strD2.input,
        transform=-pathway_bias * I,
        synapse=synapse,
    )

def run_task(
        model_params: ModelParams, 
        dots_params: DotsParams, 
        trial_params: TrialParams | None = None,
        return_probes: bool = False,
        bypass_cortex: bool = False,
        pathway_bias: float | None = None,
        swap_seeds: bool = False,
        ) -> TaskResults:
    """
    Runs a single decision making task. The task terminates when the model makes a decision. The functions logs relevant data, such as reaction time
    and accuracy.
    The basal ganglia requires a bias input to function optimally. This bias is input in the beginning of the simulation. Yet, it takes time for the BG
    to properly integrate this bias input and for the dynamics to stabilize. A warmup period is therefore included in the beginning of the simulation, during
    which the stimulus is blocked. A poor choice of basal ganglia parameters may lead to premature decisions even in the absence of stimulus. Therefore,
    any decision made during the warmup period is not considered a valid decision. The trial will be terminated, and the decision will be marked as
    premature.

    Parameters
    --------------
    model_params: ModelParams
        - Dataclass containing the parameters for the entire model. See ModelParams for more details.
    dots_params: DotsParams
        - Dataclass containing the parameters for the input stimulus. See DotsParams for more details.
    trial_params: TrialParams, optional
        - Dataclass containing the parameters for the trial. See TrialParams for more details. If None, default parameters will be used.
        Use this to set seeds.
    return_probes: bool, optional
        - Whether or not to return neural data for each ensemble. Default is false. 'run_trials' and 'run_experiments' will always set this to False.
    bypass_cortex: bool, optional
        - If True, the input will be connected directly to the basal ganglia, bypassing the cortex. This is useful for testing the behavior of the model without
        the cortex. Default is False.
    pathway_bias: float | None, optional
        - For simulating the effects of learning on the system.This parameter changes the relative contribution of the direct and indirect pathway. See 
        add_pathway_bias function for more details. If None, there is no change in the relative pathway bias. This parameter cannot be used in conjunction with 
        bypass_cortex = True.
    swap_seeds: bool, optional
        - If True, the task seed will be swapped. When running a single task to test for bias, this ensures that the difference in decision is due to the direction of the stimulus, not the specific noise realization. 
        This is never used in normal operation. Default is False.


    Returns
    -------------
    taskresults: TaskResults
        - Dataclass containing the results of the trial. See TaskResults for more details.
                
    """
    if dots_params.direction is None:
        raise ValueError("Direction cannot be None when running a single task. Set direction to 'left' or 'right'. " \
        "If you want indiscernible stimulus, set strength to 0 or coherence to 0.")

    # ---- Set Parameters ----
    dots_params = asdict(dots_params)
    trial_params = trial_params if trial_params is not None else TrialParams()
    tau = model_params.tau # Used when probing the model and adding pathway bias.
    model_seed = trial_params.model_seed
    decision_threshold = trial_params.decision_threshold
    decision_window = trial_params.decision_window
    PD_window = trial_params.PD_window
    terminate_decision = trial_params.terminate_decision
    t_warmup = trial_params.t_warmup
    max_time = trial_params.max_time
    dt = trial_params.dt

    # Create SeedSequence to pass to 'dots'
    task_seed = trial_params.task_seed
    ss = trial_params.task_seed if isinstance(trial_params.task_seed, SeedSequence) else SeedSequence(trial_params.task_seed)

    direction = 0 if dots_params['direction'] == 'left' else 1 # Convert string to int value, for easier analysis.
    direction = None if dots_params['strength'] == 0 or dots_params['coherence'] == 0 else direction # If strength or coherence is 0,
    #there is no stimulus, only noise.

    # ---- Build Model ----
    model = CBGT(model_params, model_seed = model_seed)
    with model:
        # Connecting input
        u_L, u_R = dots(**dots_params, task_seed = ss, swap_seeds = swap_seeds, print_val = False)

        # Define warmup period, where the stimulus is blocked
        def warmup_gate(t, x):
            return 0 if t < t_warmup else x
        
        gate_L = Node(warmup_gate, size_in=1, size_out=1)
        gate_R = Node(warmup_gate, size_in=1, size_out=1)

        Connection(u_L, gate_L, synapse=None)
        Connection(u_R, gate_R, synapse=None)        

        # Connect gate to the model, either to the cortex or basal ganglia. Also adjust pathway gains if pathway_bias is specified.
        if bypass_cortex and pathway_bias is not None:
            raise ValueError("Cannot use pathway_bias when bypassing the cortex, as the pathway bias is implemented in the cortex. " \
            "Set pathway_bias to None to bypass the cortex without changing the pathway bias.")
        elif bypass_cortex:
            Connection(gate_L, model.bg.input[0], synapse = None)
            Connection(gate_R, model.bg.input[1], synapse = None)
        else:
            Connection(gate_L, model.input[0], synapse = None)
            Connection(gate_R, model.input[1], synapse = None)
            add_pathway_bias(
                source=model.cortex.output,
                bg=model.bg,
                pathway_bias=pathway_bias,
                synapse=tau)

        # Probing
        output_probe = Probe(model.output, synapse = tau)
        if return_probes:
            input_probe = Probe(model.input, synapse = None)
            cortex_probe = Probe(model.cortex.output, synapse = tau)
            dSPN_probe = Probe(model.bg.strD1.output, synapse = tau)
            iSPN_probe = Probe(model.bg.strD2.output, synapse = tau)
            STN_probe = Probe(model.bg.stn.output, synapse = tau)
            GPe_probe = Probe(model.bg.gpe.output, synapse = tau)
            GPi_probe = Probe(model.bg.gpi.output, synapse = tau)

    #---- Run Simulation ----
    n_window = int(round(decision_window / dt)) # Number of time steps in the decision window
    buffer = deque(maxlen = n_window) # Buffer to store the last n_window values of the thalamus output.

    with Simulator(model, dt = dt, progress_bar = False) as sim:
        while sim.time < max_time:
            sim.step()
            thal = sim.data[output_probe][-1].copy()
            buffer.append(thal)

            if len(buffer) == n_window and terminate_decision:
                thal_mean = np.mean(buffer, axis = 0) # Average thalamus output across the decision window  
                if np.any(thal_mean >= decision_threshold): # Detect decision
                    decision = int(np.argmax(thal_mean))
                    RT = sim.time - t_warmup if sim.time >= t_warmup else None 
                    
                    # Determine outcome of the trial. If the decision is made during the warmup period, it is considered a premature decision
                    # If the decision is made after the warmup period, it is logged as either a correct or wrong decision.
                    if sim.time < t_warmup:
                        outcome = 'Premature'
                    elif decision == direction: 
                        outcome = 'Correct Decision'
                    else:
                        outcome = 'Wrong Decision'
                    break
        else:
            decision = None
            RT = None
            outcome = 'No Decision'
        
        # Calculate post-decision variability if a decision has been made that is not premature.
        if outcome in ['Correct Decision', 'Wrong Decision'] and PD_window > 0:
            pd_steps = int(round(PD_window / dt))
            pd_start = len(sim.data[output_probe])
            
            sim.run_steps(pd_steps)

            # Calculate post-decision variability.
            pd_trace = sim.data[output_probe][pd_start:, decision]
            PD_variability = np.std(pd_trace)
        else:
            PD_variability = None

    
    accurate = (decision == direction)
    
    

    
    # ---- Gather Information ----
    stim_info = dots_params
    dots_params['direction'] = direction # Convert back to int value for easier analysis. None if no stimulus.

    decision_info = {
    'Decision': decision,
    'Accurate': accurate,
    'RT (ms)': RT*1000 if RT is not None else None,
    'Outcome': outcome,
    'PD Variability': PD_variability
    }

    trial_info = {
        "model_seed": model.seed,
        "task_seed": task_seed,
        "decision_threshold": decision_threshold,
        "max_time (s)": max_time} # Mixing units here, which may be bad practice, but it is easier to read the RT in milliseconds and the max_time in seconds. 
    
    if return_probes:
        probes = {
            'time': sim.trange(),
            'input': sim.data[input_probe],
            'output': sim.data[output_probe],
            'cortex': sim.data[cortex_probe],
            'dSPN': sim.data[dSPN_probe],
            'iSPN': sim.data[iSPN_probe],
            'STN': sim.data[STN_probe],
            'GPe': sim.data[GPe_probe],
            'GPi': sim.data[GPi_probe]
        }

    taskresults = TaskResults(
        stim_info = stim_info,
        decision_info = decision_info,
        trial_info = trial_info,
        probes = probes if return_probes else None
    )

    return taskresults





def run_trials(
        model_params: ModelParams,
        dots_params: DotsParams,
        n_trials: int,
        trial_params: TrialParams | None = None,
        bypass_cortex: bool = False,
        pathway_bias: float | None = None,
        exp_id: int | None = None
        ) -> TrialResults:
    """
    Runs multiple iterations of the decision making task for a given amount of trials. The input parameters are mostly the same for each trial, except for the
    two parameters: task_seed, which is varied to ensure different input stimuli across trials, and direction, which varies to test for bias. 
    A new model is built each new trial, but model_seed is held constant
    to ensure consistency across trials. The results of each trial are stored in a pandas dataframe, and summary statistics are calculated across trials and
    stored in a dictionary

    Parameters
    -----------------
    model_params: ModelParams
        - Dataclass containing the parameters for the entire model. See ModelParams for more details.
    dots_params: DotsParams
        - Dataclass containing the parameters for the input stimulus. See DotsParams for more details.
    n_trials: int
        - Number of trials to run. Must be an even number to ensure equal number of left and right trials, to properly test for bias.
    trial_params: TrialParams, optional
        - Dataclass containing the parameters for the trial. See TrialParams for more details. If None, default parameters will be used.
        Use this to set seeds.
    bypass_cortex: bool, optional
        - If True, bypasses the cortex in the model. Default is False.
    pathway_bias: float | None, optional
        - For simulating the effects of learning on the system.This parameter changes the relative contribution of the direct and indirect pathway. See 
        add_pathway_bias function for more details. If None, there is no change in the relative pathway bias. This parameter cannot be used in conjunction with 
        bypass_cortex = True.
    exp_id: int | None, optional
        - Experiment ID, used for tracking trial and experiment completion. Used only internally and should never be set by the user.


    Returns
    -----------------
    trialresults: TrialResults
        - Dataclass containing the results of the trials. See TrialResults for more details. Contains:
            - trial_data: A pandas dataframe containing the data from each trial.
            - summary_stats: A dictionary containing summary statistics calculated across trials, such as accuracy and mean reaction time.
    """
    if n_trials % 2 != 0:
        raise ValueError("n_trials must be an even number to ensure equal number of left and right trials")
        
    # Set Parameters
    trial_params = trial_params if trial_params is not None else TrialParams()
    task_seed = trial_params.task_seed

    # Create master SeedSequence and spawn seeds
    master_ss = task_seed if isinstance(task_seed, SeedSequence) else SeedSequence(task_seed) # If a SeedSequence is already given, reuse that.
    trial_ss = master_ss.spawn(n_trials)

    # Create rng for bootstrapping
    bootstrap_ss = master_ss.spawn(1)[0]
    bootstrap_rng = default_rng(bootstrap_ss)

    #---- Run Trials and Build Dataframe ----#
    rows = []

    for trial_id, trial_SS in enumerate(trial_ss):
        # Define parameters
        direction = 'right' if trial_id % 2 == 0 else 'left' # Alternate between left and right trials
        trial_params = replace(trial_params, task_seed = trial_SS, terminate_decision = True) # Update task_seed for each trial
        dots_params = replace(dots_params, direction = direction) # Update direction for each trial

        # Run task
        result = run_task(model_params, dots_params, trial_params, return_probes = False, bypass_cortex=bypass_cortex, pathway_bias=pathway_bias)
        row = asdict(result)
        row.pop('probes', None) # Remove the 'probes' column, as it is never included in the dataframe. 
        row['trial_id'] = trial_id
        rows.append(row)

        # Print
        print(f"Trial {trial_id} finished") if exp_id is None else print(f"Experiment {exp_id}, Trial {trial_id} finished")
    
    # Dataframe
    df = pd.json_normalize(rows).set_index('trial_id') # Create flattened dataframe from results.
    df['trial_info.task_seed'] = task_seed # Replace the seeds for input stimulus with the master seed.
    df = df.rename(columns={'trial_info.task_seed': 'trial_info.master_seed'})

    # ==== Calculate Summary Statistics ==== #
    # --- Accuracy ---
    # Mean accuraciy
    accuracy = df['decision_info.Accurate'].mean() 

    # 95% Confidence Interval
    # Count
    accurate_counts = df[df['decision_info.Accurate']]['decision_info.Accurate'].count()
    # CI
    accuracy_ci_low, accuracy_ci_high = proportion_confint(
        count = accurate_counts,
        nobs = n_trials,
        alpha = 0.05,
        method = 'wilson')

    # Accuracy for left and right trials, to test for bias 
    lr_acc = left_right_accuracy_diff_ci(df, alpha = 0.05)

    # --- Reaction times for correct decisions ---
    # Mean
    mean_correct_RT = df[df['decision_info.Outcome'] == 'Correct Decision']['decision_info.RT (ms)'].mean()
    
    # SEM
    mean_correct_RT_sem = df[df['decision_info.Outcome'] == 'Correct Decision']['decision_info.RT (ms)'].sem()

    # 95% confidence interval for mean correct RT
    counts = df[df['decision_info.Outcome'] == 'Correct Decision']['decision_info.RT (ms)'].count()
    # Compute only confidence interval is sample size is large enough. 
    if counts >= 30:
        mean_correct_RT_ci_low, mean_correct_RT_ci_high = t.interval(
            confidence = 0.95,
            df = counts - 1,
            loc = mean_correct_RT,
            scale = mean_correct_RT_sem)
        RT_CI_margin = (mean_correct_RT_ci_high - mean_correct_RT_ci_low)/2 # Margin of error for the confidence interval
    else:
        RT_CI_margin = np.nan

    # std   
    RT_correct = df[df['decision_info.Outcome'] == 'Correct Decision']['decision_info.RT (ms)']
    RT_std_info = bootstrap_ci(RT_correct, 'std', confidence_level = 0.95, n_resamples = 10000, rng = bootstrap_rng, method = 'BCa')
    RT_std = RT_std_info['metric']
    RT_std_ci_low = RT_std_info['ci_low']
    RT_std_ci_high = RT_std_info['ci_high']

    # --- Inverse efficency ---
    inverse_efficiency = mean_correct_RT / accuracy if accuracy > 0 else None

    # --- Outcome ---
    # Means
    correct_decision = (df['decision_info.Outcome'] == 'Correct Decision').mean()
    wrong_decision = (df['decision_info.Outcome'] == 'Wrong Decision').mean() 
    no_decision = (df['decision_info.Outcome'] == 'No Decision').mean()
    premature = (df['decision_info.Outcome'] == 'Premature').mean()

    # 95% confidence interval. Calculated only for wrong decision
    wrong_count = df[df['decision_info.Outcome'] == 'Wrong Decision']['decision_info.Outcome'].count() 
    wrong_decision_ci_low, wrong_decision_ci_high = proportion_confint(
        count = wrong_count,
        nobs = n_trials,
        alpha = 0.05,
        method = 'wilson'
    )

    # --- Post-decision variability ---
    pd_variability = df['decision_info.PD Variability']

    # Mean
    mean_pd_variability = pd_variability.mean()

    # SEM
    mean_pd_variability_sem = pd_variability.sem()

    # 95% confidence interval for mean post-decision variability
    pd_counts = pd_variability.count()
    if pd_counts >= 30:
        mean_pd_variability_ci_low, mean_pd_variability_ci_high = t.interval(
            confidence = 0.95,
            df = pd_counts - 1,
            loc = mean_pd_variability,
            scale = mean_pd_variability_sem)
        pd_variability_CI_margin = (mean_pd_variability_ci_high - mean_pd_variability_ci_low)/2 # Margin of error for the confidence interval
    else:
        pd_variability_CI_margin = np.nan


    # --- Create dictionary of summary statistics ---
    summary_stats = {
        # - Accuracy -
        # Percentages
        'Accuracy (%)': round(float(accuracy * 100), 2),
        'Accuracy Difference (%)': round(float(lr_acc['Accuracy Left-Right Diff'] * 100), 2),
        # Confidence Intervals for accuracy
        'Accuracy CI Low (%)': round(float(accuracy_ci_low * 100), 2),
        'Accuracy CI High (%)': round(float(accuracy_ci_high * 100), 2),
        'Accuracy Diff CI Low (%)': round(float(lr_acc['Diff CI Low'] * 100), 2),
        'Accuracy Diff CI High (%)': round(float(lr_acc['Diff CI High'] * 100), 2),

        # - Reaction Times -
        'Mean Correct RT (ms)': round(float(mean_correct_RT), 2),
        'Mean Correct RT CI Margin (ms)': round(float(RT_CI_margin), 2),
        'Correct RT std (ms)': round(float(RT_std), 2),
        'Correct RT std CI Low (ms)': round(float(RT_std_ci_low), 2),
        'Correct RT std CI High (ms)': round(float(RT_std_ci_high), 2),
        # Inverse efficiency
        'Inverse Efficiency': round(float(inverse_efficiency), 2) if inverse_efficiency is not None else None,
        
        # - Outcome -
        'Correct Decision (%)': round(float(correct_decision * 100), 2),
        'Wrong Decision (%)': round(float(wrong_decision * 100), 2),
        'No Decision (%)': round(float(no_decision * 100), 2),
        'Premature (%)': round(float(premature * 100), 2),
        # Confidence Intervals for wrong decisions
        'Wrong Decision CI Low (%)': round(float(wrong_decision_ci_low * 100), 2),
        'Wrong Decision CI High (%)': round(float(wrong_decision_ci_high * 100), 2),
        # Post-decision variability
        'Mean PD Variability': round(float(mean_pd_variability), 3) if pd_counts > 0 else None,
        'Mean PD Variability CI Margin': round(float(pd_variability_CI_margin), 3) if pd_counts >= 30 else None
    }

    # ---- Enter into dataclass and return ---- #

    trialresults = TrialResults(df, summary_stats)
    return trialresults

def run_experiments(model_params: ModelParams, 
                    base_dots_params: DotsParams,
                    varying_params: dict,
                    n_trials_per_exp: int,
                    trial_params: TrialParams | None = None,
                    return_trials: bool = False,
                    bypass_cortex: bool = False,
                    pathway_bias: float | None = None
                    ) -> ExperimentResults:
                    
    """
    Runs multiple trials of the decision making task for varying parameters. This function is used to test the robustness of the model under varying conditions.
    A set of base dots parameters must be provided, serving as the default parameters for the stimulus. A dictionary of lists must be provided for varying 
    parameters. The function will run a set of trials for each combination of varying parameters. The function will a pandas dataframe containing summary 
    statistics for each combination of varying parameters. Optionally, it returns a dataframe containing the data from each trial, in the same manner as
    'run_trials'.

    Be aware that varying many parameters at once and running many trials can lead to very long simulation times

    Parameters
    ------------------------
    model_params: ModelParams
        - Dataclass containing the parameters for the entire model. See ModelParams for more details.
    base_dots_params: DotsParams
        - Dataclass containing the base parameters for the input stimulus. See DotsParams for more details. These parameters will not be varied except
        explicitly specified in the 'varying_params' dictionary
    varying_params: dict
        - Dictionary of lists containing the parameters to be varied and their corresponding values. The keys must correspond to the parameters in DotsParams.
        The values must be provided as lists, even if only one value is provided.
    n_trials_per_exp: int
        - Number of trials to run for each combination of varying parameters. Must be an even number to ensure equal amounts of left and right stimulus.
    trial_params: TrialParams, optional
        - Dataclass containing the parameters for the trial. See TrialParams for more details. If None, default parameters will be used.
    return_trials: bool, optional
        - Whether or not to return the dataframe containing the data from each trial. Default is False, as this dataframe can become very large.
    bypass_cortex: bool, optional   
        - If True, the input will be connected directly to the basal ganglia, bypassing the cortex. This is useful for testing the behavior of the model without
        the cortex. Default is False.
    pathway_bias: float | None, optional
        - For simulating the effects of learning on the system.This parameter changes the relative contribution of the direct and indirect pathway. See 
        add_pathway_bias function for more details. If None, there is no change in the relative pathway bias. This parameter cannot be used in conjunction with 
        bypass_cortex = True.
    """
    # Set Parameters
    trial_params = trial_params if trial_params is not None else TrialParams()
    task_seed = trial_params.task_seed

    # Create master SeedSequence and spawn seeds
    n_experiments = np.prod([len(v) for v in varying_params.values()])
    master_ss = SeedSequence(task_seed)
    trial_ss = master_ss.spawn(n_experiments)

    # ---- Vary parameters and do experiments ----
    summary_rows = []
    trial_rows = []

    keys = list(varying_params.keys())
    values = [varying_params[k] for k in keys]

    # Experiments
    print(f"Running {n_experiments} experiments with {n_trials_per_exp} trials each, for a total of {n_experiments * n_trials_per_exp} trials.")

    for exp_id, combination in enumerate(product(*values)):
        print(f"Experiment {exp_id} with parameters: {dict(zip(keys, combination))}")
        updates = dict(zip(keys, combination))
        new_params = replace(base_dots_params, **updates)
        trial_params = replace(trial_params, task_seed = trial_ss[exp_id])  

        results = run_trials(model_params = model_params, 
                            dots_params = new_params, 
                            n_trials = n_trials_per_exp,
                            trial_params = trial_params,
                            bypass_cortex = bypass_cortex,
                            pathway_bias = pathway_bias,
                            exp_id = exp_id
                            )
        summary_rows.append({
            "exp_id": exp_id,
            **updates,
            **results.summary_stats
        })
        if return_trials:
            trial_df = results.trial_data.copy()
            trial_df['exp_id'] = exp_id
            trial_df = trial_df.reset_index().set_index(['exp_id', 'trial_id'])
            trial_rows.append(trial_df)

    # Build dataframes
    summary_stats = pd.json_normalize(summary_rows).set_index('exp_id') # Create flattened dataframe from results.

    if return_trials:
        trial_data = pd.concat(trial_rows) # Create flattened dataframe from trial data.
        trial_data['trial_info.master_seed'] = task_seed # Replace the seeds for input stimulus with the master seed.

    experiment_results = ExperimentResults(
        summary_stats = summary_stats,
        trial_data = trial_data if return_trials else None
    )

    return experiment_results


# ============ Function for extracting confidence intervals for plotting ============ #
def get_CI(df, metric):
    """
    Extract value along with confidence intervals for given metric from dataframe containing data. The values are meant to be plotted with plt.errorbar,
    where y corresponds to the value of the metric and yerr corresponds to the error bars for the confidence interval.

    Parameters
    -----------------
    df: pd.DataFrame
        Dataframe containing data for which the metric and confidence should be extracted. 
    metric: str
        Column name of the metric for which confidence intervals should be calculated.
    Returns
    -----------------
    y: pd.Series
        The values of the metric for each row in the dataframe. 
    yerr: np.ndarray
        The error values for the confidence intervals.
    """
    CI_specs = {
    "Accuracy (%)": {
        "low": "Accuracy CI Low (%)",
        "high": "Accuracy CI High (%)",
    },
    "Accuracy Difference (%)": {
        "low": "Accuracy Diff CI Low (%)",
        "high": "Accuracy Diff CI High (%)",
    },
    "Wrong Decision (%)": {
        "low": "Wrong Decision CI Low (%)",
        "high": "Wrong Decision CI High (%)",
    },
    "Correct RT std (ms)": {
        "low": "Correct RT std CI Low (ms)",
        "high": "Correct RT std CI High (ms)",
    },
    "Mean Correct RT (ms)": {
        "margin": "Mean Correct RT CI Margin (ms)",
    },
    "Mean PD Variability": {
        "margin": "Mean PD Variability CI Margin",
    }
}
    
    if metric not in CI_specs:
        raise ValueError(f"Metric '{metric}' does not exist. Available metrics: {list(CI_specs.keys())}")
    spec = CI_specs[metric]
    y = df[metric]

    if "margin" in spec:
        yerr = df[spec["margin"]]
    else:
        lower = y - df[spec["low"]]
        upper = df[spec["high"]] - y
        yerr = np.vstack([lower, upper])
    return y, yerr

