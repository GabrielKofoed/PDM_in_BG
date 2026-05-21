# Perceptual decision making in the basal ganglia

This repository contains code for my bachelor project on a cortico-basal
ganglia-thalamic model for perceptual decision making, with a focus on possible
implementation in neuromorphic hardware.

The model is implemented in Python using Nengo and simulates a two-alternative
perceptual decision-making task inspired by random dot motion experiments.

## Project overview

The project investigates the performance a basal ganglia-inspired action selection
circuit perceptual decision making under ambiguous sensory input. Furthermore,
it investigates whether a cortex-inspired pre-processing unit can improve performance,
and how global modulation of excitability through learning could affect the decision-making
process

The model consists of:

- a cortex-inspired preprocessing module
- a basal ganglia action-selection circuit
- a thalamic output stage
- simulation tools for testing accuracy, reaction time, and pathway bias effects

The task uses left/right stimulus inputs with varying strength, coherence, and
noise.

Furthermore, this repository contains the data and figures generated for the thesis.

## Usage
Download dependencies from requirements.txt and pyproject.toml. If pyproject.toml is not
incorporated correctly, the system will not be able to find the source files.

Look to the scripts folder if you want to reproduce data from the thesis. Look to the notebooks
if you want to reproduce the figures.
You can also run your own experiments. Study the networks carefully along with the simulation builder.

[!CAUTION]

The generated figures depend upon a working Tex installation, such as MikTex of TeX Live.
The notebooks will not run if you do not have Tex installed. Alternatively, remove the line 
`use_latex_fonts()` in the beginning of each notebook. I cannot guarantee that the plots
will look pretty, however.
Also, if you wish to reproduce data by using the given scripts, be aware that data generation
will take a long time. Each script can take a few hours to complete. Alternatively, the amount
of neurons per ensemble can be reduced.

## Repository structure

```text
data/ # Contains the data generated in this project
├── cortical dynamics/ # For testing the effect of the cortex on performance
├── learning/          # For testing the effects of global modulation on excitability
└── model realizations # For testing model performance for different seeds

figures/               # Figures generated for the thesis
notebooks/             # Exploratory notebooks, used for generating figures 
scipts/                # scipts for generating data

src/
├── networks/          # Model components
    ├── cbgt.py        # CBGT model builder
    └── cortex.py      # Cortex builder
└── utilities/         # Simulation builder, statistics, and helper functions
    ├── io.py          # Functions for saving data, loading data, and generating figures.
    ├── simulation.py  # Simulation builder. For running tasks and generating data
    └── statistics.py  # Helper functions for calculating statistics with confidence intervals

notebooks/           # Exploratory notebooks and figure generation
pyproject.toml       # Configuration. It allows src to be installed as a local package.
requirements.txt     # Python dependencies
README.md            # Project description
LICENSE              # Repository license
