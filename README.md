# Generalizable Deep Reinforcement Learning with Observational Dropout
Dockerized Tensorflow 2.0 implementation of *Observational Dropout* in OpenAI's new procedurally generated [Procgen](https://openai.com/blog/procgen-benchmark/) environment released in December 2019. To learn more about Google Brain's research on Observational Dropout, check out this [direct link to their github page](https://learningtopredict.github.io/ "Observational Dropout"). 

The gif below shows an untrained Reinforcement Learning agent exploring the *Starpilot* environment from OpenAI's new Procgen benchmark suite.

![untrained RL agent](https://github.com/brianmcera/DRL_ObservationalDropout/blob/master/static/RL1.gif) 

## Project Motivation
Deep Reinforcement Learning (RL) is an amazing AI approach that has enabled us to solve some really [interesting](https://openai.com/blog/emergent-tool-use/) and [challenging](https://deepmind.com/research/case-studies/alphago-the-story-so-far) [problems](https://openai.com/blog/openai-five/) by leveraging the ability to probe interactions in real or simulated environments. As Reinforcement Learning approaches further develop, RL has the potential to revolutionize and disrupt billion dollar industries in robotics and autonomous systems, learning recommender systems, adaptive optimizers, real-time financial tech, and more.

![RL agent trained with PPO](https://github.com/brianmcera/DRL_ObservationalDropout/blob/master/static/RL2.gif)

Despite this, Reinforcement Learning has struggled to become as widely adopted in large-scale engineering applications due to these algorithms' [brittleness to hyperparameter selection](https://arxiv.org/abs/1811.02553), [lack of explainability or safety guarantees](https://arxiv.org/abs/1606.06565), and [poor reproducibility](https://arxiv.org/abs/1904.12901). In particular, two challenges impeding more widespread adoption of Reinforcement Learning approaches are: 1) sample efficiency and 2) domain transfer. To motivate and challenge researchers to understand how generalizable, reliable, and performant current state-of-the-art RL approaches are, OpenAI released the Procgen environment in December 2019 which features randomly procedurally generated environments to help assess whether Reinforcement Learning agents learn truly generalizable skills or if they're just memorizing optimal trajectories for each environment.

In leveraging this new simulation environment, this project investigates an interesting approach called *Observational Dropout* to see if forcing RL agents to use 'imagined' future states can lead to more generalizable behavior even with limited sampled data. This more challenging operating condition can be representative of situations where observational sensor data might be difficult or expensive to acquire and can highlight some of the challenges regarding sample efficiency and domain transfer with Reinforcement Learning.

## Organization for this project repo:
- **src** : Source code for production within structured directory
- **tests** : Scripts for testing 
- **configs** : Preset model variables within single directory 
- **data** : Small amount of data in the Github repository so tests can be run to validate installation
- **build** : Scripts that automate building of a standalone environment
- **static** : Images or content for the README 

## Prerequisites
- Python 3
- NVIDIA GPU & CUDA cuDNN
- [OpenAI Gym](https://gym.openai.com/)
- [OpenAI Procgen](https://openai.com/blog/procgen-benchmark/)
- Conda (Anaconda or Miniconda)
- [Streamlit](https://www.streamlit.io/)

## Setup
### Installation
- This repo supports Tensorflow 2.0 and 2.1 exclusively. If you don't have Tensorflow 2.0 already, install using the following ([GPU version](https://www.tensorflow.org/install/gpu) recommended if you have a CUDA-compatible GPU and proper drivers):
    ```bash
    >> pip install tensorflow-gpu==2.1  
    ```
- Clone this repository to your home folder
    ```bash
    >> cd ~
    >> git clone https://github.com/brianmcera/DRL_ObservationalDropout.git
    >> cd DRL_ObservationalDropout
    ```
- Automatically install using Conda and the requirements.txt file: 
    ```bash
    >> conda create --name OD-DRL --file requirements.txt
    ```
- or using the provided YAML file:
    ```bash
    >> conda env create -f OD-DRL.yml
    ```

## Train your own RL Agent!
- To run a training session with a Reinforcement Learning agent using the Proximal Policy Optimization algorithm, run the following commands from the ROOT of this repo (the -v flag toggles visualization of the runs ON):
    ```bash
    >> python src/agents/PPO_Agent.py -v
    ```
- To compare with this agent, run a Reinforcement Learning agent using added Observational Dropout:
    ```bash
    >> python src/agents/OD_PPO_Agent.py -v
    ```
- If you'd like to compare how you'd perform on these very same tasks, you can interactively play with these games using the following command:
    ```bash
    >> python -m procgen.interactive --env_name starpilot
    ```
The keys are: left/right/up/down + q, w, e, a, s, d for the different (environment-dependent) actions. 
Possible environments that you can try are:
   *bigfish, bossfight, caveflyer, chaser, climber, coinrun, dodgeball, fruitbot, heist, jumper, leaper, maze, miner, ninja, plunder,* and *starpilot.* Please see the [official Procgen Github](https://github.com/openai/procgen) for the specific details of each environment.
   
## Results, Analysis, and Discussion
Check back here for interesting results/analysis to come!

