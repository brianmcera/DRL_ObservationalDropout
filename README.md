# Deep Reinforcement Learning with Observational Dropout
Dockerized Tensorflow 2.0 implementation of *Observational Dropout* in OpenAI's new procedurally generated Procgen environment released in December 2019.

To learn more about Google Brain's research on Observational Dropout, check out this [direct link to their github page](https://learningtopredict.github.io/ "Observational Dropout"). 

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
- This repo supports Tensorflow 2.0 and 2.1 exclusively. If you don't have Tensorflow 2.0 already, install using the following (gpu recommended):
    ```
    >> pip install tensorflow-gpu==2.1  # if you have a CUDA-compatible GPU and proper drivers
    ```
- Install OpenAI Gym and Procgen
    ```bash
    >> pip install gym
    >> pip install procgen
    ```
- Install tqdm
    ```bash
    >> pip install tqdm
    ```
- Clone this repository to your home folder
    ```bash
    >> cd ~
    >> git clone https://github.com/brianmcera/DRL_ObservationalDropout.git
    >> cd DRL_ObservationalDropout
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
   *bigfish, bossfight, caveflyer, chaser, climber, coinrun, dodgeball, fruitbot, heist, jumper, leaper, maze, miner, ninja, plunder,* and *starpilot.* Please see the official Procgen Github for more details on each environment.

