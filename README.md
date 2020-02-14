# Deep Reinforcement Learning with Observational Dropout
Dockerized Tensorflow 2.0 implementation of *Observational Dropout* in OpenAI's new procedurally generated Procgen environment released in December 2019.

This repo 
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

## Tensorflow Versions
This repo supports Tensorflow 2.0 and 2.1 exclusively. 

## Setup
### Installation
- If you don't have Tensorflow 2.0 already, install using the following (gpu recommended):
    ```
    pip install tensorflow-gpu==2.1  # if you have a CUDA-compatible GPU and proper drivers
    ```
- Install OpenAI Gym and Procgen
    ```bash
    pip install gym
    pip install procgen
    ```
- Install tqdm
    ```bash
    pip install tqdm
    ```
- Clone this repository to your home folder
    ```bash
    cd ~
    git clone https://github.com/brianmcera/DRL_ObservationalDropout.git
    cd DRL_ObservationalDropout
    ```

## Test
- Include instructions for how to run all tests after the software is installed
```
# Example

# Step 1
# Step 2
```

## Run Inference
- Include instructions on how to run inference
- i.e. image classification on a single image for a CNN deep learning project
```
# Example

# Step 1
# Step 2
```

## Build Model
- Include instructions of how to build the model
- This can be done either locally or on the cloud
```
# Example

# Step 1
# Step 2
```

## Serve Model
- Include instructions of how to set up a REST or RPC endpoint
- This is for running remote inference via a custom model
```
# Example

# Step 1
# Step 2
```

## Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results
```
# Example

# Step 1
# Step 2
```
