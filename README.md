# CaloGAN
Simulating 3D High Energy Particle Showers in Multi-Layer Electromagnetic Calorimeters with Generative Adversarial Networks

## Goal
The goal of this project is to help physicists at CERN speed up their simulations by encoding the most computationally expensive portion of the simulation process (i.e., showering in the EM calorimeter) in a deep generative model.

The challenges come from the fact that this portion of the detector is longitudinally segmented into three layers with heterogeneous granularity. For simplicity, we can visualize the energy depositions of particles passing through the detector as a series of three images per shower, but it is important to encode the sequential nature of their relationship in the generator. 

![3D shower in the EM calorimeter](/figures/3d.jpg)

## Getting Started

This repository contains three main folders: `generation`, `models` and `analysis`, which represent the three main stages in this project. You may decide to only engage in a subset of them, so we decided to provide you with all you need to jump right in at the level you are interested in. 

`generation` contains the code to build the electromagnetic calorimeter geometry and shoot particles at it with a give energy. This is all based on `Geant4`. For more instructions, go to [Generation on PDSF](#generation-on-pdsf).

`models` contains the core ML code for this project. The file `train.py` takes care of loading the data and training the GAN. 

`analysis` contains Jupyter Notebooks used to evaluate performance.

## Generation on PDSF

On PDSF, your should run all generation code that outputs big files to a scratch space. To make a scratch environment, run `mkdir /global/projecta/projectdirs/atlas/{your-username}`, and for convenience, link to home via `ln -s /global/projecta/projectdirs/atlas/{your-username} ~/scratch`.

To build the generation code on PDSF, simply run `source cfg/pdsf-env.sh` from the `generation/` folder in the repository. This loads modules.

Next, you can type `make` which should build an executable called `generate`. Because of how Geant4 works, this executable gets deposited in `~/geant4_workdir/bin/Linux-g++/`, which is in your `$PATH` when the modules from `cfg/pdsf-env.sh` are loaded.	

To run the generation script, run `generate -m cfg/run2.mac`.

To launch a batch job on PDSF, simply run `./launch <num_jobs>`, to launch `<num_jobs>` concurrent tasks in a job array.

## The CaloGAN Model
### Generator
![Generator](figures/caloGAN_gen.jpg)
### Discriminator
![Discriminator](figures/caloGAN_discr_rev.jpg)
