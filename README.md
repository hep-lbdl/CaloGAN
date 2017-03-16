# caloGAN
Generative Adversarial Networks for High Energy Physics extended to a multi-layer calorimeter simulation

# Generation

To build the generation code on PDSF, simply run `source cfg/pdsf-env.sh` from the `generation/` folder in the repository. This loads modules.

Next, you can type `make` which should build an executable called `generate`. Because of how Geant4 works, this executable gets deposited in `~/geant4_workdir/bin/Linux-g++/`, which is in your `$PATH` when the modules from `cfg/pdsf-env.sh` are loaded.	

To run the generation script, run `generate -m cfg/run2.mac`.

