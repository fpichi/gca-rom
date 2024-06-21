# GCA-ROM 

GCA-ROM is a library which implements graph convolutional autoencoder architecture as a nonlinear model order reduction strategy.

## Installation

GCA-ROM requires `pytorch`, `pyg`, `matplotlib`, `scipy` and `h5py`. 
They can be easily installed via `pip` or `conda`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/fpichi/gca-rom/tree/main/notebook)

**In the notebook folder, one can find the `*.ipynb` files corresponding to the tutorials to run the models in Google Colab without installing the package.**

#### **MacOS**
The latest version of `pyg` is currently not available on `conda`. The required dependencies, exported in *utils/gca_rom.yml*, can be automatically installed in a new environment via  

```bash
conda env create -f gca_rom.yml
```

#### **Linux**

```bash
conda create -n 'gca_rom' python=3.10
conda activate gca_rom
conda install pytorch -c pytorch 
conda install pyg -c pyg
conda install matplotlib pandas scipy jupyter h5py
```

The official distribution is on GitHub, and you can clone the repository using

```bash
git clone git@github.com:fpichi/gca-rom.git
```

## Summary of GCA-ROM Features

### - OFFLINE PHASE
<!-- ![](docs/images/gca_off_1.png) -->
<p align="center">
<img src="docs/images/gca_off_1.png"/>
</p>

### - ONLINE PHASE
<!-- ![](docs/images/gca_on_1.png) -->
<p align="center">
<img src="docs/images/gca_on_1.png" width="70%" height="70%"/>
</p>
The proposed modular architecture, namely Graph Convolutional Autoencoder for Reduced Order Modelling (GCA-ROM), subsequently exploits:
<ol>
<li> a graph-based layer to express an unstructured dataset;</li>
<li> an encoder module compressing the information through:<ol>
<li> spatial convolutional layers based on <a href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Monti_Geometric_Deep_Learning_CVPR_2017_paper.pdf">MoNet</a> to identify patterns between geometrically close regions;</li>
<li> skip-connection operation, to keep track of the original information and help the learning procedure;</li>
<li> a pooling operation, to down-sample the data to obtain smaller networks;</li></ol></li>
<li> a bottleneck, connected to the encoder by means of a dense layer, which contains the latent
behavior in a vector;</li>
<li> a decoder module, recovering the original data by applying the same operations as in the encoder, but in reverse order.</li>
</ol>

## Tutorials

The nonlinear ROM methodology has been tested on 14 different benchmarks, including:
* scalar/vector and linear/nonlinear equations (*01_poisson.ipynb*)
* advection-dominated regime (*02_advection.ipynb*)
* physical and geometrical parametrizations (*03_graetz.ipynb*)
* bifurcating phenomena (*04_navier_stokes_vx.ipynb*, *05_navier_stokes_vy.ipynb*, *06_navier_stokes_p.ipynb*)
* time-dependent models (*07_diffusion.ipynb*, *08_poiseuille.ipynb*)
* a 3D elastic problem (*09_elasticity.ipynb*)
* high-dimensional parametric applications (*10_stokes.ipynb*)
* complex time-dependent problems (*11_holed_advection.ipynb*, *12_lid_driven_cavity.ipynb*, *13_moving_hole_advection.ipynb*)

To run a benchmark, navigate to the tutorial folder and run the corresponding *file.ipynb*. If available, a GUI will open with preset values for the hyperparameter configuration of the network. Once the window is closed, the code starts the training phase, unless a trained model with the same configuration already exists. 

<p align="center">
<img src="docs/images/gui.png" width="90%" height="70%"/></p>

After the GCA-ROM is evaluated, many plots are automatically generated, ranging from training losses, latent evolution, relative errors, solution and error fields, and gif of the dynamics.
Below are some snaphots of the approximated solutions for the available benchmarks:

<p align="center">
<img src="tutorials/poisson/_standard/U_poisson_lmap10.0_btt15_seed10_lv4_hc3_nd50_ffn200_skip1_lr0.001_sc4_rate30/field_solution_63_standard_U.png" width="35.7%" height="70%"/>
<img src="tutorials/advection/_standard/U_advection_lmap10.0_btt15_seed10_lv4_hc2_nd100_ffn200_skip1_lr0.001_sc4_rate30/field_solution_46_standard_U.png" width="31%" height="70%"/>
<img src="tutorials/elasticity/_standard/U_elasticity_lmap10.0_btt15_seed10_lv4_hc3_nd50_ffn200_skip1_lr0.001_sc4_rate30/field_solution_79_standard_U.png" width="32%" height="70%"/>
<img src="tutorials/graetz/_standard/U_graetz_lmap10.0_btt25_seed10_lv4_hc2_nd50_ffn200_skip1_lr0.001_sc4_rate30/field_solution_175_standard_U.png" width="60.1%" height="70%"/>
<img src="tutorials/graetz/_standard/U_graetz_lmap10.0_btt25_seed10_lv4_hc2_nd50_ffn200_skip1_lr0.001_sc4_rate30/field_solution_4_standard_U.png" width="38.3%" height="70%"/>
<img src="tutorials/poiseuille/_standard/U_poiseuille_lmap10.0_btt15_seed10_lv4_hc3_nd50_ffn200_skip1_lr0.001_sc4_rate30/field_solution_388_standard_U.png" width="62%" height="70%"/>
<img src="tutorials/diffusion/_standard/U_diffusion_lmap10.0_btt15_seed10_lv4_hc3_nd50_ffn200_skip1_lr0.001_sc4_rate30/field_solution_27_standard_U.png" width="34.85%" height="70%"/>
<img src="tutorials/stokes_u/_standard/U_stokes_u_lmap10.0_btt25_seed10_lv4_hc2_nd50_ffn200_skip1_lr0.001_sc4_rate30/field_solution_7_standard_U.png" width="32%" height="70%"/>
<img src="tutorials/stokes_u/_standard/U_stokes_u_lmap10.0_btt25_seed10_lv4_hc2_nd50_ffn200_skip1_lr0.001_sc4_rate30/field_solution_266_standard_U.png" width="32.5%" height="70%"/>
<img src="tutorials/stokes_u/_standard/U_stokes_u_lmap10.0_btt25_seed10_lv4_hc2_nd50_ffn200_skip1_lr0.001_sc4_rate30/field_solution_326_standard_U.png" width="30%" height="70%"/>
<img src="tutorials/navier_stokes/_standard/VX_navier_stokes_lmap1.0_btt25_seed10_lv4_hc3_nd100_ffn200_skip1_lr0.001_sc4_rate10/field_solution_57_standard_U.png" width="49%" height="70%"/>
<img src="tutorials/navier_stokes/_standard/VY_navier_stokes_lmap1.0_btt25_seed10_lv4_hc3_nd100_ffn200_skip1_lr0.001_sc4_rate10/field_solution_150_standard_U.png" width="49%" height="70%"/>
</p>


## Cite GCA-ROM

[1] Pichi, F., Moya, B. and Hesthaven, J.S. (2023) ‘A graph convolutional autoencoder approach to model order reduction for parametrized PDEs’. Available at: [arXiv](https://arxiv.org/abs/2305.08573), [Journal of Computational Physics](https://www.sciencedirect.com/science/article/pii/S0021999124000111?via%3Dihub)

If you use GCA-ROM for academic research, you are encouraged to cite the paper using:


```tex
@article{PichiGraphConvolutionalAutoencoder2024,
  title = {A Graph Convolutional Autoencoder Approach to Model Order Reduction for Parametrized {{PDEs}}},
  author = {Pichi, Federico and Moya, Beatriz and Hesthaven, Jan S.},
  year = {2024},
  journal = {Journal of Computational Physics},
  volume = {501},
  pages = {112762},
  doi = {10.1016/j.jcp.2024.112762},
  urldate = {2024-01-18}
}
```

## Authors and contributors
- Federico Pichi (fpichi@sissa.it)
- Beatriz Moya García (beatriz.moya@cnrsatcreate.sg)

in collaboration with the MCSS group at EPFL of Prof. Jan S. Hesthaven.

With contributions from:

- Oisín Morrison (oisinmmm@gmail.com)
- Francesco Pettenon (francesco.pettenon@epfl.ch)
- Francesco Sala (francesco.sala@epfl.ch)
- Leonardo Trentini (leonardo.trentini@epfl.ch)
