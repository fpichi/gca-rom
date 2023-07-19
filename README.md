# GCA-ROM

GCA-ROM is a library which implements graph convolutional autoencoder architecture as a nonlinear model order reduction strategy.

## Installation

GCA-ROM requires `pytorch`, `pyg`, `matplotlib`, `scipy` and `h5py`. They can be easily installed via `pip` or `conda`.

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
<li> spatial convolutional layers based on MoNet [57] to identify patterns between geomet- rically close regions;</li>
<li> skip-connection operation, to keep track of the original information and help the learn- ing procedure;</li>
<li> a pooling operation, to down-sample the data to obtain smaller networks;</li></ol></li>
<li> a bottleneck, connected to the encoder by means of a dense layer, which contains the latent
behavior in a vector;</li>
<li> a decoder module, recovering the original data by applying the same operations as in the
encoder, but in reverse order.</li>
</ol>

## Cite GCA-ROM

Pichi, F., Moya, B. and Hesthaven, J.S. (2023) ‘A graph convolutional autoencoder approach to model order reduction for parametrized PDEs’. Available at: [arXiv](https://arxiv.org/abs/2305.08573).

If you use GCA-ROM for academic research, you are encouraged to cite the paper using:


```tex
@misc{PichiGraphConvolutionalAutoencoder2023,
  title = {A Graph Convolutional Autoencoder Approach to Model Order Reduction for Parametrized {{PDEs}}},
  author = {Pichi, Federico and Moya, Beatriz and Hesthaven, Jan S.},
  year = {2023},
  number = {arXiv:2305.08573},
  eprint = {2305.08573},
  primaryclass = {cs, math},
  publisher = {{arXiv}},
  doi = {10.48550/arXiv.2305.08573},
  archiveprefix = {arxiv}
}
```

## Authors and contributors
- Federico Pichi (federico.pichi@epfl.ch)
- Beatriz Moya García (beam@unizar.es)

in collaboration with the MCSS group at EPFL of Prof. Jan S. Hesthaven

<!-- with contributions from:

- Name Surname (name.surname@institute.xy) -->

<a href="https://github.com/fpichi/gca-rom/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=fpichi/gca-rom" />
</a>

Made with [contrib.rocks](https://contrib.rocks).