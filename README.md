# GCA-ROM

GCA-ROM is a library which implements graph convolutional autoencoder architecture as a nonlinear model order reduction strategy.

## Requirements

The required packages are contained in the file requirements.txt, which can be used to create the conda virtual environment with:

```
$ conda create --name <env> --file requirements.txt
```

## Summary of GCA-ROM Features

### - OFFLINE PHASE
![](docs/images/gca_off_1.png)

### - ONLINE PHASE
![](docs/images/gca_on_1.png)

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

If you use GCA-ROM for academic research, you are encouraged to cite the following paper:

```
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

