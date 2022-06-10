# GOLUM

GOLUM: Gravitational-wave analysis Of Lensed and Unlensed waveform Models

The GOLUM methodology is aimed at getting fast and accurate parameter estimation for the strongly-lensed gravitational waves. 

The methods' details can be found in
-  A fast and precise methodology to search for and analyze strongly lensed gravitational-wave events, [Janquart et al, 2021](https://academic.oup.com/mnras/article/506/4/5430/6321838?login=false).

### Git structure
This git contains two main folders:
- `golum`: the GOLUM code itself
- `example`: a set of examples showing how the code can be used to do strong-lensing analyses.

### Requirements
`GOLUM` is developed as a package relying on [`bilby`](https://git.ligo.org/lscsoft/bilby), a gravitational-wave data analysis package.

The required Python packages are:
- scipy
- bilby 
- numpy 

We also recommend installing the `pymultinest` sampler, which is used in the examples.

### Citations and related work
If you use `GOLUM` for your research, please cite the associated [method paper](https://academic.oup.com/mnras/article/506/4/5430/6321838?login=false).

Other works using `GOLUM`:
- On the identification of individual gravitational wave image types of a lensed system using higher-order modes, [Janquart et al, 2021](https://arxiv.org/abs/2110.06873)
- Ordering the confusion: A study of the impact of lens models on gravitational-wave lensing detection capabilities, [Janquart et al, 2022](https://arxiv.org/abs/2205.11499v1)

### Other information
If you have any questions or comments, feel free t contact us at j.janquart@uu.nl. 

In case you want to suggest upgrades, bug fixes, etc, do not hesitate to open an issue and/or to do a merge request.