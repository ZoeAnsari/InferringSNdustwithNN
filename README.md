
# Inferring Supenova dust with Neural Networks

##Description

A trained neural network with simulated photometric data sets.
We used the advanced, fully three-dimensional radiative transfer code, MOCASSIN to simulated ... SEDs. Then we conveleved photometric data sets using the JWST bandpass filters.
The neural network is trained with three different photometric data sets defined in three different scenarios as follows:
In the first scenario, S1, 

In the second scenario, S2,

In the third scenario, S3,


## Dependencies

Keras

[scikit-learn](https://scikit-learn.org/stable/)

[MDN](https://github.com/ZoeAnsari/keras-mdn-layer)

## Usage

1.By an optimal or a minimum given set of observed magnitudes with JWST filters as follows:
Optimal:
Minimum:


The neural network that is trained in S3, can estimate the amount, temperature and the spcices of dust (only Carbon, Silicate, and a a mixture of Carbon and Silicate), with an estimated uncertainty for each value.



2.To re-train the neural network with an actual observational data, change ... “path”s in __init__.py, for reading the data and training the neural network.


## Acknowledgement

I acknowledge Roger Wesson for providing the simulated MOCASSIN models, further detailed discussions, and always being helpful and supportive throughout the project. I also had great pleasure of working with Oswin Kruse, without whom I wouldn't be able to tackle the advanced statistical, and technical problems throughout the implementation and evaluation of the project. I would also like to extend my gratitude to Christa Gall, for her excellent guidance, and encouragement throughtout the project. 
