
# Inferring Supenova dust with Neural Networks

## Description


Here is a designed neural network whose output models a normal distribution to estimate a target value.
We produced a large set of simulated spectral energy distributions (SEDs) for a wide range of different supernova and dust properties using the advanced,
fully three-dimensional radiative transfer code MOCASSIN. We then convolved each SED with the entire suite of James Web Space
Telescope (JWST) bandpass filters to synthesise a photometric data set.



To find out how accurately the neural network can predict the dust properties from the simulated data, we fabricate three different scenarios.


First, in S1, we assuming a uniform distance of ∼ 0.43 Mpc for all simulated SEDs.

Next, in S2, we uniformly distribute all simulated SEDs within a volume of 0.43–65 Mpc.

Finally, in S3, we artificially add random noise corresponding to a photometric uncertainty of about 0.1 mag.

We also applied a feature selection process using Shapley values, in each scenario to find the least number of JWST filters to be able to estimate supernova dust properties. Therefore, in each step different subsets aree used for training the neural network. In the figures bellow the corresponding filters in each step of the feature selection process are shown.

S1:
 
  <img src="https://user-images.githubusercontent.com/29614210/147156639-70dee60a-7c9e-4888-9890-fa2458a97a95.png" width="35%">


S2: 

<img src="https://user-images.githubusercontent.com/29614210/147156716-a44f3e95-e628-4c9c-ab2e-d109ecbd24f7.png" width="35%">


S3: 

<img src="https://user-images.githubusercontent.com/29614210/147156777-cc973599-506e-40fa-81ab-2ad2518b7c3c.png" width="35%">




For more details of the process of training and evaluation please see our paper here!




## Dependencies

Keras

[scikit-learn](https://scikit-learn.org/stable/)

[MDN](https://github.com/ZoeAnsari/keras-mdn-layer)

## Usage

1.By a given SED of a supernova in the format of a subset of JWST filters that is defined in our feature selection process and are shhown in the Figures above, the amount, temperature and the spcices of dust (only Carbon, Silicate, and a a mixture of Carbon and Silicate), with an estimated uncertainty for each value will be estimated that we apply a crietrion on to label a predicted value as reliable/unreliable.

### Example

As an example you may run '__init__predict_newSED.py' which estimates the dust properties for an example set of SEDs from the data set in our S3.



2.To re-train the neural network with an actual observational data, change ... “path”s in __init__.py, for reading the data and training the neural network.


## Acknowledgement

I acknowledge Roger Wesson for providing the simulated MOCASSIN models, further detailed discussions, and always being helpful and supportive throughout the project. I also had great pleasure of working with Oswin Kruse, without whom I wouldn't be able to tackle the advanced statistical, and technical problems throughout the implementation and evaluation of the project. I would also like to extend my gratitude to Christa Gall, for her excellent guidance, and encouragement throughtout the project. 
