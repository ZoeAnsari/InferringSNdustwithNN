
# Inferring Supenova dust with Neural Networks

## Description

### Model
Here is a designed and trained neural network that esiamtes three target values; amount, temperature and dominant grain species of dust in around supernovae. The outcome of the neural neural network provides an estimation for each target value along with an estimated uncertainty for each estiamed target value. We applied a defined criterion using the estimated uncertatinties to define the reliablity of each estimated target value.

### Data
Our data set consists of a large set of simulated spectral energy distributions (SEDs) for a wide range of different supernova and dust properties using the advanced, fully three-dimensional radiative transfer code MOCASSIN. We trained the neural network with a synthesised photometric data set in which each MOCASSIN SED is convolved with the entire suite of James Web Space Telescope (JWST) bandpass filters.


### Training \& feature selection
To find out how accurately the neural network can predict the dust properties, we fabricate three different scenarios.


First, in S1, we assuming a uniform distance of ∼ 0.43 Mpc for all simulated SEDs.<br/>
Next, in S2, we uniformly distribute all simulated SEDs within a volume of 0.43–65 Mpc.<br/>
Finally, in S3, we artificially add random noise corresponding to a photometric uncertainty of about 0.1 mag.<br/>

Also, for each scenario, we applied a feature selection process using Shapley values, to find the least number of JWST filters to be able to estimate supernova dust properties. Therefore, in each step different subsets of JWST filters are used for training the neural network. In the figures bellow the corresponding filters in each step of the feature selection process are shown.

                 S1                              S2                                S3

<img src="https://user-images.githubusercontent.com/29614210/147156639-70dee60a-7c9e-4888-9890-fa2458a97a95.png" alt="S1" width="30%"/> <img src="https://user-images.githubusercontent.com/29614210/147156716-a44f3e95-e628-4c9c-ab2e-d109ecbd24f7.png" width="30%"/> <img src="https://user-images.githubusercontent.com/29614210/147156777-cc973599-506e-40fa-81ab-2ad2518b7c3c.png" width="30%"/> 


For more details of the training and the evaluation process please see our paper here!




## Dependencies

Keras

[scikit-learn](https://scikit-learn.org/stable/)

[MDN](https://github.com/ZoeAnsari/keras-mdn-layer)

## Usage

 1. By a given SED of a supernova in the format of a subset of JWST filters the amount, temperature and the spcices of dust (only Carbon, Silicate, and a a mixture of Carbon and Silicate), with an estimated uncertainty for each value will be provided by the trained neural network. Moreover, by the applied crietrion on the estimated uncertainties, reliablity of the estimated value for each target value will be shown.

  #### Example

  As an example you may run '__init__predict_newSED.py' which estimates the dust properties for an example set of SEDs from the data set from our S3.



 2. To re-train the neural network with a different set of filters (e.g. filter sets of other instruments), change the path for reading the following tabels under '##read the preprocessed data' comment in __init__training.py:<br/>
'y_data' as the target values<br/>
'otherfeatures' the other parameters of the simulated SEDs that are not included in the training neither as input nor the target values<br/>
'X_mags_cutoff' as the synthetic magnitudes from a set of filters of a desired instrument*.<br/>


  *For producing synthetic magnitudes of our MOCASSIN SEDs with other photometric bandpass filters, download the corresponding transmission curves from [Spanish Virtual Observatory](http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=JWST&asttype=) and upload them in Data->bands directory. Then follow the example code in "src->preprocessing" and change the JWST filters to the desired set of filters.



## Acknowledgement

I acknowledge Roger Wesson for providing the simulated MOCASSIN models, further detailed discussions, and always being helpful and supportive throughout the project. I also had great pleasure of working with Oswin Kruse, without whom I wouldn't be able to tackle the advanced statistical, and technical problems throughout the implementation and evaluation of the project. I would also like to extend my gratitude to Christa Gall, for her excellent guidance, and encouragement throughtout the project. 
