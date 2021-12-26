import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from src.preparing_format import preparing_tabular_data
from src.preprocessing import preprocess
from src.sensitvity_brightness import sensitivity_and_Brightness_cutoff
from src.training import NN_reg_dense
import pickle

preparing_tabular_data()
name_="_Z"
preprocess(name_)


sensitivity_Jy_list_NIRCam = np.array([22.5, 15.3, 13.2, 19.4, 10.6,
                                       21.4, 145, 16.1, 133, 9.1,
                                       14.9, 129, 32.1,  14.3, 25.8,
                                       194, 9.1, 21.8, 12.1, 20.7,
                                       158, 24.7, 50.9, 23.6, 46.5,
                                       274, 302, 67.9]) * 1e-9 #S/N = 10 in 10,000 s
Bright_MagVega_list_NIRCam = [14.43, 15.24, 15.44, 14.62, 15.37, 14.40,
                              11.99, 14.36, 11.68, 14.80, 13.66, 11.38,
                              14.14, 15.23, 13.93, 15.66, 11.02, 13.65,
                              14.42, 13.45, 10.44, 13.06, 12.09, 13.68,
                              11.52, 9.64, 9.45, 11.55]  #FULL (21.4 s)

sensitivity_Jy_list_MIRI = np.array([0.13, 0.24, 0.52, 1.22, 0.92, 1.45, 2.97, 5.17, 17.3]) * 1e-6#S/N = 10 in 10 ksec
Bright_Jy_list_MIRI = np.array([4.07, 3.58, 7.82, 33.11, 13.65, 17.54, 31.79, 32.42, 100.23]) * 1e-3#Full Frame


telescope_name = "JWST"
name_="_Z"
path_preprocessing="Data/All_MOCASSIN"+str(name_)
preprocessed_mags=pd.read_pickle(str(path_preprocessing)+"/preprocessed_mags.pkl")


sensitivity_and_Brightness_cutoff(preprocessed_mags, path_preprocessing, name_,sensitivity_Jy_list_NIRCam, Bright_MagVega_list_NIRCam,
                                  sensitivity_Jy_list_MIRI, Bright_Jy_list_MIRI,telescope_name)




