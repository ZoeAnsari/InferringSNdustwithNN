import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.noise_S_N10 import add_random_noise

def preprocess(name_):

    #####------------------------------------------
    x_mags=pd.read_pickle("Data/All_MOCASSIN"+str(name_)+"/data_mags.pkl")
    data_tot=pd.read_pickle("Data/All_MOCASSIN"+str(name_)+"/data_tot.pkl")




    print(np.isnan(data_tot["Md"]).any())
    data_tot=data_tot[~data_tot.duplicated(keep=False)]
    data_tot["tempDust"]=data_tot["tempDust"].replace(np.nan, int(-1))
    data_tot=data_tot[data_tot["tempDust"] != int(-1)]#785

    data_tot["Md"] = data_tot["Md"].replace(1e-20,0)
    data_tot["species"] = data_tot["species"].replace(0.5,0.75)
    data_tot["species"] = data_tot["species"].replace(0,0.5)



    ###defining no-dust models
    temp_thresh=800

    for i in data_tot.index:
        if data_tot["Md"][i] < 5e-5:
            if data_tot["tempDust"][i] < temp_thresh:
                data_tot["species"][i] = -0.5


    data_tot.to_pickle("Data/All_MOCASSIN"+str(name_)+"/data_tot_nodust.pkl")

    x_mags=x_mags[x_mags.index.isin(data_tot.index)]

    # exit()

    ################replacing -innf with 0
    for i_col in x_mags.columns:
        x_mags[i_col]=x_mags[i_col].replace(-np.inf, -20)
        x_mags[i_col] = x_mags[i_col].replace(np.inf, 0)
    print(np.isinf(x_mags).any())
    print(np.isnan(x_mags).any())





    otherfeatures_data=data_tot[["tempDust", 'dusttemp_sigma', 'tempSN','radiusSN', 'L', 'Md', 'grainSize', 'tau',  "file", "Model_x",
                              "species", "z", "massDust"]]

    otherfeatures_data=otherfeatures_data.rename(columns={
                                                          "Model_x": "Model",
                                                          })


    otherfeatures_data_check=otherfeatures_data.drop(columns={'Model'})
    # print(np.isnan(otherfeatures_data_check).any())


    ###Normalisinng target values
    y_data=pd.DataFrame(1e+1 * otherfeatures_data["Md"])
    y_data["tempDust"]=(( otherfeatures_data['tempDust'])/2200)




    otherfeatures_data["species"]=otherfeatures_data["species"].astype(float)
    y_data["species"]= otherfeatures_data['species']
    y_data = y_data.replace([np.inf, -np.inf], np.nan)
    # print(y_data.isna().sum())
    # print(y_data.isnull().any())

    path_preprocessing="Data/All_MOCASSIN"+str(name_)
    x_mags.to_pickle(str(path_preprocessing)+"/preprocessed_mags.pkl")
    y_data.to_pickle(str(path_preprocessing)+"/y_data.pkl")
    otherfeatures_data.to_pickle(str(path_preprocessing)+"/otherfeatures_data.pkl")
    if name_ == '_Z':
        path_preprocessing_="Data/All_MOCASSIN"+str(name_)+"ERROR"
        y_data.to_pickle(str(path_preprocessing)+"/y_data.pkl")
        otherfeatures_data.to_pickle(str(path_preprocessing)+"/otherfeatures_data.pkl")
        add_random_noise(x_mags, path_preprocessing_)


