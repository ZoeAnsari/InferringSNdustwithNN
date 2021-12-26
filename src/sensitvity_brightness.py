import pandas as pd
import numpy as np
import pickle
import time as time

def sensitivity_and_Brightness_cutoff(X_Mags, path_preprocessing, name_, sensitivity_Jy_list_NIRCam, Bright_MagVega_list_NIRCam,
                                      sensitivity_Jy_list_MIRI, Bright_Jy_list_MIRI,telescope_name):


    sensitivity_MagAB_list_NIRCam = (-2.5 * (np.log10(sensitivity_Jy_list_NIRCam) - 23)) - 48.6


    df = pd.read_csv("Data/table.csv")
    table = df[df["name"].str.contains(telescope_name)]
    ##drop 150w2 from JWST
    a = ["JWST_NIRCAM_F150W2"]
    table = table[~table['name'].isin(a)]

    Vega_zero_mag = table["Vega mag"]
    AB_zero_mag = table["AB mag"]
    table["offset"] = table["AB mag"] - table["Vega mag"]


    offset = np.array(table["offset"])

    Bright_MagAB_list_NIRCam = Bright_MagVega_list_NIRCam + offset

    Bright_Jy_list_NIRCam = 10 ** (23 - (Bright_MagAB_list_NIRCam + 48.6) / 2.5)



    # Mag= -2.5 * np.log10(Jy)


    sensitivity_MagAB_list_MIRI = (-2.5 * (np.log10(sensitivity_Jy_list_MIRI) - 23)) - 48.6



    Bright_MagAB_list_MIRI = (-2.5 * (np.log10(Bright_Jy_list_MIRI) - 23)) - 48.6

    # 10,000 secinds
    sensitivity_MagAB_list = list(sensitivity_MagAB_list_NIRCam) + list(sensitivity_MagAB_list_MIRI)
    sensitivity_mJy_list = list(sensitivity_Jy_list_NIRCam * 1e+3) + list(sensitivity_Jy_list_MIRI * 1e+3)



    Bright_MagAB_list = list(Bright_MagAB_list_NIRCam) + list(Bright_MagAB_list_MIRI)
    Bright_mJy_list = list(Bright_Jy_list_NIRCam * 1e+3) + list(Bright_Jy_list_MIRI * 1e+3)



    for i_col in X_Mags.columns:
        for j_index in X_Mags.index:
            if X_Mags[i_col][j_index] > sensitivity_MagAB_list[i_col]:
                X_Mags[i_col][j_index] = sensitivity_MagAB_list[i_col]
            elif X_Mags[i_col][j_index] < Bright_MagAB_list[i_col]:
                X_Mags[i_col][j_index] = Bright_MagAB_list[i_col]

    # X_Mags_extra=pd.DataFrame(columns='dropped_fil')
    #
    # for j_index in X_Mags.index:
    #     col_dropped = []
    #     for i_col in X_Mags.columns:
    #         if X_Mags[i_col][j_index] > sensitivity_MagAB_list[i_col]:
    #             X_Mags[i_col][j_index] = sensitivity_MagAB_list[i_col]
    #             # col_ind.append(0)
    #             col_dropped.append(1)
    #         elif X_Mags[i_col][j_index] < Bright_MagAB_list[i_col]:
    #             X_Mags[i_col][j_index] = Bright_MagAB_list[i_col]
    #             col_dropped.append(1)
    #     X_Mags_extra['dropped_fil'][j_index]=np.sum(col_dropped)
    #
    # X_Mags_extra.to_pickle(str(path_preprocessing)+"/X_Mags_extra"+str(name_)+".pkl")


    file_name_Bright_MagAb_list=str(path_preprocessing)+"/Bright_MagAB_list"+str(name_)+".pkl"
    open_file = open(file_name_Bright_MagAb_list, "wb")
    pickle.dump(Bright_MagAB_list, open_file)
    open_file.close()

    file_name_sensitivity_MagAB_list = str(path_preprocessing)+"/sensitivity_MagAB_list" + str(name_) + ".pkl"
    open_file = open(file_name_sensitivity_MagAB_list, "wb")
    pickle.dump(sensitivity_MagAB_list, open_file)
    open_file.close()

    file_name_Bright_mJy_list=str(path_preprocessing)+"/Bright_mJy_list"+str(name_)+".pkl"
    open_file = open(file_name_Bright_mJy_list, "wb")
    pickle.dump(Bright_mJy_list, open_file)
    open_file.close()

    file_name_sensitivity_mJy_list = str(path_preprocessing)+"/sensitivity_mJy_list" + str(name_) + ".pkl"
    open_file = open(file_name_sensitivity_mJy_list, "wb")
    pickle.dump(sensitivity_mJy_list, open_file)
    open_file.close()
    #
    X_Mags.to_pickle(str(path_preprocessing)+"/mags_with_senbri_cutoff.pkl")

