import pandas as pd
from src.convert_data_format import convert_data_to_tables, preprocess_concat
from astropy.cosmology import FlatLambdaCDM , z_at_value
import astropy.units as u
from src.plot_sample_SEDs import plot_sample, plot_sample_bands
import pickle
from src.pyphot_filters import set_filters
# from src.NIRCam_bands_from_scratch_Apr import NIRCam_bands_Apr
# from src.MIRI_bands_from_scratch_Apr import MIRI_bands_Apr
import os
import numpy as np


def preparing_tabular_data():


    ##TODO: define a cosmological model
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)



    path_preprocessing="Data/Results_from_MOCASSIN"
    if not os.path.exists("{}/preprocessed".format(str(path_preprocessing))):
        os.makedirs("{}/preprocessed".format(str(path_preprocessing)))
    ##TODO : change the following values according to the size of the results from MOCASSIN
    range1_outputs=0 #change accordingly after downloading all data
    range11_outputs=1#
    range2_outputs=20#00 #change accordingly after downloading all data
    dataset_label=""
    ###example : outputs_"+str(dataset_label)+str(c1)+"/output_"+str(c12)+str(c2)+"/output/"
    ### c1: all values from range1_outputs to range11_outputs
    ### c12:all values from range1_outputs to range11_outputs
    ### c2: all values in range range2_outputs


    # #####preprocessing
    name_="_Z" ### A scenario in which we place the SEDs at different redshfits


    z_upper_limit=150 # unit is 1e-4
    #TODO: "comment/uncomment here"
    #
    convert_data_to_tables(path_preprocessing, name_, range1_outputs,
                           range11_outputs,range2_outputs, z_upper_limit, dataset_label)

    preprocess_concat(path_preprocessing, name_, range1_outputs,
                      range11_outputs,range2_outputs)


    ### reading the preprocessed data
    data=pd.read_pickle(str(path_preprocessing) + "/preprocessed/allSEDs_"+str(name_)+".pkl")


    data=data.rename(columns={"silicate": "species",
                              'dusttemp_mu':"tempDust",
                              'Rout' :'radiusSN',
                              'T':  'tempSN',
                              'gs': 'grainSize',
                              'dustmass': 'massDust'})
    data.to_pickle("Data/All_MOCASSIN"+str(name_)+"/data_tot.pkl")

    data["Md"]=data["massDust"]
    otherfeatures_data=data[["tempDust", 'dusttemp_sigma', 'tempSN','radiusSN', 'L', 'Md', 'grainSize', 'tau',  "file", "Model_x",
                             "Model_y", "species", "z", "massDust"]]


    #####




    path_filterbands=str(path_preprocessing)+"/photometricBands"+str(name_)
    if not os.path.exists("{}/photometricBands{}".format(str(path_preprocessing),
                                                          str(name_))):
        os.makedirs("{}/photometricBands{}".format(str(path_preprocessing),
                                                    str(name_)))
    title=str(name_)
    file_name_z_list = str(path_preprocessing)+"/preprocessed/z_list_withoutDrop" + str(name_) + ".pkl"
    open_file = open(file_name_z_list, "rb")
    z_list_withoutDrop=pickle.load(open_file)
    open_file.close()


    otherfeatures_data.index=range(len(data))

    flux = data.drop(columns={"tempDust", 'dusttemp_sigma', 'Model_x','tempSN','radiusSN',
                              'L', 'Md',  'grainSize',
                              'Model_y', 'tau', 'file',  "species", "z", "massDust","index"})

    #####plot a bunch of MOCASSIN models before convolving the photometric bands
    ##TODO : to plot differnt models change the path bellow
    path = "Data/Results_from_MOCASSIN/outputs_000/output_0001/"
    list_plot=list(range(0, 10))
    txt = path + 'SED.out'
    df_read_wl = pd.read_pickle(path + "df_z.pkl")
    df_read_wl = df_read_wl[df_read_wl["lambda_um"] < 30.5] ###We only plot out the models up to ~30 micron, one can extend this range
    plot_sample(data,flux, df_read_wl, otherfeatures_data, path_filterbands,
                path_preprocessing, name_, title, z_list_withoutDrop, z_upper_limit, list_plot)
    # # exit()


    ### TODO: define the name of transmission curves that you downloaded from SVO and uploaded in Data>bands director
    ###read transmission curve for NIRCam and MIRI
    list_filters_NIRCam=["JWST_NIRCam.F070W.dat", "JWST_NIRCam.F090W.dat", "JWST_NIRCam.F115W.dat",
                  "JWST_NIRCam.F140M.dat", "JWST_NIRCam.F150W.dat",
                  "JWST_NIRCam.F162M.dat",
                  "JWST_NIRCam.F164N.dat", "JWST_NIRCam.F182M.dat", "JWST_NIRCam.F187N.dat",
                  "JWST_NIRCam.F200W.dat", "JWST_NIRCam.F210M.dat", "JWST_NIRCam.F212N.dat",
                  "JWST_NIRCam.F250M.dat", "JWST_NIRCam.F277W.dat", "JWST_NIRCam.F300M.dat",
                  "JWST_NIRCam.F322W2.dat", "JWST_NIRCam.F323N.dat", "JWST_NIRCam.F335M.dat",
                  "JWST_NIRCam.F356W.dat", "JWST_NIRCam.F360M.dat", "JWST_NIRCam.F405N.dat",
                  "JWST_NIRCam.F410M.dat", "JWST_NIRCam.F430M.dat", "JWST_NIRCam.F444W.dat",
                  "JWST_NIRCam.F460M.dat", "JWST_NIRCam.F466N.dat", "JWST_NIRCam.F470N.dat",
                  "JWST_NIRCam.F480M.dat"]
    list_filters_name_NIRCam=["JWST_F070W", "JWST_F090W", "JWST_F115W",
                      "JWST_F140M", "JWST_F150W",
                      "JWST_F162M",
                      "JWST_F164N", "JWST_F182M", "JWST_F187N",
                      "JWST_F200W", "JWST_F210M", "JWST_F212N",
                      "JWST_F250M", "JWST_F277W", "JWST_F300M",
                      "JWST_F322W2", "JWST_F323N", "JWST_F335M",
                      "JWST_F356W", "JWST_F360M", "JWST_F405N",
                      "JWST_F410M", "JWST_F430M", "JWST_F444W",
                      "JWST_F460M", "JWST_F466N", "JWST_F470N",
                      "JWST_F480M"]
    filters_NIRCam_clWL = [0.706, 0.904, 1.157, 1.406,
                           1.504,
                           1.628, 1.645,
                           1.847, 1.874, 1.993, 2.096,
                           2.121, 2.504, 2.769, 2.991,
                           3.267, 3.237, 3.364, 3.577,
                           3.626, 4.052, 4.084, 4.282,
                           4.416, 4.630, 4.654, 4.708,
                           4.819]





    list_filters_MIRI=["JWST_MIRI.F560W.dat", "JWST_MIRI.F770W.dat", "JWST_MIRI.F1000W.dat",
                       "JWST_MIRI.F1130W.dat", "JWST_MIRI.F1280W.dat", "JWST_MIRI.F1500W.dat",
                       "JWST_MIRI.F1800W.dat", "JWST_MIRI.F2100W.dat", "JWST_MIRI.F2550W.dat"]
    list_filters_name_MIRI=["JWST_F560W","JWST_F770W", "JWST_F1000W",
                      "JWST_F1130W", "JWST_F1280W", "JWST_F1500W",
                      "JWST_F1800W", "JWST_F2100W", "JWST_F2550W"]
    filters_MIRI_clWL = [5.6, 7.7, 10, 11.3,
                         12.8, 15, 18, 21, 25.5]





    # # #######NIRCam filters
    set_filters(data,flux,df_read_wl,path_filterbands, z_list_withoutDrop, name_,
                list_filters_NIRCam,list_filters_name_NIRCam, filter_set_name="NIRCam") ### NIRCam filters

    # # ##########MIRI filters
    set_filters(data,flux,df_read_wl,path_filterbands, z_list_withoutDrop, name_,
                list_filters_MIRI,list_filters_name_MIRI, filter_set_name='MIRI') ### MIRI filters

    list_wls=[filters_NIRCam_clWL , filters_MIRI_clWL]
    filter_sets_name=["NIRCam", "MIRI"]
    plot_sample_bands(data,flux, df_read_wl, otherfeatures_data, path_filterbands,
                      path_preprocessing, name_, title, z_list_withoutDrop,
                      z_upper_limit, list_plot,filter_sets_name , list_wls)

    ###concat the two data sets with two filter sets
    df_bands_1= pd.read_pickle(str(path_filterbands)+"/logBandsFlux-NIRCam_"+str(name_)+".pkl") #magnitudes form NIRCam filters
    df_bands_2= pd.read_pickle(str(path_filterbands)+"/logBandsFlux-MIRI_"+str(name_)+".pkl") #magnitudes form MIRI filters
    df_mags_1 = ((-2.5 * (df_bands_1 - 23)) - 48.6)
    df_mags_2 = ((-2.5 * (df_bands_2 - 23)) - 48.6)

    df_mags_1["index"]=df_mags_1.index
    df_mags_2["index"]=df_mags_2.index


    data_mags=df_mags_1.merge( df_mags_2, on="index")
    data_mags=data_mags.drop(columns="index")
    data_mags.columns=range(0,len(data_mags.columns))
    data_mags.to_pickle("Data/All_MOCASSIN"+str(name_)+"/data_mags.pkl")




