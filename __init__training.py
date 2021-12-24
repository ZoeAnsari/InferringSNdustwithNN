import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
# from src.preprocessing import preprocess
from src.training import NN_reg_dense
import pickle
from src.SHAP_calculator import shap_on_NN_reg_loop
from src.evaluation import evaluation_any

####define a scenario
####for S1 -> fixedz
# scenario="fixedz"
####for S2 -> Z
# scenario="Z"
####for S3 -> ZERROR
scenario="ZERROR"




# preprocess(scenario)



##read the preprocessed data
path_preprocessing="Data/All_MOCASSIN_"+str(scenario)
y_data=pd.read_pickle(str(path_preprocessing)+"/y_data.pkl")
otherfeatures=pd.read_pickle(str(path_preprocessing)+"/otherfeatures_data.pkl")
X_mags_cutoff=pd.read_pickle(str(path_preprocessing)+"/MissingLabel_After_sen_cutoff.pkl")


X_data=X_mags_cutoff.rename(columns={0: "F070W_NIR", 1:"F090W_NIR", 2:"F115W_NIR", 3:"F140M_NIR",4:"F150W_NIR",
                                     5:"F162M_NIR", 6:"F164N_NIR", 7:"F182M_NIR", 8:"F187N_NIR", 9:"F200W_NIR",
                                     10:"F210M_NIR", 11:"F212N_NIR", 12:"F250M_NIR", 13:"F277W_NIR", 14:"F300M_NIR",
                                     15:"F322W2_NIR", 16:"F323N_NIR", 17:"F335M_NIR", 18:"F356W_NIR", 19:"F360M_NIR",
                                     20:"F405N_NIR", 21:"F410M_NIR", 22:"F430M_NIR", 23:"F444W_NIR", 24:"F460M_NIR",
                                     25:"F466N_NIR", 26:"F470N_NIR", 27:"F480M_NIR", 28:"F560W_MIRI", 29:"F770W_MIRI",
                                     30:"F1000W_MIRI", 31:"F1130W_MIRI", 32:"F1280W_MIRI", 33:"F1500W_MIRI", 34:"F1800W_MIRI",
                                     35:"F2100W_MIRI",36:"F2550W_MIRI"})



X_data["z"]=otherfeatures["z"]





#####defining hyper parameters for the neural network

n_=1

data_size=len(X_data)
epoch_size=1500
batch_size=64
learning_rate=1e-5




list_cols_modeified=list(X_data.columns)
list_filter_NIRCam_name_modified=["JWST_NIRCAM_F070W","JWST_NIRCAM_F090W","JWST_NIRCAM_F115W",
                                  "JWST_NIRCAM_F140M", "JWST_NIRCAM_F150W", "JWST_NIRCAM_F162M",
                                  "JWST_NIRCAM_F164N", "JWST_NIRCAM_F182M", "JWST_NIRCAM_F187N",
                                  "JWST_NIRCAM_F200W", "JWST_NIRCAM_F210M", "JWST_NIRCAM_F212N",
                                  "JWST_NIRCAM_F250M", "JWST_NIRCAM_F277W", "JWST_NIRCAM_F300M",
                                  "JWST_NIRCAM_F322W2", "JWST_NIRCAM_F323N", "JWST_NIRCAM_F335M",
                                  "JWST_NIRCAM_F356W", "JWST_NIRCAM_F360M", "JWST_NIRCAM_F405N",
                                  "JWST_NIRCAM_F410M","JWST_NIRCAM_F430M", "JWST_NIRCAM_F444W",
                                  "JWST_NIRCAM_F460M", "JWST_NIRCAM_F466N", "JWST_NIRCAM_F470N",
                                  "JWST_NIRCAM_F480M"]
list_NIRCam_filters_modified=["JWST_NIRCam.F070W.dat","JWST_NIRCam.F090W.dat","JWST_NIRCam.F115W.dat",
                              "JWST_NIRCam.140M.dat","JWST_NIRCam.F150W.dat", "JWST_NIRCam.F162M.dat",
                              "JWST_NIRCam.F164N.dat", "JWST_NIRCam.F182M.dat", "JWST_NIRCam.F187N.dat",
                              "JWST_NIRCam.F200W.dat", "JWST_NIRCam.F210M.dat", "JWST_NIRCam.F212N.dat",
                              "JWST_NIRCam.F250M.dat", "JWST_NIRCam.F277W.dat", "JWST_NIRCam.F300M.dat",
                              "JWST_NIRCam.F322W2.dat", "JWST_NIRCam.F323N.dat", "JWST_NIRCam.F335M.dat",
                              "JWST_NIRCam.F356W.dat", "JWST_NIRCam.F360M.dat", "JWST_NIRCam.F405N.dat",
                              "JWST_NIRCam.F410M.dat", "JWST_NIRCam.F430M.dat", "JWST_NIRCam.F444W.dat",
                              "JWST_NIRCam.F460M.dat", "JWST_NIRCam.F466N.dat", "JWST_NIRCam.F470N.dat",
                              "JWST_NIRCam.F480M.dat"]



list_filter_MIRI_name_modified=["JWST_MIRI_F560W", "JWST_MIRI_F770W","JWST_MIRI_F1000W",
                                "JWST_MIRI_F1130W","JWST_MIRI_F1280W","JWST_MIRI_F1500W",
                                "JWST_MIRI_F1800W","JWST_MIRI_F2100W", "JWST_MIRI_F2550W"]

list_MIRI_filters_modified=["JWST_MIRI.F560W.dat", "JWST_MIRI.F770W.dat","JWST_MIRI.F1000W.dat",
                            "JWST_MIRI.F1130W.dat","JWST_MIRI.F1280W.dat","JWST_MIRI.F1500W.dat",
                            "JWST_MIRI.F1800W.dat","JWST_MIRI.F2100W.dat","JWST_MIRI.F2550W.dat"]



# ###########a
df_num_test=[]
df_num_lim_test=[]
df_loss_train=[]
df_loss_val=[]
df_md_frac_lim_test=[]
df_md_bias_lim_test=[]
df_md_rms_lim_test=[]
df_td_frac_lim_test=[]
df_td_bias_lim_test=[]
df_td_rms_lim_test=[]
conf_lim_test_low=[]
conf_lim_test_c=[]
conf_lim_test_m=[]
conf_lim_test_s=[]
df_md_frac_test=[]
df_md_bias_test=[]
df_md_rms_test=[]
df_td_frac_test=[]
df_td_bias_test=[]
df_td_rms_test=[]
conf_test_c=[]
conf_test_m=[]
conf_test_s=[]
counter_c_all=[]
counter_m_all=[]
counter_s_all=[]
counter_c_lim=[]
counter_m_lim=[]
counter_s_lim=[]
counter_c_test=[]
counter_m_test=[]
counter_s_test=[]
counter_c_lim_per=[]
counter_m_lim_per=[]
counter_s_lim_per=[]
df_num_lim_test_dustTemp=[]
df_num_lim_test_spe=[]
df_num_lim_test_dustMass=[]
df_num_test_pre=[]
df_num_test_new=[]
df_num_lim_test_pre=[]
df_num_lim_test_new=[]
df_num_lim_test_pre_dustTemp=[]
df_num_lim_test_new_dustTemp=[]
df_num_lim_test_pre_spe=[]
df_num_lim_test_new_spe=[]
df_num_lim_test_pre_dustMass=[]
df_num_lim_test_new_dustMass=[]




list_shap_loop=list(range(1,13,1))

for i in list_shap_loop:
    print("--------SHAP step:",i)
    seed_rnd_num=30+i
    shap_num=i
    model_num=str(shap_num)+"SHAP"+str(n_)+"compo_"+str(scenario)#+"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)

    ###reading the filters that are not elimniated in the previous step
    if i != 1:
        model_num_mod=str(shap_num-1)+"SHAP"+str(n_)+"compo_"+str(scenario)#+"_RndSeed"+str(seed_rnd_num-1)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)

        file_name = "Data/ML/model" + str(model_num_mod) +"_RndSeed"+str(seed_rnd_num-1)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate) + \
                    "/model/list_cols_modified_shap" + str(shap_num -1 ) + ".pkl"
        open_file = open(file_name, "rb")
        list_cols_modeified=pickle.load(open_file)
        open_file.close()


        file_name = "Data/ML/model" + str(model_num_mod) +"_RndSeed"+str(seed_rnd_num-1)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate) \
                    +"/model/list_filter_NIRCam_name_modified_shap" + str(shap_num-1) + ".pkl"
        open_file = open(file_name, "rb")
        list_filter_NIRCam_name_modified=pickle.load(open_file)
        open_file.close()



        file_name = "Data/ML/model" + str(model_num_mod) +"_RndSeed"+str(seed_rnd_num-1)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate) \
                    +"/model/list_NIRCam_filters_modified_shap" + str(shap_num-1) + ".pkl"
        open_file = open(file_name, "rb")
        list_NIRCam_filters_modified=pickle.load(open_file)
        open_file.close()




        file_name = "Data/ML/model" + str(model_num_mod) +"_RndSeed"+str(seed_rnd_num-1)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate) \
                    +"/model/list_filter_MIRI_name_modified_shap" + str(shap_num-1) + ".pkl"
        open_file = open(file_name, "rb")
        list_filter_MIRI_name_modified=pickle.load(open_file)
        open_file.close()






        file_name = "Data/ML/model" + str(model_num_mod) +"_RndSeed"+str(seed_rnd_num-1)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate) \
                    +"/model/list_MIRI_filters_modified_shap" + str(shap_num-1) + ".pkl"
        open_file = open(file_name, "rb")
        list_MIRI_filters_modified=pickle.load(open_file)
        open_file.close()





    X_data=X_data[list_cols_modeified]
    # print(X_data_reg_cluster_.columns)

    ###split the dataset

    otherfeatures_train_split, otherfeatures_TEST, \
    X_train_split, X_TEST, y_train_split, y_TEST \
        =train_test_split(otherfeatures,
                          X_data, y_data,test_size=0.15, random_state=seed_rnd_num)

    otherfeatures_train, otherfeatures_val, \
    X_train, X_val, y_train, y_val \
        =train_test_split(otherfeatures_train_split,
                          X_train_split,y_train_split,test_size=0.2, random_state=seed_rnd_num)


    counter_c_test.append(len(otherfeatures_TEST[otherfeatures_TEST["species"]==float(0.5)]))
    counter_m_test.append(len(otherfeatures_TEST[otherfeatures_TEST["species"]==float(0.75)]))
    counter_s_test.append(len(otherfeatures_TEST[otherfeatures_TEST["species"]==int(1)]))


    ##format the input data set into a suitable type for NN

    ####add the indeces as a column
    # X_test_ind=pd.Series(X_val.index)
    # X_all_ind=pd.Series(X_data.index)


    print("new cols",X_train.columns)
    print(len(X_train.columns))


    ##
    # X_all=X_data[list_cols_modeified]
    # y_all=y_data
    # otherfeatures_all = otherfeatures


    NN_reg_dense(n_, X_data, X_train, X_val,
                 y_data, y_train, y_val,
                 otherfeatures, otherfeatures_train, otherfeatures_val,
                 model_num, seed_rnd_num, epoch_size,
                   batch_size,learning_rate, shap_num)

    # exit()
    X_train.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                           + "/MissingLabel_After_sen_cutoff_X_train.pkl")
    y_train.to_pickle("Data/ML/model" + str(model_num)+"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                          + "/MissingLabel_After_sen_cutoff_y_train.pkl")
    X_val.to_pickle("Data/ML/model" + str(model_num)+"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                         + "/MissingLabel_After_sen_cutoff_X_val.pkl")
    y_val.to_pickle("Data/ML/model" + str(model_num)+"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                        + "/MissingLabel_After_sen_cutoff_y_val.pkl")




    X_TEST.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                          + "/MissingLabel_After_sen_cutoff_X_TEST.pkl")
    y_TEST.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                         + "/MissingLabel_After_sen_cutoff_y_TEST.pkl")
    otherfeatures_TEST.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                                          + "/MissingLabel_After_sen_cutoff_otherfeatures_TEST.pkl")


    X_data.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                                  + "/MissingLabel_After_sen_cutoff_X_all.pkl")



    y_data.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                         + "/MissingLabel_After_sen_cutoff_y_all.pkl")

    otherfeatures.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                                     + "/MissingLabel_After_sen_cutoff_otherfeatures_all.pkl")


    ###define the dataset that you want to evaluate the predictions on
    ###test data set -> TEST
    ###training data set -> train
    ###validation data set -> val
    ###The entire data set -> all
    redshift_all=X_TEST['z']
    data_set='TEST'
    saved_dataset='TEST'



    df_num_test_i, df_num_lim_test_i, \
    df_md_frac_lim_test_i, df_md_bias_lim_test_i, df_md_rms_lim_test_i, df_td_frac_lim_test_i, \
    df_td_bias_lim_test_i, df_td_rms_lim_test_i, conf_lim_test_low_i, conf_lim_test_c_i, conf_lim_test_m_i, \
    conf_lim_test_s_i, \
    df_md_frac_test_i, df_md_bias_test_i, df_md_rms_test_i, df_td_frac_test_i, \
    df_td_bias_test_i, df_td_rms_test_i,  conf_test_c_i, conf_test_m_i, \
    conf_test_s_i, \
    counter_c_all_i, counter_m_all_i, counter_s_all_i,counter_c_lim_i, counter_m_lim_i, counter_s_lim_i, \
    df_num_lim_test_dustTemp_i,df_num_lim_test_spe_i,df_num_lim_test_dustMass_i, \
    df__test_pre_i, df__test_new_i, df__lim_test_pre_i, df__lim_test_new_i, df__lim_test_pre_dustTemp_i, \
    df__lim_test_new_dustTemp_i, df__lim_test_pre_spe_i, df__lim_test_new_spe_i, \
    df__lim_test_pre_dustMass_i, df__lim_test_new_dustMass_i \
        =evaluation_any(model_num, seed_rnd_num,
                        epoch_size,
                        batch_size, learning_rate, redshift_all, path_preprocessing,
                        data_set, saved_dataset)




    df_num_test_pre_i=len(df__test_pre_i)
    df_num_test_new_i=len(df__test_new_i)
    df_num_lim_test_pre_i=len(df__lim_test_pre_i)
    df_num_lim_test_new_i=len(df__lim_test_new_i)
    df_num_lim_test_pre_dustTemp_i=len(df__lim_test_pre_dustTemp_i)
    df_num_lim_test_new_dustTemp_i=len(df__lim_test_new_dustTemp_i)
    df_num_lim_test_pre_spe_i=len(df__lim_test_pre_spe_i)
    df_num_lim_test_new_spe_i=len(df__lim_test_new_spe_i)
    df_num_lim_test_pre_dustMass_i=len(df__lim_test_pre_dustMass_i)
    df_num_lim_test_new_dustMass_i=len(df__lim_test_new_dustMass_i)

    df_num_test_pre.append(df_num_test_pre_i)
    df_num_test_new.append(df_num_test_new_i)
    df_num_lim_test_pre.append(df_num_lim_test_pre_i)
    df_num_lim_test_new.append(df_num_lim_test_new_i)
    df_num_lim_test_pre_dustTemp.append(df_num_lim_test_pre_dustTemp_i)
    df_num_lim_test_new_dustTemp.append(df_num_lim_test_new_dustTemp_i)
    df_num_lim_test_pre_spe.append(df_num_lim_test_pre_spe_i)
    df_num_lim_test_new_spe.append(df_num_lim_test_new_spe_i)
    df_num_lim_test_pre_dustMass.append(df_num_lim_test_pre_dustMass_i)
    df_num_lim_test_new_dustMass.append(df_num_lim_test_new_dustMass_i)


    df_name_list=["df__test_pre_i", "df__test_new_i", "df__lim_test_pre_i", "df__lim_test_new_i", "df__lim_test_pre_dustTemp_i",
                  "df__lim_test_new_dustTemp_i", "df__lim_test_pre_spe_i", "df__lim_test_new_spe_i",
                  "df__lim_test_pre_dustMass_i", "df__lim_test_new_dustMass_i"]
    df_list=[df__test_pre_i, df__test_new_i, df__lim_test_pre_i, df__lim_test_new_i, df__lim_test_pre_dustTemp_i,
             df__lim_test_new_dustTemp_i, df__lim_test_pre_spe_i, df__lim_test_new_spe_i,
             df__lim_test_pre_dustMass_i, df__lim_test_new_dustMass_i]
    counter_df=0
    for i_df in df_list:
        df_name=df_name_list[counter_df]

        i_df.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                       + "/NeatPlot_"+str(saved_dataset)+"/"+str(df_name)+".pkl")

        counter_df=counter_df+1


    df_num_test.append(df_num_test_i)
    df_num_lim_test.append(df_num_lim_test_i)
    # df_loss_train.append(df_loss_train_i)
    # df_loss_val.append(df_loss_val_i)
    df_md_frac_lim_test.append(df_md_frac_lim_test_i)
    df_md_bias_lim_test.append(df_md_bias_lim_test_i)
    df_md_rms_lim_test.append(df_md_rms_lim_test_i)
    df_td_frac_lim_test.append(df_td_frac_lim_test_i)
    df_td_bias_lim_test.append(df_td_bias_lim_test_i)
    df_td_rms_lim_test.append(df_td_rms_lim_test_i)
    conf_lim_test_low.append(conf_lim_test_low_i)
    conf_lim_test_c.append(conf_lim_test_c_i)
    conf_lim_test_m.append(conf_lim_test_m_i)
    conf_lim_test_s.append(conf_lim_test_s_i)


    df_md_frac_test.append(df_md_frac_test_i)
    df_md_bias_test.append(df_md_bias_test_i)
    df_md_rms_test.append(df_md_rms_test_i)
    df_td_frac_test.append(df_td_frac_test_i)
    df_td_bias_test.append(df_td_bias_test_i)
    df_td_rms_test.append(df_td_rms_test_i)
    conf_test_c.append(conf_test_c_i)
    conf_test_m.append(conf_test_m_i)
    conf_test_s.append(conf_test_s_i)
    counter_c_all.append(counter_c_all_i)
    counter_m_all.append(counter_m_all_i)
    counter_s_all.append(counter_s_all_i)
    counter_c_lim.append(counter_c_lim_i)
    counter_m_lim.append(counter_m_lim_i)
    counter_s_lim.append(counter_s_lim_i)


    counter_c_lim_per.append(counter_c_lim_i/counter_c_all_i)
    counter_m_lim_per.append(counter_m_lim_i/counter_m_all_i)
    counter_s_lim_per.append(counter_s_lim_i/counter_s_all_i)

    df_num_lim_test_dustTemp.append(df_num_lim_test_dustTemp_i)
    df_num_lim_test_spe.append(df_num_lim_test_spe_i)
    df_num_lim_test_dustMass.append(df_num_lim_test_dustMass_i)
    # break
    # exit()




    shap_on_NN_reg_loop(X_data,y_data,otherfeatures,
                        model_num, seed_rnd_num,
                        epoch_size,
                        batch_size, learning_rate, shap_num,
                        otherfeatures_train, otherfeatures_val,
                        X_train, X_val, y_train, y_val,
                        list_filter_NIRCam_name_modified ,
                        list_NIRCam_filters_modified,
                        list_filter_MIRI_name_modified ,
                        list_MIRI_filters_modified, list_cols_modeified)




    #
    #
    #
    #
    # file_name = "Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)\
    #             +"/model/list_filter_NIRCam_name_modified_shap" + str(shap_num) + ".pkl"
    # open_file = open(file_name, "rb")
    # list_filter_NIRCam_name_modified=pickle.load(open_file)
    # open_file.close()
    #
    #
    #
    # file_name = "Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)\
    #             +"/model/list_NIRCam_filters_modified_shap" + str(shap_num) + ".pkl"
    # open_file = open(file_name, "rb")
    # list_NIRCam_filters_modified=pickle.load(open_file)
    # open_file.close()
    #
    #
    #
    #
    # file_name = "Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)\
    #             +"/model/list_filter_MIRI_name_modified_shap" + str(shap_num) + ".pkl"
    # open_file = open(file_name, "rb")
    # list_filter_MIRI_name_modified=pickle.load(open_file)
    # open_file.close()
    #
    #
    #
    #
    #
    #
    # file_name = "Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)\
    #             +"/model/list_MIRI_filters_modified_shap" + str(shap_num) + ".pkl"
    # open_file = open(file_name, "rb")
    # list_MIRI_filters_modified=pickle.load(open_file)
    # open_file.close()
    #
    #
    # file_name = "Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)\
    #             +"/model/list_cols_modified_shap" + str(shap_num) + ".pkl"
    # open_file = open(file_name, "rb")
    # list_cols_modeified=pickle.load(open_file)
    # open_file.close()


file_list= [df_num_test, df_num_lim_test, df_loss_train, df_loss_val,
            df_md_frac_lim_test, df_md_bias_lim_test, df_md_rms_lim_test, df_td_frac_lim_test,
            df_td_bias_lim_test, df_td_rms_lim_test, conf_lim_test_low, conf_lim_test_c, conf_lim_test_m,
            conf_lim_test_s,
            df_md_frac_test, df_md_bias_test, df_md_rms_test, df_td_frac_test,
            df_td_bias_test, df_td_rms_test,  conf_test_c, conf_test_m,
            conf_test_s,
            counter_c_all, counter_m_all, counter_s_all, counter_c_lim, counter_m_lim,  counter_s_lim,
            counter_c_test, counter_m_test, counter_s_test,
            counter_c_lim_per,counter_m_lim_per,counter_s_lim_per,
            df_num_lim_test_dustTemp, df_num_lim_test_spe, df_num_lim_test_dustMass,
            df_num_test_pre, df_num_test_new, df_num_lim_test_pre, df_num_lim_test_new,
            df_num_lim_test_pre_dustTemp, df_num_lim_test_new_dustTemp, df_num_lim_test_pre_spe,
            df_num_lim_test_new_spe, df_num_lim_test_pre_dustMass, df_num_lim_test_new_dustMass]

file_list_name= ["df_num_test", "df_num_lim_test", "df_loss_train", "df_loss_val",
                 "df_md_frac_lim_test", "df_md_bias_lim_test", "df_md_rms_lim_test", "df_td_frac_lim_test",
                 "df_td_bias_lim_test", "df_td_rms_lim_test", "conf_lim_test_low", "conf_lim_test_c", "conf_lim_test_m",
                 "conf_lim_test_s",
                 "df_md_frac_test", "df_md_bias_test", "df_md_rms_test", "df_td_frac_test",
                 "df_td_bias_test", "df_td_rms_test", "conf_test_c", "conf_test_m",
                 "conf_test_s",
                 "counter_c_all","counter_m_all", "counter_s_all", "counter_c_lim", "counter_m_lim",  "counter_s_lim",
                 "counter_c_test", "counter_m_test", "counter_s_test",
                 "counter_c_lim_per","counter_m_lim_per","counter_s_lim_per",
                 "df_num_lim_test_dustTemp", "df_num_lim_test_spe", "df_num_lim_test_dustMass",
                 "df_num_test_pre", "df_num_test_new", "df_num_lim_test_pre", "df_num_lim_test_new",
                 "df_num_lim_test_pre_dustTemp", "df_num_lim_test_new_dustTemp", "df_num_lim_test_pre_spe",
                 "df_num_lim_test_new_spe", "df_num_lim_test_pre_dustMass", "df_num_lim_test_new_dustMass"]



file_i_list = []
counter=0
for file_i in file_list:
    print("file_i",file_i)
    file_i_name=file_list_name[counter]
    with open("Data/ML/"+str(file_i_name)+"_"+str(scenario)+".txt", "w") as f:
        for s in file_i:
            f.write(str(s) +"\n")
            print("f---",f)

    with open("Data/ML/"+str(file_i_name)+"_"+str(scenario)+".txt", "r") as f:
        for line in f:
            file_i_list.append(float(line.strip()))
    counter=counter+1
