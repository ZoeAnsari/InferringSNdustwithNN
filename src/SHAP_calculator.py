import shap
import tensorflow.keras.models as models
import mdn
import tensorflow.keras.backend
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time as time
import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split
tf.disable_v2_behavior()
tf.reset_default_graph()



# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4,5,6"


def shap_on_NN_reg_loop(X,y, otherfeatures,
                        model_num, seed_rnd_num,
                        epoch_size,
                        batch_size, learning_rate, shap_num, otherfeatures_train, otherfeatures_test,
                        X_train, X_test, y_train, y_test, list_filter_NIRCam_name_modified ,list_NIRCam_filters_modified,
                        list_filter_MIRI_name_modified ,list_MIRI_filters_modified, cols):


    X_df=X
    OUTPUT_DIMS=3
    N_MIXES=1

    model =models.load_model("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                             +"/model/Dust_NN_"+str(shap_num)+".h5"
                             , custom_objects={'MDN': mdn.MDN,
                                               'mdn_loss_func': mdn.get_mixture_loss_func(OUTPUT_DIMS,N_MIXES)})

    print("after loading model")
    y_pred=model.predict(X)

    X_train.index=np.array(list(range(len(X_train))))
    X_test.index=np.array(list(range(len(X_test))))



    X_train = np.array(X_train)
    X_test = np.array(X_test)
    st = time.time()


    ### todo: change 5000 to a desired number of data points you would like to calculate the expected values for
    train=X_train[np.random.choice(X_train.shape[0], 5000)]
    explainer = shap.DeepExplainer(model, train)
    df_train=pd.DataFrame(train)
    df_train.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                       +"/model/shap_train"+str(shap_num)+".pkl")


    ### todo: change 1000 to a desired number of data points you would like to calculate the shapley values for
    val=X_test[np.random.choice(X_test.shape[0], 1000)]
    df_val=pd.DataFrame(val)
    df_val.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                     +"/model/val_train"+str(shap_num)+".pkl")
    shap_values = explainer.shap_values(val)
    ed=time.time()
    print("time for shap calculation:", str(ed- st))


    file_name="Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)\
              +"/model/shap"+str(shap_num)+".pkl"
    open_file = open(file_name, "wb")
    pickle.dump(shap_values, open_file)
    open_file.close()




    #shap values0: mu_mass
    # shap values3: sig_mass

    # shap values1: mu_temp
    # shap values4: sig_temp

    # shap values2: mu_spe
    # shap values5: sig_spe

    #shap values6: p=1


    num_features = len(cols)


    values_j=[]
    list_mus=list(range(6))
    for j in range(num_features):
        values_i = 0
        for i in list_mus:
            values_i += np.abs(shap_values[i][0][j]).mean()
        values_j.append(values_i)

    values=values_j

    file_name = "Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)\
                + "/model/values" + str(shap_num) + ".pkl"
    open_file = open(file_name, "wb")
    pickle.dump(values, open_file)
    open_file.close()

    open_file = open(file_name, "rb")
    values = pickle.load(open_file)
    open_file.close()

    df = pd.read_csv("Data/table.csv")


    #####NIRCam bands
    list_NIRCam_filters = list_NIRCam_filters_modified
    list_filter_NIRCam_name = list_filter_NIRCam_name_modified



    #######-------MIRI bands
    list_MIRI_filters = list_MIRI_filters_modified


    list_filter_MIRI_name = list_filter_MIRI_name_modified



    filter_names=list_filter_NIRCam_name + list_filter_MIRI_name + ["z"]
    filters=list_NIRCam_filters + list_MIRI_filters + ["z"]


    shap_df = {}
    shap_df["it"] = []
    shap_df["filter_names"] = []
    shap_df["filters"]=[]
    shap_df["cols"]=[]

    for i_n  in range(len(values)  ):

        shap_df["it"].append(values[i_n])
        shap_df["filter_names"].append(filter_names[i_n])
        shap_df["filters"].append(filters[i_n])
        shap_df["cols"].append(cols[i_n])


    shap_df=pd.DataFrame.from_dict(shap_df)
    shap_df.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                      + "/model/sf_shap" + str(shap_num) + ".pkl")


    shap_df = shap_df.iloc[:-1 , :]
    min_index = np.abs(shap_df["it"]).idxmin()

    plt.figure(figsize=(10,6))
    plt.title("SHAP values")
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.1, bottom=0.22, right=0.96, top=0.96)
    sns.barplot(shap_df["filter_names"], shap_df["it"])
    plt.grid()
    plt.xlabel("photometric bands")
    plt.xticks(rotation=90, fontsize=10)
    plt.ylabel("mean(|SHAP value|) (average impact on model output magnitude)")
    plt.savefig("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                + "/model/shap_bar"+str(shap_num)+".png")

    file_name_NIRCam=[]
    file_name_MIRI=[]

    filter_name_file_NIRCam=[]
    filter_name_file_MIRI=[]
    cols_fil=[]
    for shap_i in range(3):
        min_index = np.abs(shap_df["it"]).idxmin()

        fil_name = shap_df["filter_names"][min_index]
        filter = shap_df["filters"][min_index]
        col= shap_df["cols"][min_index]
        shap_df=shap_df.drop(index=min_index)
        cols_fil.append(col)

        if fil_name in list_filter_NIRCam_name:
            file_name_NIRCam.append(fil_name)
            filter_name_file_NIRCam.append(filter)


        elif fil_name in list_filter_MIRI_name:
            file_name_MIRI.append(fil_name)
            filter_name_file_MIRI.append(filter)

    list_filter_NIRCam_name_modified_tosave= [x for x in list_filter_NIRCam_name if x not in file_name_NIRCam]

    file_name = "Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)\
                + "/model/list_filter_NIRCam_name_modified_shap" + str(shap_num) + ".pkl"
    open_file = open(file_name, "wb")
    pickle.dump(list_filter_NIRCam_name_modified_tosave, open_file)
    open_file.close()


    list_NIRCam_filters_modified_tosave=[x for x in list_NIRCam_filters if x not in filter_name_file_NIRCam]
    file_name = "Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)\
                + "/model/list_NIRCam_filters_modified_shap" + str(shap_num) + ".pkl"
    open_file = open(file_name, "wb")
    pickle.dump(list_NIRCam_filters_modified_tosave, open_file)
    open_file.close()



    list_filter_MIRI_name_modified_tosave=[x for x in list_filter_MIRI_name if x not in file_name_MIRI]
    file_name = "Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)\
                + "/model/list_filter_MIRI_name_modified_shap" + str(shap_num) + ".pkl"
    open_file = open(file_name, "wb")
    pickle.dump(list_filter_MIRI_name_modified_tosave, open_file)
    open_file.close()





    list_MIRI_filters_modified_tosave=[x for x in list_MIRI_filters if x not in filter_name_file_MIRI]
    file_name = "Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)\
                + "/model/list_MIRI_filters_modified_shap" + str(shap_num) + ".pkl"
    open_file = open(file_name, "wb")
    pickle.dump(list_MIRI_filters_modified_tosave, open_file)
    open_file.close()



    list_cols_modified_tosave=[x for x in cols if x not in cols_fil]
    file_name = "Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)\
                + "/model/list_cols_modified_shap" + str(shap_num) + ".pkl"
    open_file = open(file_name, "wb")
    pickle.dump(list_cols_modified_tosave, open_file)
    open_file.close()


    ed=time.time()
    print("time for shap calculation:", str(ed- st))

