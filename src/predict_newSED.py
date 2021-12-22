from tensorflow import keras
import pandas as pd
import numpy as np
import mdn

def predict_SNdust(X,trainedmodel_path, epoch, n_, scenario_indicator, rnd_seed, batch_size, learning_rate):

    N_MIXES = n_  # (n_components)  # number of mixture components
    OUTPUT_DIMS = 3  # number of real-values predicted by each mixture component
    model = keras.models.load_model(str(trainedmodel_path) + "/model/"
                                                             "Dust_NN_"+str(rnd_seed -30)+".h5", custom_objects={'MDN': mdn.MDN,
                                                            'mdn_loss_func': mdn.get_mixture_loss_func(OUTPUT_DIMS, N_MIXES)})


    ###predict the target values from the trained model
    y_pred = model.predict(X)

    # Split up the mixture parameters (for future fun)

    prediction_ = np.apply_along_axis((lambda a: a[:OUTPUT_DIMS]), 1, y_pred)  # the means
    sigs_ = np.apply_along_axis((lambda a: a[OUTPUT_DIMS:2 * OUTPUT_DIMS]), 1,
                                y_pred)  # the standard deviations
    pis_ = np.apply_along_axis((lambda a: mdn.softmax(a[2 * OUTPUT_DIMS:])), 1,
                               y_pred)  # the mixture components


    ####saving outcome
    prediction_=pd.DataFrame(prediction_)
    sigs_=pd.DataFrame(sigs_)
    pis_=pd.DataFrame(pis_)

    print("prediction----------", prediction_)


    ######rescale the target values
    result = pd.DataFrame( 1e-1 *prediction_[0])
    result=result.rename(columns={0: "pred_dustMass"})
    result["ML_uncertanity_dustMass"] = sigs_[0]

    result["pred_dustTemp"]=((prediction_[1]) * 2200)
    result["ML_uncertanity_dustTemp"] = sigs_[1]

    result["pred_spe"] = prediction_[2]
    result["ML_uncertanity_spe"] = sigs_[2]

    result["ML_prob"] = pis_[0]




    print("------------------------------saving the result--------------------------------------------------------------")

    result.to_pickle("Data/Prediction/predicted_with_model_" + str(scenario_indicator) + "_RndSeed" + str(rnd_seed)+"_"+str(epoch) +
                     "epoch_" + str(batch_size) + "_LR" + str(learning_rate) +
                     ".pkl")
    #_ZERROR_RndSeed31_1500epoch_64_LR1e-05

    #####implementing sigma_pred criteria

    coef_k=a1=1 #a1
    coef=a2=0.2 #a2


    result["pred_dustMass_noTransformed"]=  1e+1 * result["pred_dustMass"]
    result["pred_dustTemp_noTransformed"]= (result['pred_dustTemp'])/2200
    result["pred_spe_noTransformed"]=result["pred_spe"]


    result["ML_uncertanity_rescaled_dustMass"]=  1e-1 * result["ML_uncertanity_dustMass"]
    result["ML_uncertanity_rescaled_dustTemp"]= (result["ML_uncertanity_dustTemp"]) * 2200



    ###save the table of predictions

    df=result
    pro_list=['dustMass','dustTemp','spe']
    pro_label_list=["Md", "Td", "species"]
    df=result
    df["predicted_Md"]=df["pred_dustMass_noTransformed"]
    df["predicted_sigma_Md"]=df["ML_uncertanity_rescaled_dustMass"]
    df["predicted_Td"]=df["pred_dustTemp_noTransformed"]
    df["predicted_sigma_Td"]=df["ML_uncertanity_rescaled_dustTemp"]
    df["predicted_species"]=df["pred_spe_noTransformed"]
    df["predicted_sigma_species"]=df["ML_uncertanity_spe"]




    df["predicted_Md(Msun)"]=df["pred_dustMass"]
    df["predicted_sigma_Md(Msun)"]=df["ML_uncertanity_rescaled_dustMass"]
    df["reliable_predicted_Md"]=df["pred_dustMass"]

    df["predicted_Td(K)"]=df["pred_dustTemp"]
    df["predicted_sigma_Td(K)"]=df["ML_uncertanity_rescaled_dustTemp"]
    df["reliable_predicted_Td"]=df["pred_dustTemp"]

    df["predicted_species"]=df["pred_spe"]
    df["predicted_sigma_species"]=df["ML_uncertanity_spe"]
    df["reliable_predicted_species"]=df["pred_spe"]


    df["Md_pred_cond"]= (a2*a1)* df["predicted_Md(Msun)"]
    df["Td_pred_cond"]= (a2*a1)* df["predicted_Td(K)"]
    df["species_pred_cond"]= (a2*a1)


    for ind in df.index:

        if df["pred_spe"][ind] <= float(0.25):
            df["predicted_species"][ind]="noDust"#append(float(1))
        elif df["pred_spe"][ind] <= float(0.675):
            df["predicted_species"][ind]="Carbon"#append(float(1))
        elif df["pred_spe"][ind] <= float(0.875):
            df["predicted_species"][ind]="Mixed"#append(float(2))
        elif df["pred_spe"][ind] > float(0.875):
            df["predicted_species"][ind]="Silicate"


        if df["Md_pred_cond"][ind] > df["predicted_sigma_Md(Msun)"][ind] :
            df["reliable_predicted_Md"][ind]="Yes"
        elif df["Md_pred_cond"][ind] <= df["predicted_sigma_Md(Msun)"][ind]:
            df["reliable_predicted_Md"][ind]="No"


        if df["Td_pred_cond"][ind] > df["predicted_sigma_Td(K)"][ind] :
            df["reliable_predicted_Td"][ind]="Yes"
        elif df["Td_pred_cond"][ind] <= df["predicted_sigma_Td(K)"][ind]:
            df["reliable_predicted_Td"][ind]="No"

        if df["species_pred_cond"][ind] > df["predicted_sigma_species"][ind] :
            df["reliable_predicted_species"][ind]="Yes"
        elif df["species_pred_cond"][ind] <= df["predicted_sigma_species"][ind]:
            df["reliable_predicted_species"][ind]="No"








    df=df[["predicted_Md(Msun)","predicted_sigma_Md(Msun)","reliable_predicted_Md",
           "predicted_Td(K)","predicted_sigma_Td(K)","reliable_predicted_Td",
           "predicted_species","predicted_sigma_species","reliable_predicted_species"]]
    df.to_csv("Data/Prediction/Prediction_table.csv")
    print("Predictions are saved in Data/Prediction/Prediction_table.csv")
    print("First line of predictions: ",df.iloc[0])