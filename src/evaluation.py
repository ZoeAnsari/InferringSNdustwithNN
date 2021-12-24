import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
import matplotlib.cm as cm
from matplotlib import colors




def eval_measure(result, predicted_feat, model_num, seed_rnd_num,
                 epoch_size,
                 batch_size, learning_rate, data_set, name_):

    result["pred_" + str(predicted_feat)]=result["pred_" + str(predicted_feat)]
    result["act_" + str(predicted_feat)]=result["act_" + str(predicted_feat)]



    if predicted_feat  == 'dustMass':
        delta = (np.log10(result["pred_" + str(predicted_feat)]) - np.log10(result["act_" + str(predicted_feat)]))
    else:
        delta = (result["pred_" + str(predicted_feat)] - result["act_" + str(predicted_feat)])

    delta.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                    + "/NeatPlot_"+str(data_set)+"/delta"+str(predicted_feat)+"_"+str(name_)+".pkl")



    data_eval_2 = np.sqrt(np.mean(np.power(delta, 2)))
    data_eval_2_3 = 3 * data_eval_2

    out_3sig = 0
    in_3sig = 0
    df_misszs = {}
    df_misszs["index"] = []
    delta_i_sum = []

    if len(result) != 0:
        for i_dat in result.index:
            if predicted_feat  == 'dustMass':
                delta_i = ((result["pred_" + str(predicted_feat)][i_dat]) - (result["act_" + str(predicted_feat)][i_dat]))/(result["act_" + str(predicted_feat)][i_dat])
            else:
                delta_i=(result["pred_" + str(predicted_feat)][i_dat] - result["act_" + str(predicted_feat)][i_dat])

            delta_i_2=np.abs(delta_i)
            if delta_i_2 <= data_eval_2_3:
                delta_i_sum.append(delta_i)
                in_3sig = in_3sig + 1
            else:
                out_3sig = out_3sig + 1
                df_misszs["index"].append(i_dat)

        frac_out = (out_3sig / len(result)) * 100


        delta_eval = np.mean(delta_i_sum)
        delta_eval_2 = np.sqrt(np.mean(np.power(delta_i_sum, 2)))
    else:
        # print("It is in else")
        frac_out= 1000
        delta_eval = 1000
        delta_eval_2 = 1000



    out_file = open("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                    + "/NeatPlot_"+str(data_set)+"/_outliers_frac"+str(predicted_feat)+"_"+str(name_)+str(len(result))+".txt", "w")


    out_file.write("frac percentage:%s\n" % frac_out)
    out_file.write("mean delta:%s\n" % delta_eval)
    out_file.write("rms delta:%s\n" % delta_eval_2)
    return frac_out, delta_eval, delta_eval_2

def conf_plot(result, predicted_feat,model_num, seed_rnd_num,
              epoch_size,
              batch_size, learning_rate, name, data_set):


    input_file = open("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate) +
                      "/loss_eval.txt", "r")

    loss_info = input_file.read()
    input_file.close()
    y_pred_species = []
    y_true_species = []
    counter_c=0
    counter_m=0
    counter_s=0

    for i_d in result.index:
        if result["act_" + str(predicted_feat)][i_d] == float(0.5) :
            y_true_species.append("Carbon")
            counter_c=counter_c+1
        elif result["act_" + str(predicted_feat)][i_d] == int(1) :
            y_true_species.append("Silicate")
            counter_s=counter_s+1
        elif result["act_" + str(predicted_feat)][i_d] == float(0.75) :
            y_true_species.append("Mixed")
            counter_m=counter_m+1
        elif result["act_" + str(predicted_feat)][i_d] == float(-0.5) :
            y_true_species.append("lowDust")


    for i_d in result.index:
        if result["pred_"+str(predicted_feat)][i_d] <= float(0.25):
            y_pred_species.append("lowDust")
        elif result["pred_"+str(predicted_feat)][i_d] <= float(0.675):
            y_pred_species.append("Carbon")
        elif result["pred_"+str(predicted_feat)][i_d] <= float(0.875):
            y_pred_species.append("Mixed")
        elif result["pred_"+str(predicted_feat)][i_d] > float(0.875):
            y_pred_species.append("Silicate")

    if len(y_pred_species) != 0:

        plt.figure(figsize=(5,5))
        conf=confusion_matrix(y_true_species, y_pred_species, labels=["Carbon", "Mixed","Silicate"])
        conf=pd.DataFrame(conf)
        conf["sum"]=conf.sum(axis=0)
        conf = conf.iloc[:, :].div(conf["sum"])
        conf=conf.drop(columns={"sum"})

        conf_lim_test_c_i= conf[0][0]
        conf_lim_test_m_i=conf[1][1]
        conf_lim_test_s_i=conf[2][2]



        sns.set(font_scale=1.5)
        ax = sns.heatmap(conf, linewidth=0.5, annot=True, vmin=0, vmax=1,cmap=cm.gist_heat_r,#copper_r,#, cmap=cm.viridis_r #
                         xticklabels=["Carbon", "Mixed","Silicate"], yticklabels=["Carbon", "Mixed","Silicate"]
                         )
        ax.set_xticks([0.5, 1.5, 2.5])
        ax.set_yticks([0.6, 1.6, 2.6])
        plt.ylim(0,3)
        plt.xlabel('Simulated class', size=18)
        plt.ylabel('Predicted class', size=18)

        plt.savefig("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                    + "/NeatPlot_"+str(data_set)+"/"+str(name)+str(len(result))+"_Conf_matrix.png",
                    bbox_inches='tight')
    else:
        conf_lim_test_c_i=0
        conf_lim_test_m_i=0
        conf_lim_test_s_i=0
    #####predicted dust temo for nodust with wrong predicted duttemp
    result["pred_spe"]=y_pred_species
    result["act_spe"]=y_true_species




    return conf_lim_test_c_i, conf_lim_test_m_i, conf_lim_test_s_i, counter_c, counter_m, counter_s





def plot_pred_act__uncertanity(result, predicted_feat,model_num, seed_rnd_num,
                               epoch_size,
                               batch_size, learning_rate, x_label, y_label, name, data_set, coef):


    ######-----
    input_file = open("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate) +
                      "/loss_eval.txt", "r")

    loss_info = input_file.read()
    input_file.close()



    ######-----


    plt.figure(figsize=(10, 8))
    plt.plot([0, np.max(result["act_"+str(predicted_feat)])], [0, np.max(result["act_"+str(predicted_feat)])], ls="--", c=".3", linewidth=0.5)


    if predicted_feat == 'dustMass' :
        plt.errorbar((result["act_"+str(predicted_feat)]), (result["pred_"+str(predicted_feat)]),
                     yerr=( result["ML_uncertanity_"+str(predicted_feat)]) , fmt='o')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(str(loss_info) + "for " + str(len(result)) + " objects")
        plt.savefig(
            "Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
            + "/NeatPlot_"+str(data_set)+"/Scatter/" + str(name) + "test_prediction_" + str(
                predicted_feat) + "_ML_uncertanity.png")



    elif predicted_feat == 'dustTemp':
        plt.errorbar((result["act_" + str(predicted_feat)]), (result["pred_" + str(predicted_feat)]),
                     yerr=(result["ML_uncertanity_" + str(predicted_feat)])  , fmt='o')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(str(loss_info) +"for " + str(len(result)) + " objects")
        plt.savefig(
            "Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)+
        "/NeatPlot_"+str(data_set)+"/Scatter/"+str(name)+"test_prediction_"+str(predicted_feat)+"_ML_uncertanity.png")



    plt.figure(figsize=(10, 8))
    plt.plot([0, np.max(result["act_" + str(predicted_feat)])], [0, np.max(result["act_" + str(predicted_feat)])],
             ls="--", c=".3", linewidth=0.5)





    if predicted_feat == 'dustMass' :
        resultlim = result[( result["ML_uncertanity_" + str(predicted_feat)])  <
                           coef * np.mean( result["ML_uncertanity_" + str(predicted_feat)])]



        plt.errorbar((resultlim["act_"+str(predicted_feat)]), (resultlim["pred_"+str(predicted_feat)]),
                     yerr=( resultlim["ML_uncertanity_"+str(predicted_feat)]) , fmt='o')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(str(loss_info) + "for " + str(len(resultlim)) + " objects")
        plt.savefig(
            "Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
            + "/NeatPlot_"+str(data_set)+"/Scatter/" + str(name) + "test_prediction_" + str(
                predicted_feat) + "_ML_uncertanity_limit.png")


    elif predicted_feat == 'dustTemp':

        resultlim = result[ result["ML_uncertanity_" + str(predicted_feat)] <
                            (coef * np.mean(result["ML_uncertanity_" + str(predicted_feat)]))]
        print(len(resultlim))
        plt.errorbar((resultlim["act_" + str(predicted_feat)]), (resultlim["pred_" + str(predicted_feat)]),
                     yerr=( resultlim["ML_uncertanity_" + str(predicted_feat)])  , fmt='o')  # ,


        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(str(loss_info) +"for " + str(len(resultlim)) + " objects")
        plt.savefig(
            "Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
            + "/NeatPlot_"+str(data_set)+"/Scatter/"+str(name)+"test_prediction_"+str(predicted_feat)+"_ML_uncertanity_limit.png")




def plot_pred_act_otherfeatures(result, oth_feat, predicted_feat,model_num, seed_rnd_num,
                                epoch_size,
                                batch_size, learning_rate, x_label, y_label, name, feat, resultlim_all, data_set, coef):




    ######-----
    input_file = open("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate) +
                      "/loss_eval.txt", "r")

    loss_info = input_file.read()
    input_file.close()
    ######-----


    plt.figure(figsize=(10, 8))
    plt.plot([0, np.max(result["act_"+str(predicted_feat)])], [0, np.max(result["act_"+str(predicted_feat)])], ls="--", c=".3", linewidth=0.5)
    sns.scatterplot((result["act_"+str(predicted_feat)]), (result["pred_"+str(predicted_feat)]),
                    hue=np.array(result[oth_feat]))


    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.legend(title="Relatively "+str(oth_feat) +" values")
    plt.title(str(loss_info) +"for " + str(len(result)) + " objects")
    plt.savefig(
        "Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
        + "/NeatPlot_"+str(data_set)+"/Scatter/"+str(name)+"test_prediction_"
        +str(predicted_feat)+"_Colorized_"+str(oth_feat)+".png")



    if feat == 'dustMass' :

        resultlim = result[ (result["ML_uncertanity_" + str(predicted_feat)])  <
                            coef * np.mean( result["ML_uncertanity_" + str(predicted_feat)]) ]

        plt.errorbar((resultlim["act_"+str(predicted_feat)]), (resultlim["pred_"+str(predicted_feat)]),
                     yerr=(resultlim["ML_uncertanity_"+str(predicted_feat)] ) , fmt='o')

        plt.figure(figsize=(10, 8))
        plt.plot([np.min(resultlim["act_" + str(predicted_feat)]), np.max(resultlim["act_" + str(predicted_feat)])],
                 [np.min(resultlim["act_" + str(predicted_feat)]), np.max(resultlim["act_" + str(predicted_feat)])],
                 ls="--", c=".3", linewidth=0.5)

        sns.scatterplot((resultlim["act_" + str(predicted_feat)]), (resultlim["pred_" + str(predicted_feat)]),
                        hue=np.array(resultlim[oth_feat]))
        plt.xscale('log')
        plt.yscale('log')

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.legend(title="Relatively " + str(oth_feat) + " values")
        plt.title(str(loss_info) + "for " + str(len(resultlim)) + " objects")
        plt.savefig(
            "Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
            + "/NeatPlot_"+str(data_set)+"/Scatter/" + str(name) + "test_prediction_" + str(
                predicted_feat) + "_Colorized_" + str(oth_feat) + "_" +str(feat) + "_lim.png")


    elif feat == 'dustTemp':

        resultlim = result[ (result["ML_uncertanity_" + str(predicted_feat)]) <
                            coef * np.mean(result["ML_uncertanity_" + str(predicted_feat)])]
        plt.errorbar((resultlim["act_" + str(predicted_feat)]), (resultlim["pred_" + str(predicted_feat)]),
                     yerr=( resultlim["ML_uncertanity_" + str(predicted_feat)]) , fmt='o')  # ,






        plt.figure(figsize=(10, 8))
        plt.plot([np.min(resultlim["act_" + str(predicted_feat)]), np.max(resultlim["act_" + str(predicted_feat)])],
                 [np.min(resultlim["act_" + str(predicted_feat)]), np.max(resultlim["act_" + str(predicted_feat)])],
                 ls="--", c=".3", linewidth=0.5)
        sns.scatterplot((resultlim["act_" + str(predicted_feat)]), (resultlim["pred_" + str(predicted_feat)]),
                        hue=np.array(resultlim[oth_feat]))
        plt.xscale('log')
        plt.yscale('log')

        plt.xlabel(x_label)
        plt.ylabel(y_label)


        plt.legend(title="Relatively " + str(oth_feat) + " values")
        plt.title(str(loss_info) + "for " + str(len(resultlim)) + " objects")
        plt.savefig(
            "Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
            + "/NeatPlot_"+str(data_set)+"/Scatter/" + str(name) + "test_prediction_" + str(
                predicted_feat) + "_Colorized_" + str(oth_feat) + "_" +str(feat) + "_lim.png")


    else:
        feat="all"
        resultlim=resultlim_all


        plt.figure(figsize=(10, 8))
        plt.plot([np.min(resultlim["act_" + str(predicted_feat)]), np.max(resultlim["act_" + str(predicted_feat)])],
                 [np.min(resultlim["act_" + str(predicted_feat)]), np.max(resultlim["act_" + str(predicted_feat)])],
                 ls="--", c=".3", linewidth=0.5)
        sns.scatterplot((resultlim["act_" + str(predicted_feat)]), (resultlim["pred_" + str(predicted_feat)]),
                        hue=np.array(resultlim[oth_feat]))
        plt.xscale('log')
        plt.yscale('log')

        plt.xlabel(x_label)
        plt.ylabel(y_label)


        plt.legend(title="Relatively " + str(oth_feat) + " values")
        plt.title(str(loss_info) + "for " + str(len(resultlim)) + " objects")
        plt.savefig(
            "Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
            + "/NeatPlot_"+str(data_set)+"/Scatter/" + str(name) + "test_prediction_" + str(
                predicted_feat) + "_Colorized_" + str(oth_feat) + "_" +str(feat) +"_lim.png")



def plot_pred_act_hist(result, predicted_feat,model_num, seed_rnd_num,
                       epoch_size,
                       batch_size, learning_rate, x_label, y_label, name, feat, resultlim_all, data_set):


    ######-----
    input_file = open("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate) +
                      "/loss_eval.txt", "r")

    loss_info = input_file.read()
    input_file.close()

    ######-----


    plt.figure(figsize=(7, 5))
    my_cmap = plt.cm.inferno#viridis  # plt.cm.jet
    my_cmap.set_under('w', 1)
    try:
        plt.plot([0, np.max(result["act_"+str(predicted_feat)])], [0, np.max(result["act_"+str(predicted_feat)])], ls="--", c=".3", linewidth=0.5)
        plt.hist2d((result["act_"+str(predicted_feat)]), (result["pred_"+str(predicted_feat)]), cmap=my_cmap ,
                   bins=600, cmin=1)#, norm=colors.LogNorm(vmin=0.8, vmax=100))

        plt.colorbar()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if predicted_feat == 'dustMass':
            min_lim=-0.001
        elif predicted_feat == 'dustTemp':
            min_lim=-50
        else:
            min_lim=-1
        plt.xlim(min_lim , np.max(result["act_"+str(predicted_feat)]))
        plt.ylim(min_lim, np.max(result["act_"+str(predicted_feat)]))

        plt.title(str(loss_info) +"for " + str(len(result)) + " objects")
        plt.savefig(
            "Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
            + "/NeatPlot_"+str(data_set)+"/Hist/" + str(name) + "test_prediction_" + str(
                feat) + "_lim_"+str(predicted_feat)+".png")
    except:
        print("predicted_feat gets inf values:", predicted_feat)

    plt.figure(figsize=(6, 5))
    my_cmap = plt.cm.inferno#viridis  # plt.cm.jet
    my_cmap.set_under('w', 1)

    try:
        if predicted_feat == 'dustMass':
            min_lim=-5.5
            max_lim=-0.5
            for i_z in result.index:
                if (result["act_"+str(predicted_feat)][i_z]) <= float(1e-11):
                    result["act_"+str(predicted_feat)][i_z] = 1e-11
                if (result["pred_"+str(predicted_feat)][i_z]) <= float(1e-11):
                    result["pred_"+str(predicted_feat)][i_z] = 1e-11

            act_log10=np.log10(result["act_"+str(predicted_feat)])
            pred_log10=np.log10(result["pred_"+str(predicted_feat)])


            x_label=r'$log_{10}(M^{sim}_{d}) (M_{\odot})$'
            y_label=r'$log_{10}(M^{pred}_{d}) (M_{\odot})$'
            plt.plot([-6,0], [-6,0], ls="--", c=".3", linewidth=0.5)

        elif predicted_feat == 'dustTemp':
            min_lim=-50
            max_lim=2100
            act_log10=result["act_"+str(predicted_feat)]
            pred_log10=result["pred_"+str(predicted_feat)]
            x_label=r'$T^{sim}_{d}$ (K)'
            y_label='$T^{pred}_{d}$ (K)'
            plt.plot([0, 2100], [0, 2100], ls="--", c=".3", linewidth=0.5)


        else:
            min_lim=-1
            max_lim=np.max(result["act_"+str(predicted_feat)])
            act_log10=result["act_"+str(predicted_feat)]
            pred_log10=result["pred_"+str(predicted_feat)]


        plt.hist2d(act_log10, pred_log10, cmap=my_cmap ,
                   bins=600, norm=colors.LogNorm(vmin=1, vmax=50))

        plt.colorbar()
        plt.xlabel(x_label, size=16)
        plt.ylabel(y_label, size=16)

        plt.xticks(size=14)
        plt.yticks(size=14)

        plt.xlim(min_lim, max_lim)
        plt.ylim(min_lim, max_lim)

        plt.savefig(
            "Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
            + "/NeatPlot_"+str(data_set)+"/Hist/" + str(name) + "test_prediction_logCbar" + str(
                feat) + "_lim_"+str(predicted_feat)+".png", bbox_inches='tight')
    except:
        print("predicted_feat gets inf values:", predicted_feat)




#####calling all functions
def evaluation_any(model_num, seed_rnd_num,
                   epoch_size,
                   batch_size, learning_rate, redshift, path_preprocessing, data_set,
                   saved_dataset):
    coef_k= 1#a1
    coef=0.2#a2

    if not os.path.exists(
            'Data/ML/model{}_RndSeed{}_{}epoch_{}_LR{}/NeatPlot_{}'.format(str(model_num), str(seed_rnd_num),str(epoch_size),
                                                                    str(batch_size),
                                                                    str(learning_rate),
                                                                           str(data_set))):
        os.makedirs('Data/ML/model{}_RndSeed{}_{}epoch_{}_LR{}/NeatPlot_{}'.format(str(model_num), str(seed_rnd_num),str(epoch_size),
                                                                            str(batch_size),
                                                                            str(learning_rate),
                                                                                   str(data_set)))


    if not os.path.exists(
            'Data/ML/model{}_RndSeed{}_{}epoch_{}_LR{}/NeatPlot_{}/Scatter'.format(str(model_num), str(seed_rnd_num),str(epoch_size),
                                                                           str(batch_size),
                                                                           str(learning_rate),
                                                                           str(data_set))):
        os.makedirs('Data/ML/model{}_RndSeed{}_{}epoch_{}_LR{}/NeatPlot_{}/Scatter'.format(str(model_num), str(seed_rnd_num),str(epoch_size),
                                                                                   str(batch_size),
                                                                                   str(learning_rate),
                                                                                   str(data_set)))


    if not os.path.exists(
            'Data/ML/model{}_RndSeed{}_{}epoch_{}_LR{}/NeatPlot_{}/Hist'.format(str(model_num), str(seed_rnd_num),str(epoch_size),
                                                                                   str(batch_size),
                                                                                   str(learning_rate),
                                                                                   str(data_set))):
        os.makedirs('Data/ML/model{}_RndSeed{}_{}epoch_{}_LR{}/NeatPlot_{}/Hist'.format(str(model_num), str(seed_rnd_num),str(epoch_size),
                                                                                           str(batch_size),
                                                                                           str(learning_rate),
                                                                                           str(data_set)))



    result_all=pd.read_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate) +
                              "/result_all.pkl")

    X_sample=pd.read_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                            + "/MissingLabel_After_sen_cutoff_X_"+str(saved_dataset)+".pkl")


    result=result_all[result_all.index.isin(X_sample.index)]
    other_features=pd.read_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                                  + "/MissingLabel_After_sen_cutoff_otherfeatures_"+str(saved_dataset)+".pkl")



    result["Model"] = np.array(result["Model"])
    result["redshift"]=redshift
    result["ML_uncertanity_dustMass_log"] = np.log10(result["ML_uncertanity_dustMass"])



    # otherFeatures_list = ["radiusSN","tempSN", "L", "spe", "grainSize", "dustMass", "dustTemp","dustTempSigma","tau", "redshift" ]
    # predicted_feat_list = ["dustMass","dustTemp", "spe"]#["grainSize", "spe", "dustTempSigma", "dustTemp", "dustMass" , "tau"]
    x_label_list=  [r"Models dustMass $(M_{sun})$" , "Models dustTemp (K)", "Models species" ]

    y_label_list=[r"Predicted dustMass $(M_{sun})$" , "Predicted dustTemp (K)" ,  "Predicted species"]


    result["ML_uncertanity_dustMass_norm"]=result["ML_uncertanity_dustMass"]/np.max(result["ML_uncertanity_dustMass"])
    result["ML_uncertanity_spe_norm"]=result["ML_uncertanity_spe"]/np.max(result["ML_uncertanity_spe"])
    result["ML_uncertanity_dustTemp_norm"]=result["ML_uncertanity_dustTemp"]/np.max(result["ML_uncertanity_dustTemp"])

    list_f=["ML_uncertanity_dustMass","ML_uncertanity_spe", "ML_uncertanity_dustTemp"]

    for i in list_f :
        plt.figure()
        plt.hist(result[str(i)+"_norm"], bins=100, fill= False)
        plt.title("normalized" +str(i))
        plt.yscale('log')
        plt.savefig(
            "Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
            + "/NeatPlot_"+str(data_set)+"/hist_"+str(i)+"_norm.png")


    input_file = open("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate) +
                      "/loss_eval.txt", "r")

    loss_info = input_file.read()
    input_file.close()



    result["pred_dustMass_noTransformed"]=  1e+1 * result["pred_dustMass"]
    result["pred_dustTemp_noTransformed"]= (result['pred_dustTemp'])/2200
    result["pred_spe_noTransformed"]=result["pred_spe"]

    resultlim_dustMass=result[np.abs((result["pred_dustMass_noTransformed"] + (coef_k * result["ML_uncertanity_dustMass"]) )* 1e-1
                                     - result["pred_dustMass"]) < coef * np.abs(result["pred_dustMass"])]


    resultlim_dustTemp=resultlim_dustMass[np.abs((resultlim_dustMass["pred_dustTemp_noTransformed"] + (coef_k * resultlim_dustMass["ML_uncertanity_dustTemp"])) * 2200
                                                 - resultlim_dustMass["pred_dustTemp"]) < coef * np.abs(resultlim_dustMass["pred_dustTemp"])]

    resultlim_dustTemp_sep=result[np.abs((result["pred_dustTemp_noTransformed"] + (coef_k * result["ML_uncertanity_dustTemp"])) * 2200
                                         - result["pred_dustTemp"]) < coef * np.abs(result["pred_dustTemp"])]

    resultlim_spe_sep=result[np.abs((result["pred_spe_noTransformed"] + (coef_k * result["ML_uncertanity_spe"]))
                                    - result["pred_spe"]) < coef ]#* np.abs(resultlim_dustTemp["pred_spe"])


    resultlim_all=resultlim_dustTemp[np.abs((resultlim_dustTemp["pred_spe_noTransformed"] + (coef_k * resultlim_dustTemp["ML_uncertanity_spe"]))
                                            - resultlim_dustTemp["pred_spe"]) < coef ]#* np.abs(resultlim_dustTemp["pred_spe"])




    mean_sigma_list=[np.mean(result["ML_uncertanity_dustMass"]),
                     np.mean(result["ML_uncertanity_spe"]) ,
                     np.mean(result["ML_uncertanity_dustTemp"])
        , 0]

    result_highsig=result[~result.index.isin(resultlim_all.index)]
    resultlim_all.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                            + "/resultlim_"+str(saved_dataset)+".pkl")
    result_highsig.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                             + "/result_highsig_"+str(saved_dataset)+".pkl")


    resultlim_list=[resultlim_all]
    feat_list=["all"]
    predicted_feat_list = ["dustMass","dustTemp", "spe"]
    for i_predicted_feat in range(len(predicted_feat_list)):
        predicted_feat = predicted_feat_list[i_predicted_feat]

        x_label = x_label_list[i_predicted_feat]
        y_label = y_label_list[i_predicted_feat]

        # plot_pred_act__uncertanity(result, predicted_feat, model_num, seed_rnd_num,
        #                            epoch_size,
        #                            batch_size, learning_rate, x_label, y_label, "", data_set, coef)
        # for oth_feat in otherFeatures_list:
        #     # count = 0
        #     plot_pred_act_otherfeatures(result, oth_feat, predicted_feat, model_num, seed_rnd_num,
        #                                 epoch_size,
        #                                 batch_size, learning_rate,
        #                                 x_label, y_label ,"", feat, resultlim_all, data_set, coef)
        # #     # count=count+1
        #

        countlim_feat = 0

        resultlim_=resultlim_list[0]
        coutlim_predicted_feat=0
        count_num = len(resultlim_)
        feat = feat_list[countlim_feat]
        mean_sigma = mean_sigma_list[countlim_feat]



        # eval_measure_lim(resultlim_, predicted_feat, model_num, seed_rnd_num,
        #                  epoch_size,
        #                  batch_size, learning_rate, mean_sigma, feat)

        #<= float(0.25)


        result_all=result
        result_spe=result[result["pred_spe"] > float(0.25)]


        resultlim_all_spe=resultlim_all[resultlim_all["pred_spe"] > float(0.25)]#!= 'lowDust' ]#> float(0.25)]#
        resultlim_dustTemp_sep_spe=resultlim_dustTemp_sep[resultlim_dustTemp_sep["pred_spe"] > float(0.25)]
        resultlim_dustMass_spe=resultlim_dustMass[resultlim_dustMass["pred_spe"] > float(0.25)]
        resultlim_spe_sep_spe=resultlim_spe_sep[resultlim_spe_sep["pred_spe"] > float(0.25)]



        plot_pred_act_hist(resultlim_all_spe, predicted_feat, model_num, seed_rnd_num,
                           epoch_size,
                           batch_size, learning_rate, x_label, y_label, "LIM", feat, resultlim_all, data_set)
        if predicted_feat == 'dustMass':
            df_md_frac_lim_test_i, df_md_bias_lim_test_i, df_md_rms_lim_test_i= eval_measure(resultlim_all_spe, predicted_feat, model_num, seed_rnd_num,
                                                                                             epoch_size,
                                                                                             batch_size, learning_rate, data_set, "LIM")


        elif predicted_feat == 'dustTemp':
            df_td_frac_lim_test_i, df_td_bias_lim_test_i, df_td_rms_lim_test_i= eval_measure(resultlim_all_spe, predicted_feat, model_num, seed_rnd_num,
                                                                                             epoch_size,
                                                                                             batch_size, learning_rate, data_set, "LIM")
        else:
            df_spe_frac_lim_test_, df_spe_bias_lim_test_, df_spe_rms_lim_test_= eval_measure(resultlim_all_spe, predicted_feat, model_num, seed_rnd_num,
                                                                                             epoch_size,
                                                                                             batch_size, learning_rate, data_set, "LIM")

        #
        plot_pred_act_hist(result_spe, predicted_feat, model_num, seed_rnd_num,
                           epoch_size,
                           batch_size, learning_rate, x_label, y_label, "ALL", feat, result, data_set)
        if predicted_feat == 'dustMass':
            df_md_frac_test_i, df_md_bias_test_i, df_md_rms_test_i=eval_measure(result_spe, predicted_feat, model_num, seed_rnd_num,
                                                                                epoch_size,
                                                                                batch_size, learning_rate, data_set, "ALL")
        elif predicted_feat == 'dustTemp':
            df_td_frac_test_i, df_td_bias_test_i, df_td_rms_test_i=eval_measure(result_spe, predicted_feat, model_num, seed_rnd_num,
                                                                                epoch_size,
                                                                                batch_size, learning_rate, data_set, "ALL")
        else:
            df_spe_frac_test_i, df_spe_bias_test_i, df_spe_rms_test_i=eval_measure(result_spe, predicted_feat, model_num, seed_rnd_num,
                                                                                   epoch_size,
                                                                                   batch_size, learning_rate, data_set, "ALL")



        countlim_feat = countlim_feat + 1

    i_predicted_feat = i_predicted_feat + 1
    conf_test_c_i, conf_test_m_i, conf_test_s_i, counter_c_all_i, counter_m_all_i, counter_s_all_i = \
        conf_plot(result_spe, "spe" , model_num, seed_rnd_num,
                  epoch_size,
                  batch_size, learning_rate, "", data_set)

    feat="all"
    count_num = len(resultlim_all_spe)

    conf_lim_test_c_i, conf_lim_test_m_i, conf_lim_test_s_i, counter_c_lim_i, counter_m_lim_i, counter_s_lim_i = \
        conf_plot(resultlim_all_spe, "spe",model_num, seed_rnd_num,
                  epoch_size,
                  batch_size, learning_rate, "LIM", data_set)
    df_num_test_i=len(result_spe)
    df_num_lim_test_i=len(resultlim_all_spe)
    df_num_lim_test_dustTemp_i=len(resultlim_dustTemp_sep_spe)
    df_num_lim_test_spe_i=len(resultlim_spe_sep_spe)
    df_num_lim_test_dustMass_i=len(resultlim_dustMass_spe)


    ############----------------pre-existing and newly-formed
    df_lowR=other_features[other_features["radiusSN"]< (0.8 * 5e+16)]
    df_highR=other_features[other_features["radiusSN"]>=(0.8 * 5e+16)]

    df_lowR.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                      +
                      "/df_lowR.pkl")

    df_highR.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate) +
                       "/df_highR.pkl")

    index_new=df_lowR
    index_pre=df_highR

    df__test_pre_i=result_spe[result_spe.index.isin(index_pre.index)]
    df__test_pre_i.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate) +
                             "/df_highR_withoutNodust.pkl")
    df__test_new_i=result_spe[result_spe.index.isin(index_new.index)]
    df__test_new_i.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate) +
                             "/df_lowR_withoutNodust.pkl")

    df__lim_test_pre_i=resultlim_all_spe[resultlim_all_spe.index.isin(index_pre.index)]
    df__lim_test_pre_i.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate) +
                                 "/df_highR_alllim.pkl")

    df__lim_test_new_i=resultlim_all_spe[resultlim_all_spe.index.isin(index_new.index)]
    df__lim_test_new_i.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate) +
                                 "/df_lowR_alllim.pkl")

    df__lim_test_pre_dustTemp_i=resultlim_dustTemp_sep_spe[resultlim_dustTemp_sep_spe.index.isin(index_pre.index)]
    df__lim_test_new_dustTemp_i=resultlim_dustTemp_sep_spe[resultlim_dustTemp_sep_spe.index.isin(index_new.index)]
    df__lim_test_pre_spe_i=resultlim_spe_sep_spe[resultlim_spe_sep_spe.index.isin(index_pre.index)]
    df__lim_test_new_spe_i=resultlim_spe_sep_spe[resultlim_spe_sep_spe.index.isin(index_new.index)]
    df__lim_test_pre_dustMass_i=resultlim_dustMass_spe[resultlim_dustMass_spe.index.isin(index_pre.index)]
    df__lim_test_new_dustMass_i=resultlim_dustMass_spe[resultlim_dustMass_spe.index.isin(index_new.index)]


    # df_md_frac_lim_test_i=df_md_frac_lim_test_
    # df_md_bias_lim_test_i=df_md_bias_lim_test_
    # df_md_rms_lim_test_i=df_md_rms_lim_test_
    # df_td_frac_lim_test_i=df_td_frac_lim_test_
    # df_td_bias_lim_test_i=df_td_bias_lim_test_
    # df_td_rms_lim_test_i=df_td_rms_lim_test_
    conf_lim_test_low_i=0
    # conf_lim_test_c_i=
    # conf_lim_test_m_i=
    # conf_lim_test_s_i=
    return df_num_test_i, df_num_lim_test_i, \
           df_md_frac_lim_test_i, df_md_bias_lim_test_i, df_md_rms_lim_test_i, df_td_frac_lim_test_i, \
           df_td_bias_lim_test_i, df_td_rms_lim_test_i, conf_lim_test_low_i, conf_lim_test_c_i, conf_lim_test_m_i, \
           conf_lim_test_s_i, \
           df_md_frac_test_i, df_md_bias_test_i, df_md_rms_test_i, df_td_frac_test_i, \
           df_td_bias_test_i, df_td_rms_test_i,  conf_test_c_i, conf_test_m_i, \
           conf_test_s_i, counter_c_all_i, counter_m_all_i, counter_s_all_i,counter_c_lim_i, counter_m_lim_i, counter_s_lim_i, \
           df_num_lim_test_dustTemp_i,df_num_lim_test_spe_i,df_num_lim_test_dustMass_i, \
           df__test_pre_i, df__test_new_i, df__lim_test_pre_i, df__lim_test_new_i, df__lim_test_pre_dustTemp_i, \
           df__lim_test_new_dustTemp_i, df__lim_test_pre_spe_i, df__lim_test_new_spe_i, \
           df__lim_test_pre_dustMass_i, df__lim_test_new_dustMass_i#, df_loss_train_i, df_loss_val_i


