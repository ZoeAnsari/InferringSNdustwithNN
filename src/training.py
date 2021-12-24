from tensorflow import keras, random
import matplotlib.pyplot as plt
import time as time
import numpy as np
import pandas as pd
import os
import mdn
import random  as rd
# rd.seed(21)
# random.set_seed(21)



def NN_reg_dense(n_, X_all, X_train, X_val,
                 y_all, y_train, y_val,
                 otherfeatures_all, otherfeatures_train, otherfeatures_val,
                 model_num, seed_rnd_num, epoch_size,
                 batch_size,learning_rate, shap_num):

    if not os.path.exists(
            'Data/ML/model{}_RndSeed{}_{}epoch_{}_LR{}'.format(str(model_num), str(seed_rnd_num),str(epoch_size),
                                                      str(batch_size),
                                                      str(learning_rate))):
        os.makedirs('Data/ML/model{}_RndSeed{}_{}epoch_{}_LR{}'.format(str(model_num), str(seed_rnd_num),str(epoch_size),
                                                                       str(batch_size),
                                                                       str(learning_rate)))

    if not os.path.exists(
            'Data/ML/model{}_RndSeed{}_{}epoch_{}_LR{}/plot'.format(str(model_num), str(seed_rnd_num),str(epoch_size),
                                                               str(batch_size),
                                                               str(learning_rate))):
        os.makedirs('Data/ML/model{}_RndSeed{}_{}epoch_{}_LR{}/plot'.format(str(model_num), str(seed_rnd_num),str(epoch_size),
                                                                       str(batch_size),
                                                                       str(learning_rate)))

    if not os.path.exists('Data/ML/model{}_RndSeed{}_{}epoch_{}_LR{}/model'.format(str(model_num), str(seed_rnd_num),str(epoch_size),
                                                                             str(batch_size),
                                                                             str(learning_rate))):
        os.makedirs('Data/ML/model{}_RndSeed{}_{}epoch_{}_LR{}/model'.format(str(model_num), str(seed_rnd_num),str(epoch_size),
                                                                       str(batch_size),
                                                                       str(learning_rate)))




    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    y_train=pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_val)
    y_all = pd.DataFrame(y_all)

    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_all = np.array(y_all)


    try:
        model=keras.Sequential()
        N_MIXES = n_  # (n_components)  # number of mixture components
        OUTPUT_DIMS = 3  # number of real-values predicted by each mixture component
        prelu_cont=0

        st=time.time()
        model.add(keras.layers.Dense(512, input_shape=X_train.shape[1:]))
        model.add(keras.layers.PReLU(alpha_initializer=keras.initializers.Constant(value=prelu_cont)))
        model.add(keras.layers.Dense(256))
        model.add(keras.layers.PReLU(alpha_initializer=keras.initializers.Constant(value=prelu_cont)))
        model.add(keras.layers.Dense(128))
        model.add(keras.layers.PReLU(alpha_initializer=keras.initializers.Constant(value=prelu_cont)))
        model.add(keras.layers.Dense(64))
        model.add(keras.layers.PReLU(alpha_initializer=keras.initializers.Constant(value=prelu_cont)))
        model.add(keras.layers.Dense(32))
        model.add(keras.layers.PReLU(alpha_initializer=keras.initializers.Constant(value=prelu_cont)))
        model.add(keras.layers.Dense(16))
        model.add(keras.layers.PReLU(alpha_initializer=keras.initializers.Constant(value=prelu_cont)))
        model.add(keras.layers.Dense(8))
        model.add(keras.layers.PReLU(alpha_initializer=keras.initializers.Constant(value=prelu_cont)))
        model.add(keras.layers.Dense(4))
        model.add(keras.layers.PReLU(alpha_initializer=keras.initializers.Constant(value=prelu_cont)))
        model.add(mdn.MDN(OUTPUT_DIMS, N_MIXES))

        print(model.summary())
        model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMS, N_MIXES),
                      optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

        print("before fit")
        history=model.fit(X_train, y_train, epochs=epoch_size, batch_size=batch_size, validation_data=(X_val, y_val)) #steps_per_epoch=10,
        model.save("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                   +"/model/Dust_NN_"+str(shap_num)+".h5")

        # list all data in history
        loss_df=pd.DataFrame(np.array(history.history['loss']))
        loss_df["loss_val"]=np.array(history.history['val_loss'])
        loss_df.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                          +"/plot/loss_df.pkl")



        #############################plot history--------------------------------------------------------------------------------------------------------------
        print(history.history.keys())
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                    +"/plot/loss.png")

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.ylim(-10,1)
        # plt.yscale('log')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)
                    + "/plot/loss_lim.png")

        # plt.figure()
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.yscale('log')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.savefig("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate) + "/plot/loss_log.png")
        #
        #
        # plt.figure()
        # plt.plot(np.log10(history.history['loss']))
        # plt.plot(np.log10(history.history['val_loss']))
        # plt.title('log_loss')
        # plt.ylabel('log_loss')
        # plt.xlabel('epoch')
        # plt.yscale('log')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.savefig("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate)+"/plot/logscale_log_loss.png")
        print("after fitting")


        #############################--------------------------------------------------------------------------------------------------------------


        loss_val = history.history['val_loss'][epoch_size -1]
        print("----loss val", loss_val)


        y_pred = model.predict(X_val)
        y_all_pred=model.predict(X_all)



        # Split up the mixture parameters (for future fun)

        prediction_ = np.apply_along_axis((lambda a: a[:OUTPUT_DIMS]), 1, y_pred)  # the means
        sigs_ = np.apply_along_axis((lambda a: a[OUTPUT_DIMS:2 * OUTPUT_DIMS]), 1,
                                    y_pred)  # the standard deviations
        pis_ = np.apply_along_axis((lambda a: mdn.softmax(a[2 * OUTPUT_DIMS:])), 1,
                                   y_pred)  # the mixture components

        prediction_all = np.apply_along_axis((lambda a: a[:OUTPUT_DIMS]), 1, y_all_pred)  # the means
        sigs_all = np.apply_along_axis((lambda a: a[OUTPUT_DIMS:2 * OUTPUT_DIMS]), 1,
                                       y_all_pred)  # the standard deviations
        pis_all = np.apply_along_axis((lambda a: mdn.softmax(a[2 * OUTPUT_DIMS:])), 1,
                                      y_all_pred)  # the mixture components




        prediction_=pd.DataFrame(prediction_)
        sigs_=pd.DataFrame(sigs_)
        pis_=pd.DataFrame(pis_)


        prediction_all=pd.DataFrame(prediction_all)
        sigs_all=pd.DataFrame(sigs_all)
        pis_all=pd.DataFrame(pis_all)







        y_val_=pd.DataFrame(y_val)
        y_all_=pd.DataFrame(y_all)

        print("prediction----------", prediction_)
        print("prediction_validation----------", y_val_)


        ######rescale the target values
        result = pd.DataFrame( 1e-1 *prediction_[0])
        result=result.rename(columns={0: "pred_dustMass"})
        result["act_dustMass"] = ( 1e-1 * y_val_[0])

        result["ML_uncertanity_dustMass"] = sigs_[0]

        result["pred_dustTemp"]=((prediction_[1]) * 2200)
        result["act_dustTemp"] = ((y_val_[1]) * 2200)
        result["ML_uncertanity_dustTemp"] = sigs_[1]

        result["pred_spe"] = prediction_[2]
        result["act_spe"] = y_val_[2]
        result["ML_uncertanity_spe"] = sigs_[2]

        result["ML_prob"] = pis_[0]



        ############all

        result_all = pd.DataFrame( 1e-1 * prediction_all[0])

        result_all = result_all.rename(columns={0: "pred_dustMass"})
        result_all["act_dustMass"] =  1e-1 *y_all_[0]
        result_all["ML_uncertanity_dustMass"] = sigs_all[0]

        #
        result_all["pred_dustTemp"] = (( prediction_all[1]) * 2200)
        result_all["act_dustTemp"] = (( y_all_[1]) * 2200)
        result_all["ML_uncertanity_dustTemp"] = sigs_all[1]

        result_all["pred_spe"] = prediction_all[2]
        result_all["act_spe"] = y_all_[2]
        result_all["ML_uncertanity_spe"] = sigs_all[2]


        result_all["ML_prob"] = pis_all[0]


        result["radiusSN"] = np.array(otherfeatures_val['radiusSN'])
        result["tempSN"] = np.array(otherfeatures_val["tempSN"])
        result["L"] = np.array(otherfeatures_val["L"])
        result["spe"]= np.array(otherfeatures_val["species"])
        result["grainSize"] = np.array(otherfeatures_val["grainSize"])
        result["dustMass"] = np.array(otherfeatures_val["Md"])
        result["dustTemp"] = np.array(otherfeatures_val["tempDust"])
        result["dustTempSigma"] = np.array(otherfeatures_val["dusttemp_sigma"])
        result["tau"]= np.array(otherfeatures_val["tau"])
        result["Model"] = np.array(otherfeatures_val["Model"])


        print("------------------------------saving the result--------------------------------------------------------------")


        result_all["radiusSN"] = np.array(otherfeatures_all['radiusSN'])
        result_all["tempSN"] = np.array(otherfeatures_all["tempSN"])
        result_all["L"] = np.array(otherfeatures_all["L"])
        result_all["spe"] = np.array(otherfeatures_all["species"])
        result_all["grainSize"] = np.array(otherfeatures_all["grainSize"])
        result_all["dustMass"] = np.array(otherfeatures_all["Md"])
        result_all["dustTemp"] = np.array(otherfeatures_all["tempDust"])
        result_all["dustTempSigma"] = np.array(otherfeatures_all["dusttemp_sigma"])
        result_all["tau"] = np.array(otherfeatures_all["tau"])
        result_all["Model"] = np.array(otherfeatures_all["Model"])





        result.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate) +
                         "/result.pkl")

        result_all.to_pickle("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate) +
                             "/result_all.pkl")

        out_file = open("Data/ML/model" + str(model_num) +"_RndSeed"+str(seed_rnd_num)+"_"+str(epoch_size)+"epoch_"+str(batch_size)+"_LR"+str(learning_rate) +
                        "/loss_eval.txt", "w")
        train_loss= history.history['loss'][-1]


        out_file.write("loss train:%s\n" % train_loss)
        val_loss=history.history['val_loss'][-1]
        out_file.write("loss test:%s\n" % val_loss)
        out_file.close()




        print("------------------------------saving the result is complete--------------------------------------------------------------")
        end=time.time()
        print("Training took ", str(end-st)," seconds to be done")



    except Exception as e:
        print("Error:", e)










