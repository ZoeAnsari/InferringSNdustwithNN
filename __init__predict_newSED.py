from src.predict_newSED import predict_SNdust
import pandas as pd
import pickle



##define a trained model following the example below
epoch=1500 #for S1: 2000,  for S2 and S3:1500
scenario='ZERROR' # S1:fixedz, S2:Z, S3:ZERROR
step_SHAP_process=9##corresponding JWST filters should be provided (See readme file)
step_saved_file=step_SHAP_process+1

#These values are fixed for all of our trained neural networks
n_=1####The number of compnents
learning_rate=1e-5
batch_size=64
rnd_seed= step_saved_file+30

trainedmodel_path="Data/ML/model"+str(step_saved_file)+"SHAP1compo_"+ str(scenario)+\
"_RndSeed"+str(rnd_seed)+"_"+str(epoch)+"epoch_"+ str(batch_size)+ "_LR"+str(learning_rate)


####Here we input an example set from our synthesised photometric data set in S3
Data_example=pd.read_csv("Data/JWST_MOCASSIN_SED_Example/Example_set_for_prediction.csv")
Data_example=Data_example.drop(columns={"Unnamed: 0"})
print(Data_example.head())


####Input SED with corresponding JWST filters that the trained neural network used in the chosen step (See readme file)
###Here we selected step 9
list_cols_modeified=["F070W_NIR","F140M_NIR","F356W_NIR","F480M_NIR","F560W_MIRI",
                     "F770W_MIRI","F1000W_MIRI","F1130W_MIRI","F1500W_MIRI","F1800W_MIRI",
                     "z"]
###
print("The JWST filters that are needed to estimate dust properties with the chosen trained neural network: ",
      list_cols_modeified
      )
###The input SED should be imported as a python dataframe following the example below

X_mags=Data_example[list_cols_modeified]
###Input the corresponding redshift
redshift=Data_example["z"]


X=X_mags
X["z"]=redshift


###predict the dust properties with the chosen trained neural network
predict_SNdust(X, trainedmodel_path, epoch, n_, scenario, rnd_seed, batch_size, learning_rate)
















