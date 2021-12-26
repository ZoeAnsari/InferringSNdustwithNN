import pandas as pd
import numpy as np
import gzip
import time as time
import pickle
from astropy.cosmology import FlatLambdaCDM , z_at_value
import astropy.units as u
import gc




cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)




def convert_data_to_tables(path_preprocessing,name_, range1_outputs, range11_outputs,range2_outputs, z_upper_limit
                       , dataset_label):

    #The unit of SEDs is x9.521e13 erg/s/Hz so
    # that dividing by a distance in pc squared gives you a flux in Jy.
    # LDS = D * 3.086e+18 * 1e+6  # luminosity distance from Mpc to cm
    # DPC= D * 1e+6


    ###############making dataframe for all sed.out files
    # range1_outputs=200
    list_count_files1=["%03d" % x for x in range(range1_outputs,range11_outputs)]
    # list_count_files1.remove("000")
    list_count_files12 = [x for x in range(range1_outputs,range11_outputs)]
    # list_count_files12.remove(0)
    list_count_files2 = ["%03d" % x for x in range(range2_outputs)]



    z_list_tot=[]
    count_obj=0
    start=time.time()


    ######---dust temp revised
    dust_tempnew = str(path_preprocessing)+"/dusttemperatures.txt"
    dict_read = {}
    dict_read["file"] = []
    dict_read["dusttemp_mu"] = []
    dict_read["dusttemp_sigma"] = []
    dict_read["index_dusttemp"]=[]
    dict_read["dustmass"]=[]
    with open(dust_tempnew) as f:
        count = 0
        one = 0
        two = 0
        three = 0
        four = 0
        five = 0
        six = 0
        seven = 0
        eight = 0
        nine = 0
        for line in f:
            count = count + 1
            try:

                # print(line.strip().split())
                # # exit()
                # one, two, three, four, five, six, seven, eight, nine, ten, eleven = line.strip().split()
                # # print(one.strip().split("_"))
                # # exit()
                # one_output, one_extra1, one_index = one.strip().split("_")
                # # if dataset_label == 'higherL_2_':
                # #     one_output, one_extra1, one_extra2, one_extra3, one_extra4 , one_index = one.strip().split("_")
                # # elif dataset_label == '':
                # #     one_output, one_extra1, one_index = one.strip().split("_")
                # # else:
                # #     one_output, one_extra1, one_extra2, one_extra3 , one_index = one.strip().split("_")
                #
                # dict_read["file"].append(one)
                # dict_read["index_dusttemp"].append(int(one_index))
                # dict_read["dustmass"].append(three)
                # dict_read["dusttemp_mu"].append(seven)
                # dict_read["dusttemp_sigma"].append(nine)

                # print(line.strip().split())
                # exit()
                one, two, three, four, five, six, seven, eight, nine = line.strip().split()
                # print(one.strip().split("_"))
                # exit()
                if dataset_label == 'higherL_2_':
                    one_output, one_extra1, one_extra2, one_extra3, one_extra4 , one_index = one.strip().split("_")
                elif dataset_label == '':
                    one_output, one_extra1, one_index = one.strip().split("_")
                else:
                    one_output, one_extra1, one_extra2, one_extra3 , one_index = one.strip().split("_")

                dict_read["file"].append(one)
                dict_read["index_dusttemp"].append(int(one_index))
                dict_read["dustmass"].append(three)
                dict_read["dusttemp_mu"].append(seven)
                dict_read["dusttemp_sigma"].append(nine)
                # print(one, one_index, three, seven, nine)
                # exit()

            except Exception as e:
                print("no profile", e)



    df_dusttemp_new = pd.DataFrame.from_dict(dict_read, dtype='float')
    # print("df_dusttemp_new",df_dusttemp_new)
    # exit()
    df_dusttemp_new.index = df_dusttemp_new["index_dusttemp"]
    df_dusttemp_new["index"]=df_dusttemp_new.index
    df_dusttemp_new=df_dusttemp_new.drop(columns="index_dusttemp")

    df_dusttemp_new.to_pickle(str(path_preprocessing)+"/dusttemperatures_new_"+str(name_)+".pkl")
    df_dusttemp_new=pd.read_pickle(str(path_preprocessing)+"/dusttemperatures_new_"+str(name_)+".pkl")
    # exit()




    data_count=0
    for c1 in list_count_files1:

        start0=time.time()
        print("count_obj", count_obj)
        c12 = list_count_files12[count_obj]
        z_list = 1e-4 * (np.random.uniform(1, z_upper_limit, range2_outputs))
        # z_list_tot.append(z_list)
        z_list_tot.extend(z_list)
        print("z_list_tot------------",z_list_tot)
        z_list_value=[]
        c_try=0
        data_index = []
        index_F = []

        for c2 in list_count_files2:
            dict_read = {}

            try:

                path=str(path_preprocessing)+"/outputs_"+str(dataset_label)+str(c1)+"/output_"+str(c12)+str(c2)+"/"
                print("path----------", path)
                txt=path + 'SED.out.gz'



                dict_read["nu_Ryd"]=[]
                dict_read["lambda_um"]=[]
                dict_read["F"]=[]


                with gzip.open(txt) as f:
                    count=0
                    one=0
                    two=0
                    three =0
                    for line in f:
                        count=count +1
                        # if count > 71 and count < 220:
                        # print(line.strip(" "))
                        try:
                            one, two, three=line.strip().split()
                            one=str(one)
                            two=str(two)
                            three=str(three)
                            # print(one.strip().split("'"))
                            one_extra, one_val,one_extra1=one.strip().split("'")
                            two_extra, two_val,two_extra1=two.strip().split("'")
                            three_extra, three_val, three_extra1=three.strip().split("'")
                            print(one_val)
                            # print(float(two_val))
                            # print(three_val)
                            # exit()

                            if float(two_val) < int(32): #lambda < 32micron
                                # print("sth")

                                dict_read["nu_Ryd"].append(float(one_val))
                                dict_read["lambda_um"].append(float(two_val))
                                dict_read["F"].append(float(three_val))
                            # exit()
                        except:
                            print("nothing")


                df_read=pd.DataFrame.from_dict(dict_read, dtype='float')
                print(df_read.head())
                print(df_read.tail())

                # const6 = (1e+36) / (4 * np.pi * (LDS ** 2))
                # const1= D#/(DPC**2)#(1e+36)/(4*np.pi * (LDS**2))#/(D **2)#((1e+36) /(4 * np.pi * (LDS**2)))

                z_list_value.append(z_list[int(c_try)])


                print("out of there")
                print(len(z_list))
                print(len(df_read))
                z_list_this=z_list[int(c_try)]
                D_pc= 1e+6 * (cosmo.luminosity_distance(z_list_this))
                D_pc=D_pc.value
                # print(type(D_pc))
                # D_pc=float(D_pc)
                # exit()
                print(c_try)
                f_Jy_z=[]
                for i in range(len(df_read)):

                    # mult=  (df_read["F"][i]) * (1+z_list[int(c_try)]) #* 1/const6
                    mult=  (df_read["F"][i]) * (1+z_list_this) * (1/ (4* (np.pi) * (D_pc**2)) ) #* 1/const6
                    f_Jy_z.append(mult)
                df_read["F_Jy_z"]=f_Jy_z#np.array(f_Jy_z).astype(float)
                print(df_read["F_Jy_z"])
                print("here----")
                # exit()
                # print(df_read.tail())
                ###########################-----upside down the dataframe and drop zero values on flux
                # # df_read=df_read.iloc[::-1]
                df_read=df_read.reindex(index=df_read.index[::-1])
                df_read.reset_index(inplace=True, drop=True)
                df_read = df_read[df_read["F"] != 0]
                df_read.to_pickle(path+"df"+str(name_)+".pkl")
                # print(path+"df"+str(name_)+".pkl")

                index_F.append(int(str(c12) + str(c2)))

                # print(count_obj)
                z = z_list[int(c_try)]
                print("z----",z)
                c_try = c_try + 1
                data_index.append(data_count)
                data_count = data_count + 1






            except Exception as e:

                print("no profile", e)
                # path = "Data/Results_from_MOCASSIN/outputs_" + str(c1) + "/output_" + str(c12) + str(c2) + "/"
                c_try = c_try
                data_count = data_count + 1





        inputs_path="Data/Results_from_MOCASSIN/results/inputs_"+str(dataset_label)+str(c1)+".txt"
        inputs=inputs_path + '.gz'
        print(inputs)


        dict_read={}
        dict_read["Model"]=[]
        dict_read["T"]=[]
        dict_read["L"]=[]
        dict_read["Md"]=[]
        dict_read["Rout"]=[]
        dict_read["silicate"]=[]
        dict_read["gs"]=[]
        dict_read["index_other"]=[]




        # with open(inputs) as f:
        with gzip.open(inputs) as f:
            count=0
            one=0
            two=0
            three =0
            for line in f:
                count = count + 1
                try:

                    print(line.strip().split())
                    one, two, three, four, five, six, seven=line.strip().split()
                    one=str(one)
                    print(one.strip().split("_"))
                    if dataset_label == 'higherL_2_':
                        one_output, one_extra1, one_extra2,one_extra3, one_index_ = one.strip().split("_")
                    elif dataset_label == '':
                        one_output, one_extra1, one_index_ = one.strip().split("_")
                    else:
                        one_output, one_extra1, one_extra2, one_index_ = one.strip().split("_")
                    print(one_index_.strip().split("'"))
                    # exit()
                    # one_index_=str(one_index_)
                    one_index, one_index_extra=one_index_.strip().split("'")

                    dict_read["index_other"].append(int(one_index))
                    dict_read["Model"].append(one)
                    dict_read["T"].append(two)
                    dict_read["L"].append(three)
                    dict_read["Md"].append(four)
                    dict_read["Rout"].append(five)
                    dict_read["silicate"].append(six)
                    dict_read["gs"].append(seven)

                except Exception as e:
                    print("nothing")



        inputs=pd.DataFrame.from_dict(dict_read, dtype='float')

        print(inputs.columns)
        print(len(inputs))
        inputs.to_pickle(str(path_preprocessing)+"/preprocessed/inputs_"+str(c1)+str(name_)+".pkl")
        df_inputs=pd.read_pickle(str(path_preprocessing)+"/preprocessed/inputs_"+str(c1)+str(name_)+".pkl")


        print(df_inputs.head())
        print(df_inputs.tail())
        # exit()

        opticaldepth_path="Data/Results_from_MOCASSIN/results/opticaldepths_"+str(dataset_label)+str(c1)+".txt"
        opticaldepth=opticaldepth_path + '.gz'


        dict_read={}
        dict_read["Model"]=[]
        dict_read["tau"]=[]
        dict_read["index_tau"]=[]

        with gzip.open(opticaldepth) as f:
            count=0
            one=0
            two=0
            three =0
            for line in f:
                count=count +1
                try:
                    one, two=line.strip().split()
                    one=str(one)
                    if dataset_label == 'higherL_2_':
                        one_output, one_extra1, one_extra2,one_extra3, one_index_ = one.strip().split("_")
                    elif dataset_label == '':
                        one_output, one_extra1, one_index_ = one.strip().split("_")
                    else:
                        one_output, one_extra1, one_extra2, one_index_ = one.strip().split("_")
                    # one_output, one_extra1,one_extra2, one_index_ = one.strip().split("_")
                    # print(one_index_ )
                    one_index, one_index_extra=one_index_.strip().split("'")
                    dict_read["index_tau"].append(int(one_index))
                    dict_read["Model"].append(one)
                    dict_read["tau"].append(two)

                except Exception as e:
                    print("nothing")




        df_optdepth=pd.DataFrame.from_dict(dict_read, dtype='float')
        df_optdepth.to_pickle(str(path_preprocessing)+"/preprocessed/opticaldepths_"+str(c1)+str(name_)+".pkl")
        df_optdepth=pd.read_pickle(str(path_preprocessing)+"/preprocessed/opticaldepths_"+str(c1)+str(name_)+".pkl")




        counter=0
        data = pd.DataFrame([])
        for c2 in list_count_files2:
            try:


                path=str(path_preprocessing)+"/outputs_"+str(dataset_label)+str(c1)+"/output_"+str(c12)+str(c2)+"/"
                print("path", path)

                # df_read=pd.read_pickle(path + "df"+str(name_)+".pkl")
                df_read=pd.read_pickle(path + "df"+str(name_)+".pkl")

                df_read=df_read[df_read["lambda_um"]<=30.5]
                Flux_Jy=df_read["F_Jy_z"]
                data_=Flux_Jy.apply(pd.Series)

                data = data.append(data_.T)

            except Exception as e:

                print("no profile", e)
                # data= data.append([np.nan] *len(data_.T))

        print("--index data", data_index)
        print("--index F", index_F)
        print(data.head())

        data.index=index_F
        print(data.index)


        data["index"]=index_F#data.index

        df_inputs["index"]=df_inputs["index_other"]
        df_inputs.index=df_inputs["index"] #= index_count_list#df_inputs.index  # df_inputs.index
        df_inputs["index"] = df_inputs.index
        df_inputs = df_inputs.drop(columns={"index_other", "index"})
        print("df_input", df_inputs.head())



        # df_inputs.index=index_count_list
        df_optdepth["index"]=df_optdepth["index_tau"]
        df_optdepth.index=df_optdepth["index"] #= index_count_list#df_optdepth.index  # df_optdepth.index
        df_optdepth["index"] = df_optdepth.index
        # df_optdepth.index=index_count_list
        df_optdepth = df_optdepth.drop(columns={"index_tau","index"})
        print("df_optdepth ",df_optdepth )




        data.index=data["index"]
        data=data.drop(columns={"index"})
        print(data.index)



        df_dusttemp = df_dusttemp_new[df_dusttemp_new.index.isin(data.index)]
        df_dusttemp.to_pickle(str(path_preprocessing) + "/preprocessed/dusttemperatures_" + str(c1) + str(name_) + ".pkl")
        df_dusttemp = pd.read_pickle(
            str(path_preprocessing) + "/preprocessed/dusttemperatures_" + str(c1) + str(name_) + ".pkl")
        df_dusttemp["index"]=df_dusttemp.index

        print(data.index)
        print(df_dusttemp.index)
        df_merge1=pd.merge(data, df_dusttemp, on="index")
        print("len df_merge1", len(df_merge1),df_merge1.tail())

        df_merge2=pd.merge(df_merge1, df_inputs, on="index")
        print("len df_merge2", len(df_merge2), df_merge2.tail())

        df_tot=pd.merge(df_merge2, df_optdepth, on="index")
        print("len df_tot", len(df_tot))
        print(df_tot.tail())
        print(len(z_list_tot))

        # print(len(z_list_value))




        file_name_z_list = str(path_preprocessing)+"/preprocessed/z_list_withoutDrop" + str(name_) + ".pkl"
        open_file = open(file_name_z_list, "wb")
        pickle.dump(z_list_tot, open_file)
        open_file.close()




        print(len(z_list))#z_list#
        print(len(z_list_value))#z_list_tot
        df_tot["z"]=z_list_value
        df_tot.index = data_index
        print(df_tot.index)
        data=df_tot


        print("-----------------------preprocessing is done-----------------------")
        print("--index data", data_index)


        data.to_pickle(str(path_preprocessing)+"/preprocessed/allSEDs_"+str(c1)+str(name_)+".pkl")
        print((str(path_preprocessing)+"/preprocessed/allSEDs_"+str(c1)+str(name_)+".pkl"))
        count_obj = count_obj + 1
        end0 = time.time()
        print("time for preprocessing 1 input:", end0 - start0)
        gc.collect()

    end=time.time()
    print("time for preprocessing 1 input set:", end - start)



def preprocess_concat(path_preprocessing, name_, range1_outputs,range11_outputs, range2_outputs):
    list_count_files1 = ["%03d" % x for x in range(range1_outputs,range11_outputs)]
    # list_count_files111 = ["%03d" % x for x in range(153,182)]
    # list_count_files1111 = ["%03d" % x for x in range(200,253)]
    # list_count_files1=list_count_files111+ list_count_files1111
    print("list_count_files1",list_count_files1)


    appended_data=[]#pd.DataFrame([])

    for c1 in list_count_files1:
        print(c1)
        data = pd.read_pickle(str(path_preprocessing) + "/preprocessed/allSEDs_" + str(c1) + str(name_) + ".pkl")
        print(type(data))
        print(data.head())
        appended_data.append(data)


    appended_data = pd.concat(appended_data)



    appended_data.to_pickle(str(path_preprocessing) + "/preprocessed/allSEDs_"+str(name_)+".pkl")
    gc.collect()
