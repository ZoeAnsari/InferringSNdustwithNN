import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.cosmology import FlatLambdaCDM , z_at_value
import astropy.units as u
import pickle
from numpy import inf
import seaborn as sns


def plot_sample(data, flux,  df_read_wl, otherfeatures_data, path_filterbands,
                path_preprocessing, name_, title, z_list_withoutDrop, z_fixed, list_plt):
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)


    df_flux_add = {}
    df_flux_add["flux"] = []

    for i_data in flux.index:
        f = flux.loc[i_data]  # [::-1]
        df_flux_add["flux"].append(f)  # [::-1])

    df_flux_add = pd.DataFrame.from_dict(df_flux_add)
    df_flux_add = df_flux_add["flux"].apply(pd.Series)
    df_flux_add.index=data.index
    flux = df_flux_add



    z_list = z_list_withoutDrop
    D_list = cosmo.luminosity_distance(z_list).astype("float")

    # D_list=D_list.astype("float")
    print("D_list", D_list)



    #####plot out


    df_lambda = {}
    df_lambda["lambda_um"] = []

    for i_ind in range(len(data)):
        wl = df_read_wl["lambda_um"] * (1 + z_list[i_ind])
        df_lambda["lambda_um"].append(wl)
    df_wl_um = df_lambda["lambda_um"]

    df_lambda = pd.DataFrame.from_dict(df_lambda)
    df_wavelength_um = df_lambda["lambda_um"].apply(pd.Series)
    df_wavelength_um.index=data.index

    wavelength_um = df_wavelength_um
    wavelength_nonum = wavelength_um

    wavelength = wavelength_um * 1e+4



    plt.figure(figsize=(7,5))

    for i_pl in list_plt:
        try:

            print("i_pl", i_pl)

            label_ = "sil:{:.0f}".format(otherfeatures_data["species"][i_pl])+ \
                     "Td:{:.0f}".format(otherfeatures_data["tempDust"][i_pl])+ \
                     "Md:{:.2e}".format(otherfeatures_data["massDust"][i_pl]) + \
                     "Rsn:{:.2e}".format(otherfeatures_data["radiusSN"][i_pl])+ \
                     "Tsn:{:.0f}".format(otherfeatures_data["tempSN"][i_pl]) + \
                     "_GS:{:.2e}".format(otherfeatures_data['grainSize'][i_pl]) + \
                     "_z:{:.1e}".format(otherfeatures_data['z'][i_pl])


            df_flux_9 = pd.DataFrame(wavelength_nonum.loc[i_pl])
            df_flux_9["flux"] = flux.loc[i_pl]
            df_flux_9 = df_flux_9[df_flux_9["flux"] != 0]
            df_flux_9 = df_flux_9.dropna()


            Mags = (-2.5 * (np.log10(df_flux_9["flux"]) - 23 )) - 48.6
            print(Mags)

            plt.scatter(df_flux_9[i_pl], Mags, s=3, label=label_)  # "lambda_um"

        except Exception as e:
            print(e)
            print("--")
            # i_pl=i_pl+1

    plt.xlabel(r"$\lambda (\mu m)$")
    plt.ylabel("Mag AB")
    plt.gca().invert_yaxis()
    plt.grid(True, which="both", ls="-")

    plt.title(str(title))
    plt.legend()
    # plt.legend(bbox_to_anchor=(0.4, 1.1))#, loc='upper left')
    plt.savefig(str(path_filterbands) +
                "/SEDs_Mag"+str(z_fixed)+".png", bbox_inchhes='tight')
    plt.xscale("log")
    plt.savefig(str(path_filterbands)+
                "/SEDs_Mag_log"+str(z_fixed)+".png")#, bbox_inchhes='tight')






    plt.figure(figsize=(7,5))
    for i_pl in list_plt:


        try:
            # const_mJy_ = const_mJy[i_pl]
            print("i_pllog_mjy", i_pl)
            label_ = "sil:{:.0f}".format(otherfeatures_data["species"][i_pl])+ \
                     "Td:{:.0f}".format(otherfeatures_data["tempDust"][i_pl])+ \
                     "Md:{:.2e}".format(otherfeatures_data["massDust"][i_pl]) + \
                     "Rsn:{:.2e}".format(otherfeatures_data["radiusSN"][i_pl])+ \
                     "Tsn:{:.0f}".format(otherfeatures_data["tempSN"][i_pl]) + \
                     "_GS:{:.2e}".format(otherfeatures_data['grainSize'][i_pl]) + \
                     "_z:{:.1e}".format(otherfeatures_data['z'][i_pl])

            df_flux_9 = pd.DataFrame(wavelength_nonum.loc[i_pl])
            df_flux_9["flux"] = flux.loc[i_pl]
            df_flux_9 = df_flux_9[df_flux_9["flux"] != 0]
            df_flux_9 = df_flux_9.dropna()



            plt.scatter(df_flux_9[i_pl], (df_flux_9["flux"]), s=3, label=label_)


        except:
            print("--")
            i_pl = i_pl + 1

    plt.xlabel(r"$\lambda (\mu m)$")
    plt.ylabel("Flux(mJy)")
    plt.grid()
    # plt.xlim(1e-1,1e+2)
    plt.xlim(1e-6,1e+0)
    plt.xscale("log")
    plt.grid(True, which="both", ls="-")
    plt.title(str(title))
    plt.yscale("log")
    plt.legend()

    plt.savefig(str(path_filterbands)+
                "/SEDs_mJy_log"+str(z_fixed)+".png", bbox_inchhes='tight')


def plot_sample_bands(data, flux,  df_read_wl, otherfeatures_data, path_filterbands,
                      path_preprocessing, name_, title, z_list_withoutDrop, z_upper_limit, list_plt,filter_sets_name,list_wls):
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

    df_flux_add = {}
    df_flux_add["flux"] = []

    for i_data in flux.index:
        f = flux.loc[i_data]  # [::-1]
        df_flux_add["flux"].append(f)  # [::-1])

    df_flux_add = pd.DataFrame.from_dict(df_flux_add)
    df_flux_add = df_flux_add["flux"].apply(pd.Series)
    df_flux_add.index=data.index
    flux = df_flux_add


    z_list = z_list_withoutDrop#otherfeatures_data["z"]
    D_list = cosmo.luminosity_distance(z_list).astype("float")  # [45.6] * 20#

    # D_list=D_list.astype("float")
    print("D_list", D_list)

    # LDS_list = np.array(D_list) * 3.086e+18 * 1e+6  # luminosity distance from Mpc to cm

    plt.figure(figsize=(7,5))
    counter=0
    colors_list=['darkred', 'blue']
    for i_name in filter_sets_name:
        df_bands= pd.read_pickle(str(path_filterbands)+"/logBandsFlux-"+str(i_name)+"_"+str(name_)+".pkl")
        # df_bands["index"] =  df_bands.index.astype(int)
        filters_clWL=list_wls[counter]





        df_lambda = {}
        df_lambda["lambda_um"] = []
        for i_ind in range(len(data)):
            wl = df_read_wl["lambda_um"] * (1 + z_list[i_ind])
            df_lambda["lambda_um"].append(wl)
        df_wl_um = df_lambda["lambda_um"]


        df_lambda = pd.DataFrame.from_dict(df_lambda)
        df_wavelength_um = df_lambda["lambda_um"].apply(pd.Series)
        df_wavelength_um.index=data.index
        wavelength_um = df_wavelength_um
        wavelength_nonum = wavelength_um
        wavelength = wavelength_um * 1e+4  # np.array(list(range(6, 47319)))


        for i_pl in list_plt:
            try:

                print("i_pl", i_pl)

                label_ = "sil:{:.0f}".format(otherfeatures_data["species"][i_pl])+ \
                         "Td:{:.0f}".format(otherfeatures_data["tempDust"][i_pl])+ \
                         "Md:{:.2e}".format(otherfeatures_data["massDust"][i_pl]) + \
                         "Rsn:{:.2e}".format(otherfeatures_data["radiusSN"][i_pl])+ \
                         "Tsn:{:.0f}".format(otherfeatures_data["tempSN"][i_pl]) + \
                         "_GS:{:.2e}".format(otherfeatures_data['grainSize'][i_pl]) + \
                         "_z:{:.1e}".format(otherfeatures_data['z'][i_pl])


                df_flux_9 = pd.DataFrame(wavelength_nonum.loc[i_pl])
                df_flux_9["flux"] = flux.loc[i_pl]
                df_flux_9 = df_flux_9[df_flux_9["flux"] != 0]
                df_flux_9 = df_flux_9.dropna()


                Mags = (-2.5 * (np.log10(df_flux_9["flux"]) - 23 )) - 48.6
                if counter == 0:
                    plt.scatter(df_flux_9[i_pl], Mags, s=3, label=label_)

                plt.scatter(filters_clWL, ((-2.5 * ((df_bands.loc[[i_pl],:]) - 23 )) - 48.6),
                            s=10, c=colors_list[counter])

            except Exception as e:
                print(e)
                print("No JWST band, due to lack of spectra coverage")
                # i_pl=i_pl+1
        counter=counter+1

    plt.xlabel(r"$\lambda (\mu m)$")
    plt.ylabel("Mag AB")
    plt.gca().invert_yaxis()
    plt.grid(True, which="both", ls="-")
    plt.title(str(title))
    plt.legend()

    plt.savefig(str(path_filterbands)+
                "/bandpass_filters_continuum_Mag"+str(name_)+".png")

    plt.xscale("log")
    plt.savefig(str(path_filterbands)+
                "/bandpass_filters_continuum_Mag_log"+str(name_)+".png")


    ###########-------

    plt.figure(figsize=(7,5))
    counter=0
    colors_list=['darkred', 'blue']
    for i_name in filter_sets_name:
        df_bands= pd.read_pickle(str(path_filterbands)+"/logBandsFlux-"+str(i_name)+"_"+str(name_)+".pkl")
        print("df_bands", df_bands, len(df_bands.columns))
        # df_bands["index"] =  df_bands.index.astype(int)
        filters_clWL=list_wls[counter]
        print(filters_clWL,len(filters_clWL))




        df_lambda = {}
        df_lambda["lambda_um"] = []
        for i_ind in range(len(data)):
            wl = df_read_wl["lambda_um"] * (1 + z_list[i_ind])
            df_lambda["lambda_um"].append(wl)
        df_wl_um = df_lambda["lambda_um"]


        df_lambda = pd.DataFrame.from_dict(df_lambda)
        df_wavelength_um = df_lambda["lambda_um"].apply(pd.Series)
        df_wavelength_um.index=data.index
        wavelength_um = df_wavelength_um
        wavelength_nonum = wavelength_um
        wavelength = wavelength_um * 1e+4  # np.array(list(range(6, 47319)))
        print("check size", df_bands.loc[[0],:])

        for i_pl in list_plt:
            try:

                print("i_pl", i_pl)

                label_ = "sil:{:.0f}".format(otherfeatures_data["species"][i_pl])+ \
                         "Td:{:.0f}".format(otherfeatures_data["tempDust"][i_pl])+ \
                         "Md:{:.2e}".format(otherfeatures_data["massDust"][i_pl]) + \
                         "Rsn:{:.2e}".format(otherfeatures_data["radiusSN"][i_pl])+ \
                         "Tsn:{:.0f}".format(otherfeatures_data["tempSN"][i_pl]) + \
                         "_GS:{:.2e}".format(otherfeatures_data['grainSize'][i_pl]) + \
                         "_z:{:.1e}".format(otherfeatures_data['z'][i_pl])


                df_flux_9 = pd.DataFrame(wavelength_nonum.loc[i_pl])
                df_flux_9["flux"] = flux.loc[i_pl]
                df_flux_9 = df_flux_9[df_flux_9["flux"] != 0]
                df_flux_9 = df_flux_9.dropna()



                if counter == 0:
                    plt.scatter(df_flux_9[i_pl], (df_flux_9["flux"]), s=3, label=label_)  # "lambda_um"

                plt.scatter(filters_clWL, ((10 ** (df_bands.loc[[i_pl],:]) )),
                            s=10, c=colors_list[counter])

            except Exception as e:
                print(e)
                print("No JWST band, due to lack of spectra coverage")
                # i_pl=i_pl+1
        counter=counter+1

    plt.xlabel(r"$\lambda (\mu m)$")
    plt.ylabel("Flux(mJy)")
    plt.grid()
    plt.ylim(1e-6,1e+0)
    plt.xscale("log")
    plt.grid(True, which="both", ls="-")
    plt.title(str(title))
    plt.yscale("log")
    plt.legend()

    plt.savefig(str(path_filterbands)+
                "/bandpass_filters_continuum_mJy_log"+str(name_)+".png")




