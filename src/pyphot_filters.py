
import pandas as pd
import numpy as np
from pyphot import Filter, unit



def flux_specific_filter(filter_wave_range, filter_transmit, wavelength_source, flux_source, filter_name):

    _band = Filter(filter_wave_range, filter_transmit, name=filter_name, dtype='photon', unit='Angstrom')
    wave_ = np.array(wavelength_source).astype(float)
    flux_ = np.array(flux_source).astype(float)
    photometric_value_flux = _band.get_flux(wave_, flux_)
    corresponding_wavelength = int(''.join(filter(str.isdigit,
                                                  filter_name))) * 100

    return photometric_value_flux, corresponding_wavelength



def set_filters(data,flux,df_read_wl,path_filterbands, z_list, name_, list_filters,list_filter_name, filter_set_name):


    df_lambda = {}
    df_lambda["lambda_um"] = []

    for i_ind in range(len(data)):
        wl = df_read_wl["lambda_um"] * (1 + z_list[i_ind])
        df_lambda["lambda_um"].append(wl)
    df_wl_um = df_lambda["lambda_um"]


    df_lambda = pd.DataFrame.from_dict(df_lambda)
    df_lambda.index=data.index

    df_wavelength_um = df_lambda["lambda_um"].apply(pd.Series)


    wavelength_um = df_wavelength_um
    wavelength_nonum=wavelength_um

    wavelength = wavelength_um * 1e+4  # np.array(list(range(6, 47319)))

    print("*********************wavelength_um.loc[object]*****************************", *wavelength_um.iloc[0])
    print("*********************wavelength_um.loc[object]*****************************", *wavelength_um.iloc[1])




    df_flux_add={}
    df_flux_add["flux"]=[]

    for i_data in data.index:
        f =  flux.loc[i_data]#[::-1]
        df_flux_add["flux"].append(f)#[::-1])

    df_flux_add=pd.DataFrame.from_dict(df_flux_add)
    df_flux_add=df_flux_add["flux"].apply(pd.Series)

    df_flux_add.index = data.index

    flux=df_flux_add


    survey_name="JWST"#"SDSS" ##SDSS + HST
    # ####in photometry_filters you need to change the *1e+4 to *1e+1 from nn to A units

    df_bands={}
    len_data = len(data)
    obj_list = list(data.index)
    print("obj_list", obj_list)

    for objects in obj_list:


        df_flux_objetcs = pd.DataFrame(wavelength_nonum.loc[objects])
        df_flux_objetcs["flux"] = flux.loc[objects]

        df_flux_objetcs = df_flux_objetcs[df_flux_objetcs["flux"] != 0]
        df_flux_objetcs = df_flux_objetcs.dropna()




        df_flux_objetcs=df_flux_objetcs[df_flux_objetcs[objects]<=30.5]
        df_flux_objetcs["lambda"]= df_flux_objetcs[objects] * 1e+4






        corresponding_wavelength_list=[]
        df_bands[objects] = []

        for i_filter in range(len(list_filters)):
            filter=list_filters[i_filter]
            Filter=pd.read_table("bands/"+str(filter),header=None, delimiter=r'\s+')

            filter_wl_range=Filter[0]
            filter_wave_range=filter_wl_range * unit['AA']

            filter_transmit=Filter[1]
            wavelength_source=df_flux_objetcs["lambda"] * unit['AA']
            flux_source=df_flux_objetcs['flux']


            filter_name=list_filter_name[i_filter]
            photometric_value_flux, corresponding_wavelength=flux_specific_filter(filter_wave_range, filter_transmit,
                                                                                  wavelength_source, flux_source, filter_name)


            flux_band=df_flux_objetcs[df_flux_objetcs["lambda"]>filter_wave_range[0]]
            flux_band=flux_band[flux_band["lambda"]<filter_wave_range[len(filter_wave_range)-1]]

            print("check photo values", photometric_value_flux)
            df_bands[objects].append(np.log10(photometric_value_flux))
            corresponding_wavelength_list.append(corresponding_wavelength * 1e-4)




    df_bands_=pd.DataFrame.from_dict(df_bands).T
    df_bands_.to_pickle(str(path_filterbands)+"/logBandsFlux-"+str(filter_set_name)+"_"+str(name_)+".pkl")

