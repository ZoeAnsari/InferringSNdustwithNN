import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def add_random_noise(X_data, path_preprocessing):



    all_phot= int (len(X_data) * len(X_data.columns))
    plt.figure()


    # random_uniform_error = np.random.uniform(-0.1, 0.1, all_phot)
    mu=0
    sigma=0.1
    random_uniform_error = np.random.normal(mu, sigma, all_phot)
    df_errors=pd.DataFrame(np.array(random_uniform_error).reshape(len(X_data.index), len(X_data.columns)))
    # df_errors=df_errors.rename(columns={0: "F070W_NIR", 1:"F090W_NIR", 2:"F115W_NIR", 3:"F140M_NIR",4:"F150W_NIR",
    #                                     5:"F162M_NIR", 6:"F164N_NIR", 7:"F182M_NIR", 8:"F187N_NIR", 9:"F200W_NIR",
    #                                     10:"F210M_NIR", 11:"F212N_NIR", 12:"F250M_NIR", 13:"F277W_NIR", 14:"F300M_NIR",
    #                                     15:"F322W2_NIR", 16:"F323N_NIR", 17:"F335M_NIR", 18:"F356W_NIR", 19:"F360M_NIR",
    #                                     20:"F405N_NIR", 21:"F410M_NIR", 22:"F430M_NIR", 23:"F444W_NIR", 24:"F460M_NIR",
    #                                     25:"F466N_NIR", 26:"F470N_NIR", 27:"F480M_NIR", 28:"F560W_MIRI", 29:"F770W_MIRI",
    #                                     30:"F1000W_MIRI", 31:"F1130W_MIRI", 32:"F1280W_MIRI", 33:"F1500W_MIRI", 34:"F1800W_MIRI",
    #                                     35:"F2100W_MIRI",36:"F2550W_MIRI"})
    wl=[0.7, 0.9, 1.15, 1.4, 1.5,
        1.62, 1.64, 1.82, 1.87, 2,
        2.1, 2.12, 2.5, 2.77, 3,
        3.22, 3.23, 3.35, 3.56, 3.6,
        4.05, 4.1, 4.3, 4.44, 4.6,
        4.66, 4.7, 4.8, 5.6, 7.7,
        10, 11.3, 12.8, 15, 18,
        21, 25.5]

    df_X_Mags=X_data + df_errors
    print(len(df_errors.loc[0]))
    print(len(wl))



    df_X_Mags.to_pickle(str(path_preprocessing)+"/preprocessed_mags.pkl")
    df_errors.to_pickle(str(path_preprocessing)+"/noise.pkl")

    # for i in range(2):
    #
    #     plt.scatter(wl,df_X_Mags.loc[i], label='with error', s=5)
    #     plt.scatter(wl,X_data.loc[i], label='Simulated', s=5)
    #     # plt.errorbar(wl,X_data.loc[i],yerr=df_errors.loc[i],fmt='.')#,label='Simulated')
    #
    # plt.legend()
    # plt.ylim(30,10)
    # plt.gca().invert_yaxis()
    # plt.savefig(str(path_preprocessing)+"/Noise/check_error_imp.png")
