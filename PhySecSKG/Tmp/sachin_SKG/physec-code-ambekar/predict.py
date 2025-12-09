import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


def avgprofile():
    #read training data
    filename = "mobile_combined.csv"

    enb_in = (np.array(np.loadtxt(filename, delimiter=",", skiprows=1, usecols=[3]))).reshape(-1, 1)
    eu_in = (np.array(np.loadtxt(filename, delimiter=",", skiprows=1, usecols=[4]))).reshape(-1, 1)
    lst_range = np.arange(len(eu_in))
    noise = np.array(np.random.normal(1.55, 5.4, len(eu_in))).reshape(-1, 1)
    noise_enb_in = enb_in - noise
    #normalise training data
    scalar = StandardScaler()
    enb_in_normal = scalar.fit_transform(enb_in)

    scalar = StandardScaler()
    eu_in_normal = scalar.fit_transform(eu_in)



    # calculate the normalized average of enb and ue
    avg_true = []
    for i in range(len(lst_range)):
        avg_true.append((enb_in_normal[i] + eu_in_normal[i]) / 2)

    # test data
    enb_in_test = np.array(
        [-23.3549, -26.1061, -21.5839, -25.4874, -23.8886, -24.4185, -25.72, -28.2422, -25.5367, -29.118, -30.3921,
         -28.9728, -29.0104, -29.4505, -28.2777, -27.3985, -26.6233, -26.611, -27.6393, -26.5664, -27.0786, -27.2013,
         -27.2954, -27.3661, -43.389, -24.451, -22.418, -21.9153, -21.6749, -21.3734, -21.8495, -21.7783, -22.1097,
         -22.0311, -24.6839, -29.7809, -21.9838, -25.1121, -26.8183, -23.1036, -21.9729, -21.0598, -22.6061, -19.6725,
         -23.0478, -23.4679, -22.4803, -20.0462, -20.1589, -19.9641, -20.1128, -20.4847, -20.2154, -24.0841, -24.128,
         -23.3135, -22.9186, -22.8454, -23.2341, -23.3074, -22.8936, -23.5121, -23.6285, -23.7045, -23.5549, -23.859,
         -23.4138, -23.3535, -23.9373, -23.3442, -23.2499, -23.3689, -22.1389, -19.5944, -21.4447, -21.7451, -20.7081,
         -21.1014, -21.4494, -21.3796, -21.0804, -21.8016, -21.6324, -21.308, -21.2998, -21.6363, -21.9686, -21.286,
         -21.6822, -21.699, -21.5768, -21.553, -21.5986, -21.3596, -21.2927, -22.2907, -21.6976, -21.3409, -21.1578,
         -21.8127, -21.5115, -21.1645, -21.4048, -21.0719, -20.9407, -21.0033, -21.044, -20.7611, -21.3699, -22.4285,
         -22.4597, -20.9726, -21.1569, -21.6134, -20.9295, -21.3372, -21.5772, -21.6592, -20.547, -23.7176, -23.2751,
         -23.562, -23.5856, -22.9172, -22.313, -22.1097, -22.049, -22.2981]).reshape((-1, 1))

    avg_true_test = np.array(
        [-25.037734, -21.940134, -20.877784, -22.1213345, -21.4202345, -21.4844345, -24.266234, -23.9048835, -23.138134,
         -24.01745, -24.908885, -24.117984, -23.888534, -24.364434, -23.3760345, -23.082334, -22.686334, -22.663134,
         -23.0454845, -22.5101345, -22.7000845, -23.178884, -23.21704, -23.281484, -30.74334, -21.06485, -20.1476845,
         -20.1655845, -20.416134, -20.050684, -20.442734, -19.581785, -18.941285, -20.225835, -21.3876845, -24.0697345,
         -20.471084, -22.04189, -22.766834, -21.4818835, -21.366334, -21.4226335, -21.2044835, -20.1100345, -21.8153845,
         -22.0111345, -21.76849, -19.742085, -20.004235, -19.946285, -20.019635, -20.1857, -19.6726855, -21.703035,
         -22.3479345, -21.7345345, -21.5937345, -21.5535345, -21.7351345, -21.9029345, -21.494085, -21.8802845,
         -22.1722345, -22.2002345, -22.1553845, -22.3053345, -21.8572345, -21.7794345, -22.2452345, -21.7502345,
         -21.8272845, -21.9263845, -21.0798345, -18.335087, -21.42969, -21.930483, -19.8482855, -19.5225865,
         -19.6031865, -19.383165, -18.973087, -19.6373865, -19.196637, -19.148687, -19.172087, -19.346787, -19.2204375,
         -19.259587, -19.153087, -19.461887, -19.5414865, -19.553865, -19.6244365, -19.298587, -19.4335365, -19.7908365,
         -19.6490865, -19.6079365, -19.59976, -19.7965865, -19.7889865, -19.4724865, -20.1958855, -20.1900855,
         -20.0909355, -20.1479855, -20.1357355, -19.8154355, -20.4347855, -20.8631355, -21.164785, -20.386985,
         -20.1541855, -20.1884355, -20.187605, -19.957786, -20.281455, -20.3696855, -20.662385, -23.952282, -23.451382,
         -23.5005825, -23.4900825, -23.1073325, -22.8481325, -22.8132825, -22.8461825, -23.024432]).reshape(
        (-1, 1))

    ue_in_test = np.array(
        [-26.720568, -17.774168, -20.171668, -18.755269, -18.951869, -18.550369, -22.812468, -19.567567, -20.739568,
         -18.9169, -19.42567, -19.263168, -18.766668, -19.278368, -18.474369, -18.766168, -18.749368, -18.715268,
         -18.451669, -18.453869, -18.321569, -19.156468, -19.13868, -19.196868, -18.09768, -17.6787, -17.877369,
         -18.415869, -19.157368, -18.727968, -19.035968, -17.38527, -15.77287, -18.42057, -18.091469, -18.358569,
         -18.958368, -18.97168, -18.715368, -19.860167, -20.759768, -21.785467, -19.802867, -20.547569, -20.582969,
         -20.554369, -21.05668, -19.43797, -19.84957, -19.92847, -19.92647, -19.8867, -19.129971, -19.32197, -20.567869,
         -20.155569, -20.268869, -20.261669, -20.236169, -20.498469, -20.09457, -20.248469, -20.715969, -20.695969,
         -20.755869, -20.751669, -20.300669, -20.205369, -20.553169, -20.156269, -20.404669, -20.483869, -20.020769,
         -17.075774, -21.41468, -22.115866, -18.988471, -17.943773, -17.756973, -17.38673, -16.865774, -17.473173,
         -16.760874, -16.989374, -17.044374, -17.057274, -16.472275, -17.233174, -16.623974, -17.224774, -17.506173,
         -17.55473, -17.650273, -17.237574, -17.574373, -17.290973, -17.600573, -17.874973, -18.04172, -17.780473,
         -18.066473, -17.780473, -18.986971, -19.308271, -19.241171, -19.292671, -19.227471, -18.869771, -19.499671,
         -19.297771, -19.86987, -19.80137, -19.151471, -18.763471, -19.44571, -18.578372, -18.98571, -19.080171,
         -20.77777, -24.186964, -23.627664, -23.439165, -23.394565, -23.297465, -23.383265, -23.516865, -23.643365,
         -23.750764]).reshape((-1, 1))

    # preprocessing input for model --- train the model with data and label which are enb and avg for enb side and for ue side it will ue and avg
    print("length of enb,ue and avg of test data", len(enb_in_test), len(ue_in_test), len(avg_true_test))

    Y_true = np.array(avg_true)

    X_in = preprocess_data(enb_in_normal)
    X_in_ue = preprocess_data(eu_in_normal)

    ################# model for enb #############
    clf = linear_model.LinearRegression()

    #################### model for ue ##############
    clf_ue = linear_model.LinearRegression()

    ############ model training - X_in = enb or ue based on machine ; Y_true = avg of enb and ue
    clf.fit(X_in, Y_true)

    #Save the model
    # modelname = 'finalized_model.sav'
    # pickle.dump(clf, open(modelname, 'wb'))

    # ue model training
    clf_ue.fit(X_in_ue, Y_true)

##################################################################################################################
    # Training data score enb
    print("Training acc - enb", clf.score(X_in, Y_true))

    # Training data score ue
    print("Training acc - eu", clf_ue.score(X_in_ue, Y_true))

    ########preprocessing the  test date , enb_in_test = new data for testing -################
    # Normalizing
    scalar = StandardScaler()
    enb_in_test_normal = scalar.fit_transform(enb_in_test)

    scalar = StandardScaler()
    ue_in_test_normal = scalar.fit_transform(ue_in_test)

    # polynomial feautres
    X_pred = preprocess_data(enb_in_test_normal)
    # ueside
    X_ue_pred = preprocess_data(ue_in_test_normal)

    # predicting avg from enb or UE test values ( values not used for training)
    avg_pred = clf.predict(X_pred)
    # ueside
    avg_ue_pred = clf_ue.predict(X_ue_pred)

    # detransform the normalized data
    #scalar = StandardScaler()
    avg_pred_demormalized = scalar.inverse_transform(avg_pred)
    avg_ue_pred_denormalized = scalar.inverse_transform(avg_ue_pred)

    x_axis = np.arange(len(avg_true_test))
    # plt.scatter(x_axis, avg_true_test, s = 1, color='green')
    avg_true_test_normal = data_normalize(avg_true_test)
    # plt.scatter(x_axis, avg_true_test_normal, s=1, color='green')
    #plt.plot(avg_true_test, color='green')
    #plt.plot(avg_pred_demormalized, color='blue')

    #plt.legend()

    # ueside
    # plt.plot(avg_true_test, color='green')
    #plt.plot(X_in,color='green')
    #plt.plot(X_in_ue)
    #plt.plot(avg_ue_pred_denormalized, color='red')
    #plt.show()

    return avg_pred_demormalized, avg_ue_pred_denormalized


def preprocess_data(data_in_normal):
    X_in = PolynomialFeatures(degree=4).fit_transform(data_in_normal)
    return X_in


def data_normalize(data_in):
    scalar = StandardScaler()
    data_in_normal = scalar.fit_transform(data_in)
    return data_in_normal


