import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

from channelProfile import ChannelProfile


class Reciprocityenhancer:

    def cdt(self,list):
        data_path = os.path.join(os.path.dirname(__file__), "dctmatrix.csv")
        cdt_matrix: np.ndarray = np.genfromtxt(data_path, delimiter=',')
        profile = np.asmatrix(list)
        cdt_matrix = np.asmatrix(cdt_matrix)
        m3 = np.dot(profile, cdt_matrix)
        np.savetxt("dctkey.csv", m3, delimiter=",")
        return m3

    def avgprofile(self, filename, testfilename, col_number1, col_number2):
        ###### read training data##########

        enb_in = (np.array(np.loadtxt(filename, delimiter=",", skiprows=1, usecols=[col_number1]))).reshape(-1, 1)
        eu_in = (np.array(np.loadtxt(filename, delimiter=",", skiprows=1, usecols=[col_number2]))).reshape(-1, 1)
        lst_range = np.arange(len(eu_in))
        #noise = np.array(np.random.normal(1.55, 5.4, len(eu_in))).reshape(-1, 1)
        #noise_enb_in = enb_in - noise

        ########## normalise training data##########
        scalar = StandardScaler()
        enb_in_normal = scalar.fit_transform(enb_in)

        scalar = StandardScaler()
        eu_in_normal = scalar.fit_transform(eu_in)

        ### calculate the normalized average of enb and ue#####
        avg_true = []
        for i in range(len(lst_range)):
            avg_true.append((enb_in_normal[i] + eu_in_normal[i]) / 2)

        ###########test data###############
        #profile: ChannelProfile = ChannelProfile(testfilename)
        enb_in_test = (np.array(np.loadtxt(testfilename, delimiter=",", skiprows=1, max_rows=128, usecols=[col_number1]))).reshape(-1, 1)

        ue_in_test = (np.array(np.loadtxt(testfilename, delimiter=",", skiprows=1, max_rows=128, usecols=[col_number2]))).reshape(-1, 1)

        #######calculate avg of test data  ##################
        avg_true_test = []
        test_lst_range = np.arange(len(ue_in_test))
        for i in range(len(test_lst_range)):
            avg_true_test.append((enb_in_test[i] + ue_in_test[i]) / 2)


        # preprocessing input for model --- train the model with data and label which are enb and avg for enb side and
        # for ue side it will ue and avg
        #print("length of enb,ue and avg of test data", len(enb_in_test), len(ue_in_test), len(avg_true_test))

        Y_true = np.array(avg_true)

        X_in = self.preprocess_data(enb_in_normal)
        X_in_ue = self.preprocess_data(eu_in_normal)

        ################# model for enb #############
        clf = linear_model.LinearRegression()

        #################### model for ue ##############
        clf_ue = linear_model.LinearRegression()

        ############ model training - X_in = enb or ue based on machine ; Y_true = avg of enb and ue
        clf.fit(X_in, Y_true)

        # Save the model
        # modelname = 'finalized_model.sav'
        # pickle.dump(clf, open(modelname, 'wb'))

        # ue model training
        clf_ue.fit(X_in_ue, Y_true)

        ################################################################################################################
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

        # polynomial features
        X_pred = self.preprocess_data(enb_in_test_normal)
        # ueside
        X_ue_pred = self.preprocess_data(ue_in_test_normal)

        # predicting avg from enb or UE test values ( values not used for training)
        avg_pred = clf.predict(X_pred)
        # ueside
        avg_ue_pred = clf_ue.predict(X_ue_pred)

        # detransform the normalized data
        avg_pred_demormalized = scalar.inverse_transform(avg_pred)
        avg_ue_pred_denormalized = scalar.inverse_transform(avg_ue_pred)

        x_axis = np.arange(len(avg_true_test))
        # plt.scatter(x_axis, avg_true_test, s = 1, color='green')
        avg_true_test_normal = self.data_normalize(avg_true_test)
        plt.plot(avg_true_test_normal, color='black')
        plt.plot(avg_ue_pred_denormalized, color='green')
        plt.plot(avg_pred_demormalized, color='blue')

        # plt.legend()

        # ueside
        # plt.plot(avg_true_test, color='green')
        # plt.plot(X_in,color='green')
        # plt.plot(X_in_ue)
        # plt.plot(avg_ue_pred_denormalized, color='red')
        #plt.show()

        return avg_pred_demormalized, avg_ue_pred_denormalized

    def preprocess_data(self, data_in_normal):
        X_in = PolynomialFeatures(degree=2).fit_transform(data_in_normal)
        return X_in

    def data_normalize(self, data_in):
        scalar = StandardScaler()
        data_in_normal = scalar.fit_transform(data_in)
        return data_in_normal

