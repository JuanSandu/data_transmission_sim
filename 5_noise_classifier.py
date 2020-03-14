"""
    Support Vector Machine for noise classification.
    This code has been developed by Juan Sandubete Lopez.

"""

from sklearn import svm
from sklearn.model_selection import train_test_split
from scipy import sparse
import util_functions as utils
import numpy as np
import pandas as pd
import stats_functions as stfc
from joblib import dump, load
import os
from ast import literal_eval


class SVM_classifier:
    def __init__(self):
        # Do the things you need here for the simulation
        self.module_name = "svm_classifier"
        self.separate_input_signals_df = False
        self.signal_pieces_n = 10
        self.svm_clf = svm.SVC()

    def load_data(self):
        print("Data loading.")
        # Load the data from the CSV files
        #signals_df = utils.get_original_dataframe("noisy_signals_split")
        signals_df = pd.read_csv("data/original_data/noisy_signals_split.csv",
                                 index_col=[0])
        if self.separate_input_signals_df:
            exper_list = []
            for exper in signals_df["experiment"]:
                if exper not in exper_list:
                    exper_list.append(exper)

            # exper_len = len(signals_df["experiment"] == exper_list[0])
            exper_len = len(signals_df.iloc[np.where(signals_df["experiment"]
                                            == exper_list[0])])
            sig_piece_len = int(exper_len / self.signal_pieces_n)
            print("Signal len: {}, Exper len: {}, Piece len: {}"
                  .format(len(signals_df), exper_len, sig_piece_len))

            output_df = pd.DataFrame()
            print("Split process started.")

            for i in range(int(len(signals_df)/sig_piece_len)):
                print("Splitting signal piece {}...".format(i))
                ax_list = [list(signals_df["ax"].iloc[i*sig_piece_len:
                                       (i + 1) * sig_piece_len])]
                class_val = signals_df["class"].iloc[i * sig_piece_len]
                exper_val = signals_df["experiment"].iloc[i * sig_piece_len]
                d = {'ax': ax_list, 'class': class_val, 'experiment': exper_val}
                output_df = output_df.append(pd.DataFrame(data=d))
            # output_df.to_csv("noisy_signals_split2.csv")
            print("Split process finished.")
            return output_df
        return signals_df.reset_index(drop=True)

    def preprocessing(self, orig_signals_df):
        print("Preprocessing started.")
        y_orig = orig_signals_df["class"].copy()
        orig_signals_df.drop(columns=["class"], inplace=True)
        signals_df, _, y_df, _ = train_test_split(orig_signals_df, y_orig,
                                                test_size=0.85, random_state=42)
        prep_df = pd.DataFrame()
        sig_pow = []
        autocorr_sum = []
        values_over_5v = []
        values_under_3v = []
        prep_y = []

        ten_pctg = len(signals_df)/100.0
        pctg_cont = 1
        for i in range(len(signals_df)):
            if i > pctg_cont * ten_pctg:
                print("Preprocessing at {}...".format(pctg_cont / ten_pctg))
                pctg_cont += 1
            signal = literal_eval(signals_df.iloc[i]['ax'])
            sig_pow.append(stfc.get_sig_power(signal))
            autocorr_sum.append(sum(stfc.sig_autocorr(signal)))
            values_over_5v.append(stfc.get_values_over_thld(signal, 6.0))
            values_under_3v.append(stfc.get_values_under_thld(signal, 6.0))
            prep_y.append(y_df.iloc[i])

        prep_df['sig_pow'] = sig_pow
        prep_df['autocorr_sum'] = autocorr_sum
        prep_df['values_over_5v'] = values_over_5v
        prep_df['values_under_3v'] = values_under_3v
        prep_y_df = pd.DataFrame(data={"class": prep_y})

        X_train, X_test, y_train, y_test = train_test_split(prep_df,
                        prep_y_df, test_size=0.33, random_state=42)
        print("Preprocessing finished.")
        # prep_df.to_csv("data/original_data/preproc_accel_dataset.csv")
        # signals_df.to_csv("data/original_data/reduced_accel_dataset.csv")
        return X_train, X_test, y_train, y_test

    def train_model(self, X, y):
        print("Model under training.")
        # Train the model
        self.svm_clf.fit(X, y)
        print("Training finished.")

    def save_model(self):
        cont = 1
        file_path = 'data/models/model' + str(cont) + '.joblib'
        while os.path.isfile('filename.txt'):
            cont += 1
            file_path = 'data/models/model' + str(cont) + '.joblib'
        dump(self.svm_clf, file_path)

    def load_model(self, model_id):
        file_path = 'data/models/model' + str(model_id) + '.joblib'
        self.svm_clf = load(file_path)

    def test_model(self, X, y):
        answer = 'n'
        answer = raw_input("Test the model? [y/n]: ")
        while answer == 'y':
            indx = X.sample().index.values[0]
            try:
                print("Prediction: {}".format(self.svm_clf.predict([X.iloc[indx]])))
                print("The real answer was: {}".format(y.iloc[indx].values))
            except:
                print("Index error. Please, try again.")
            answer = raw_input("Test the model again? [y/n]: ")

    def run_model(self):
        signals_df = self.load_data()
        X_train, X_test, y_train, y_test = self.preprocessing(signals_df)
        self.train_model(X_train, y_train)
        self.save_model()
        self.test_model(X_test, y_test)


svm_clf = SVM_classifier()
svm_clf.run_model()
