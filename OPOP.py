import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
from tqdm import trange


class Opop():
    def __init__(self, train_set):
        self.train_set = self.data_processor(train_set)
        self.index = self.train_set.index
        self.n_echo = 100
        self.learning_rate = 0.1
        self.A_mod = 0.5
        self.A_results = None
        self.w = tf.Variable(initial_value=[[0.5],[0.5],[0.5]])
        self.alpha = tf.Variable(initial_value=[[0.5]])
        self.bias = 0.5
        self.root_path = None


    def data_processor(self, data):
        return pd.DataFrame(data.iloc[:, 1:].values, index=data.iloc[:, 0])

    def standardization(self,data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma

    def model(self, echo=200):
        S = self.ARIMA_train().values
        #S = pd.read_csv(r'C:\Users\SXF-Admin\Desktop\分析报告\res\example2.csv',encoding='ANSI')
        #S = S.iloc[:,-1].values
        X_set = self.train_set.iloc[:, :-2]
        bias_df = pd.DataFrame(np.ones([np.shape(X_set)[0], 1]),index=X_set.index)
        X_set = pd.concat([X_set, bias_df], axis=1)
        Y_set = self.train_set.iloc[:, -1]
        Z_set = self.train_set.iloc[:, -2]
        # shift to numpy and T
        S = np.array(S)
        S = S.reshape(S.shape[0], 1)
        Y_set = np.array(Y_set)
        Y_set = Y_set.reshape(Y_set.shape[0],1)
        Z_set = np.array(Z_set)
        Z_set = Z_set.reshape(Z_set.shape[0],1)

        # shift to tf obj
        m, n = X_set.shape
        self.w = tf.Variable(tf.random.uniform([n, 1], -1.0, 1.0), name="w")
        self.alpha = tf.Variable(tf.random.uniform([1, 1], 0.0, 1.0), name="alpha")
        S = tf.constant(S, dtype=tf.float32)
        X_set = tf.constant(X_set, dtype=tf.float32)
        Y_set = tf.constant(Y_set, dtype=tf.float32)
        Z_set = tf.constant(Z_set, dtype=tf.float32)
        optimizer = tf.keras.optimizers.Adadelta(0.1)
        for i in range(echo):
            with tf.GradientTape(persistent=True) as t:
                system_predict = self.alpha * tf.add(S, tf.matmul(X_set, self.w))
                human_predict = (1 - self.alpha) * Z_set
                y_predict = tf.add(system_predict, human_predict)
                Loss_obj = tf.keras.losses.KLDivergence()
                Loss = Loss_obj(Y_set, y_predict)
            gradients = t.gradient(target=Loss, sources=[self.w, self.alpha])
            optimizer.apply_gradients(zip(gradients, [self.w, self.alpha]))
        print(f'the pras are {[self.w,self.alpha]}')

    def predict(self, data, plot_result=False):
        #data should be a df
        data = self.data_processor(data)
        a_data = data.iloc[:,-1]
        S = self.ARIMA_predict([1,len(a_data)]).values
        S2 = S
        #A = sm.tsa.statespace.SARIMAX(a_data.values, trend='n', order=pra, seasonal_order=(1, 1, 1, 12))
        #results = A.fit()
        #S = results.predict(1,len(a_data))
        #S shift to numpy and T
        S = np.array(S)
        S = S.reshape(S.shape[0],1)
        X_set = data.iloc[:, :-2]
        bias_df = pd.DataFrame(np.ones([np.shape(X_set)[0], 1]), index=X_set.index)
        X_set = pd.concat([X_set, bias_df], axis=1)
        #z set shift to numpy and transpose
        Z_set = data.iloc[:, -2].values
        Z_set = Z_set.reshape([39,1])
        #shift to tf obj
        S = tf.constant(S, dtype=tf.float32)
        X_set = tf.constant(X_set, dtype=tf.float32)
        Z_set = tf.constant(Z_set, dtype=tf.float32)
        #compute the result
        system_predict = self.alpha * tf.add(S, tf.matmul(X_set, self.w))
        human_predict = (1 - self.alpha) * Z_set
        y_predict = tf.add(system_predict, human_predict)
        #print(y_predict.numpy)
        total = pd.Series(data.iloc[:,-1].values, index=data.index)
        if plot_result == True:
            f1 = plt.figure(figsize=(16, 8))
            plt.grid()
            plt.xlabel("Date")
            plt.ylabel("Order Quantity")
            plt.title("aBOS-1000-D600")
            plt.plot(total, marker='^', color='k')
            plt.plot(y_predict, marker='o', color='b')
            plt.plot(S2, marker='o', color='g')
            plt.legend(['Real data','Adjusted predict','ARIMA predict'])
            plt.xticks(rotation=90)
            plt.show()
    def ARIMA_train(self, training_set_rate=0.7, plot_result=False):

        d1 = self.train_set.iloc[:,-1]
        train_set_index = round(training_set_rate*len(d1))
        total = pd.Series(d1.values, index=d1.index)
        train_set = pd.Series(d1.iloc[:train_set_index].values, index=list(d1.index)[:train_set_index])
        test_set = pd.Series(d1.iloc[train_set_index:].values, index=list(d1.index)[train_set_index:])
        # ARIMA-S
        new_op = Optimizer(train_set, test_set)
        best_p_list = new_op.grid_search(vrange=[0, 3])
        self.A_mod = sm.tsa.statespace.SARIMAX(train_set.values, trend='n', order=best_p_list, seasonal_order=(1, 1, 1, 12))
        self.A_results = self.A_mod.fit()
        start_po = len(train_set) + 1
        stop_po = start_po + len(test_set) - 1
        #predict = pd.Series(np.maximum(A_results.predict(start=start_po, end=stop_po, dynamic=True), 0),index=test_set.index)
        #fit_result = pd.Series(np.maximum(A_results.predict(1, start_po - 1), 0), index=train_set.index)
        overall_predict = pd.Series(np.maximum(self.A_results.predict(1, end=stop_po), 0),
                                    index=d1.index)
        if plot_result == True:
            f1 = plt.figure(figsize=(16, 8))
            plt.grid()
            plt.xlabel("Date")
            plt.ylabel("Order Quantity")
            plt.title("aBOS-1000-D600")
            plt.plot(total, marker='^', color='k')
            plt.plot(overall_predict, marker='o', color='b')
            plt.xticks(rotation=90)
            plt.show()
        return overall_predict
    def ARIMA_predict(self,step):
        #data should be a Series
        new_result = self.A_mod.fit()
        temp_res = new_result.predict(step[0], step[1])
        return pd.Series(temp_res, index=data.index)
    def generater(self,path):
        import shutil
        shutil.copy(r'\res\TrainingSetModeSheet.xlsx',path)
        shutil.copy(r'\res\Prediction.xlsx',path)
        self.root_path = path




class Optimizer():
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set
        self.cut_p = len(train_set)+1
        self.end_p = self.cut_p+len(test_set)-1

    def grid_search(self, vrange=[0, 3]):
        res_list = []
        min_v = 1000000
        best_pra = [0, 0, 0]
        v_list = []
        for p in trange(vrange[0], vrange[1] + 1):
            for q in range(vrange[0], vrange[1] + 1):
                for d in range(vrange[0], vrange[1] + 1):
                    try:
                        temp_v = self.model(pra=(p, q, d))
                    except:
                        temp_v = 10000
                    v_list.append(temp_v)


                    if temp_v < min_v:
                        min_v = temp_v
                        best_pra = [p, q, d]
        print(f'the best pra is {best_pra}')
        return best_pra

    def model(self, pra=(1, 1, 1)):
        mod = sm.tsa.statespace.SARIMAX(self.train_set.values, trend='n', order=pra, seasonal_order=(1, 1, 1, 12))
        results = mod.fit()
        predict = results.predict(start=self.cut_p, end=self.end_p, dynamic=True)
        MSE = np.sqrt((predict - self.test_set.iloc[0]) ** 2)
        return MSE.mean()
if __name__=="__main__":
    data = pd.read_csv(r'C:\Users\SXF-Admin\Desktop\分析报告\res\example2.csv',encoding='ANSI')
    model1 = Opop(data)
    model1.model()
    model1.predict(data, plot_result=True)



