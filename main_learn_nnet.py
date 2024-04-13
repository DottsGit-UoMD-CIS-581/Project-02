# -*- coding: utf-8 -*-
"""
CIS 581
Project 2

Library given by the University of Michigan - Dearborn
Author post "if __name__ == '__main__':": Nicholas Butzke
"""
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import warnings

from collections import defaultdict

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import ParameterGrid, KFold, train_test_split

from numpy.random import default_rng

rng = default_rng()

def random_uniform(n,m,R=[-1.0,1.0]):
    a, b = R[0], R[1]
    return (b - a) * rng.random((n,m)) + a

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def deriv_sigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def squared_error(y_true,y_pred):
    return 0.5 * (y_true - y_pred) ** 2

def deriv_squared_error(y_true,y_pred):
    return y_pred - y_true

def indicator(p):
    return p.astype(int)

def error_rate(y_true, y_pred):
    return 1.0 - np.mean(indicator(y_true == y_pred))

def identity(x):
    return x

def deriv_identity(x):
    return np.ones(x.shape)

def make_nunits(n,K,L,N):
    nunits = [n]
    for l in range(L):
        nunits.append(N)
    nunits.append(K)
    return nunits

def time_nnet(nunits):
    t = 0
    for l in range(len(nunits)-1):
        t += (nunits[l] + 1) * nunits[l+1]
    return t

MAX_ITERS = 50
MAX_NHIDU = 2**9
MAX_NHIDL = 2**2
MAX_M = 2000

n= 1024
K = 10

MAX_NUNITS = make_nunits(n,K,MAX_NHIDL,MAX_NHIDU)
MAX_NNET_TIME = time_nnet(MAX_NUNITS)

MAX_TIME = MAX_M * MAX_NNET_TIME * MAX_ITERS

class NNetBaseFunction:
        def __init__(self, f=None,df=None):
            self.f = f
            self.df = df

        def deepcopy(self):
            return NNetBaseFunction(f=self.f, df=self.df)

class NNetActivation(NNetBaseFunction):
        def __init__(self, f=sigmoid,df=deriv_sigmoid):
            super().__init__(f=f,df=df)

        def deepcopy(self):
            return NNetActivation(f=self.f, df=self.df)

class NNetLoss(NNetBaseFunction):
    def __init__(self, f=squared_error,df=deriv_squared_error):
        super().__init__(f=f,df=df)

    def deepcopy(self):
        return NNetLoss(f=self.f, df=self.df)

class NNetMetric(NNetBaseFunction):
    def __init__(self, f=error_rate):
        super().__init__(f=f,df=None)

    def deepcopy(self):
        return NNetMetric(f=self.f)

class NNetLayer:
    def __init__(self,n_in=1,n_out=1,W=None,
                             unit=NNetActivation(), initializer=random_uniform):
        self.n_in = n_in
        self.n_out = n_out
        if initializer is None:
            initializer =    lambda n, m : np.zeros((n,m))
        self.initializer = initializer
        if W is None:
            W = self.initializer(n_out,n_in+1)
        else:
            self.n_in, self.n_out = W.shape[1]-1, W.shape[0]
        self.W = W
        self.unit = unit

    def ds(self, x):
        return self.unit.df(x)

    def deepcopy(self):
        return NNetLayer(n_in=self.n_in,n_out=self.n_out,W=self.W.copy(),
                                         unit=self.unit)

    def copy_layer(self, layer):
        self.W[:] = layer.W[:]
        return self

    # assumes x[0,:] = +1
    def aggregation_with_dummy_input(self, x):
        return np.matmul(self.W,x)

    def aggregation(self, x):
        if x.shape[0] == self.W.shape[1]:
            x_tmp = x
        else:
            x_tmp = np.ones(self.W.shape[1],x.shape[1])
            x_tmp[1:,:] = x
        return self.aggregation_with_dummy_input(x_tmp)

    def activation(self, x):
        return self.unit.f(self.aggregation(x))

    def set_x(self, x):
        return x

    def set_y(self, y):
        return y

    def get_y(self):
        return None

class NNetIdentityLayer (NNetLayer):
    def __init__(self,n_in=1,n_out=1,W=None):
        super().__init__(n_in=n_in,n_out=n_out,W=W,
                                         unit=NNetActivation(identity,deriv_identity))

class NNetLayerProp(NNetLayer):
    def __init__(self,n_in=1,n_out=1,W=None,
                             unit=NNetActivation(sigmoid,deriv_sigmoid),m=1):
        super().__init__(n_in=n_in,n_out=n_out,W=W,unit=unit)
        # self.y = np.ones((n_out+1,m))
        # self.y[1:,:] = 0
        # self.delta = np.zeros((n_out+1,m))
        self.x = None
        self.y = None
        self.delta = None

    def deepcopy(self):
        copy = super().deepcopy()
        # Input is not "stored" by layer
        copy.x = self.x
        copy.y = None if self.y is None else self.y.copy()
        copy.delta = None if self.delta is None else self.delta.copy()
        return copy

    def set_x(self, x):
        self.x = x
        return x

    def set_y(self, y):
        self.y = y
        return y

    def set_delta(self, delta):
        self.delta = delta
        return delta

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_delta(self):
        return self.delta

    def dW(self):
        return np.matmul(self.delta,self.x.T)

class NNetInputLayerProp(NNetLayerProp):
    def __init__(self,n_in=1,n_out=1,W=None,m=1):
        super().__init__(n_in=n_in,n_out=n_out,W=W,unit=NNetActivation(identity,deriv_identity))
        self.y = None

    def deepcopy(self):
        obj = super().deepcopy()
        obj.y = None if self.y is None else self.y.deepcopy()
        return obj

class NNetOptimizer:
    def __init__(self,loss=NNetLoss(),metric=NNetMetric()):
        self.loss = loss
        self.metric = metric
        self.best_nnet = None
        self.last_nnet = None
        self.train_err = []
        self.test_err = []
        return self

    def deepcopy(self):
        opt = NNetOptimizer(loss=self.loss.deepcopy(),metric=self.metric.deepcopy())
        opt.best_nnet = None if self.best_nnet is None else self.best_nnet.deepcopy()
        opt.last_nnet = None if self.best_nnet is None else self.last_nnet.deepcopy()
        opt.train_err = self.train_err.deepcopy()
        opt.test_err = self.test_err.deepcopy()
        return opt

    def run(self,nnet,X,y):
        return self.best_nnet

class NNet:
    def __init__(self, nunits=[0,0], unit=NNetActivation(sigmoid,deriv_sigmoid),
                             output_unit=None, Layer=NNetLayerProp, InputLayer=NNetInputLayerProp):
        self.nunits = nunits
        self.unit = unit
        self.output_unit = unit if output_unit is None else output_unit
        self.nlayers = len(nunits)
        self.layer = []
        self.layer.append(InputLayer(n_in=1,n_out=nunits[0]))

        for l in range(1,self.nlayers-1):
            self.layer.append(Layer(n_in=nunits[l-1],n_out=nunits[l],unit=unit))

        self.layer.append(Layer(n_in=nunits[-2],n_out=nunits[-1],
                                                                unit=self.output_unit))

    def copy(self, nnet_copy=None, Layer=NNetLayerProp, InputLayer=NNetInputLayerProp):
        if nnet_copy is None:
            nnet_copy = NNet(nunits=self.nunits,unit=self.unit,output_unit=self.output_unit, Layer=Layer, InputLayer=InputLayer)
        nnet_copy.copy_layers(self)
        return nnet_copy

    def deepcopy(self, nnet_copy=None):
        nnet_copy = self.copy(nnet_copy=nnet_copy)

        nnet_copy.nunits = copy.deepcopy(self.nunits)
        nnet_copy.unit = self.unit.deepcopy()
        nnet_copy.output_unit = self.output_unit.deepcopy()

        for l in range(1,self.nlayers):
            nnet_copy.layer[l] = self.layer[l].deepcopy()

        return nnet_copy

    def copy_layers(self, nnet_copy_from):
        for l in range(self.nlayers):
            self.layer[l].copy_layer(nnet_copy_from.layer[l])
        return self

    def error(self, X, y, loss=squared_error, metric=None):
        output = self.forwardprop(X.T)
        err = np.mean(loss(y.T, output))
        err_rate = 1.0 if metric is None else metric(y.T,output)
        return err, err_rate

    def forwardprop(self,X):
        m = X.shape[1]
        out_vals = np.ones((X.shape[0]+1,m))
        out_vals[1:,:] = X
        self.layer[0].set_y(out_vals)

        for l in range(1,self.nlayers):
            self.layer[l].set_x(self.layer[l-1].get_y())
            del out_vals
            out_vals = np.ones((self.nunits[l]+1,m))
            out_vals[1:,:] = self.layer[l].activation(self.layer[l].get_x())
            self.layer[l].set_y(out_vals)

        return out_vals[1:,:]

    def backprop(self,X,y,dE='deriv_squared_error'):
        net_output = self.forwardprop(X)

        layer = self.layer[self.nlayers-1]
        layer.set_delta(layer.ds(net_output) * dE(y,net_output))

        for l in range(self.nlayers-1,1,-1):
            next_layer = self.layer[l]
            layer = self.layer[l-1]
            x = layer.get_y()[1:,:]
            d = next_layer.delta
            layer.set_delta(layer.ds(x) * np.matmul(next_layer.W[:,1:].T,d))

        dW = []
        for l in range(self.nlayers):
            dW.append(None)

        for l in range(self.nlayers-1,0,-1):
            dW[l] = self.layer[l].dW()

        return dW

    def fit(self, X, y, X_test=None, y_test=None, optimizer=None, verbose=0):
        if optimizer is None:
            optimizer = NNetGDOptimizer(loss=NNetLoss())

        best_nnet = optimizer.run(self,X,y,X_test,y_test,verbose)

        self.copy_layers(best_nnet)

        return self

class NNetGDOptimizer(NNetOptimizer):
    def __init__(self,loss=NNetLoss(),max_iters=100, learn_rate=1, reg_param=0,
                             change_thresh=1e-4, change_err_thresh=1e-6,metric=NNetMetric()):
        super().__init__(loss=loss,metric=metric)
        self.max_iters = max_iters
        self.learn_rate = learn_rate
        self.reg_param = reg_param
        self.change_thresh = change_thresh
        self.change_err_thresh = change_err_thresh

    def deepcopy(self):
        opt = super().deepcopy()
        return NNetGDOptimizer(loss=opt.loss, max_iters=self.max_iters, learn_rate=self.learn_rate, reg_param=self.reg_param,
                             change_thresh=self.change_thresh, change_err_thresh=self.change_err_thresh,metric=opt.metric)

    def run(self, nnet, X, y, X_test=None, y_test=None, verbose=0):
        m = X.shape[0]
        eval_test = X_test is not None and y_test is not None
        new_nnet = NNet(nunits=nnet.nunits,unit=nnet.unit,output_unit=nnet.output_unit,Layer=NNetLayerProp,InputLayer=NNetInputLayerProp)
        new_nnet.copy_layers(nnet)

        t = 0
        max_change = math.inf
        min_change_err = math.inf

        train_err = []
        test_err = []

        err, err_rate = new_nnet.error(X, y, loss=self.loss.f, metric=self.metric.f)
        if verbose > 0:
            print((err,err_rate))
        min_err, min_err_rate = err, err_rate

        if eval_test:
            cv_err,cv_err_rate = new_nnet.error(X_test, y_test, loss=self.loss.f, metric=self.metric.f)

        best_nnet = nnet.deepcopy()
        best_nnet.copy_layers(new_nnet)

        while min_change_err > self.change_err_thresh and max_change > self.change_thresh and t < self.max_iters:
            if verbose > 0:
                print(t)
                print("Backprop...")

            dW = new_nnet.backprop(X.T, y.T,dE=self.loss.df)

            if verbose > 0:
                print("done.")
                print("Update...")

            max_change = 0

            for l in range(new_nnet.nlayers-1,0,-1):
                delta_W = self.learn_rate * (dW[l] / m + self.reg_param * new_nnet.layer[l].W)
                new_nnet.layer[l].W[:] = new_nnet.layer[l].W[:] - delta_W[:]
                max_change = max(max_change, np.max(np.absolute(delta_W)))

            del dW[:]

            if verbose > 0:
                print("done.")

            last_err = err
            err,err_rate = new_nnet.error(X, y, loss=self.loss.f, metric=self.metric.f)
            if verbose > 0:
                print((err,err_rate))
            min_change_err = np.absolute(err-last_err)


            if verbose > 0:
                print("max_change")
                print(max_change)

            if eval_test:
                cv_err,cv_err_rate = new_nnet.error(X_test, y_test, loss=self.loss.f, metric=self.metric.f)

            if verbose > 0:
                if eval_test:
                    print("(test_err,test_err_rate)")
                    print((cv_err,cv_err_rate))

            if err < min_err:
                min_err = err
                min_err_rate = err_rate
                best_nnet.copy_layers(new_nnet)

            t += 1

            train_err.append([err, err_rate])
            if eval_test:
                test_err.append([cv_err, cv_err_rate])

        if verbose > 0:
            if eval_test:
                print("(best_train_err,best_train_err_rate)")
                print((min_err,min_err_rate))

        self.train_err = train_err
        self.test_err = test_err

        return best_nnet

if __name__ == '__main__':
    # model_select is for debugging purposes. -1: all, 0: perceptron, 1: DNN, 2: DNN2
    model_select = 2

    D = np.loadtxt("./optdigits_train.dat") 
    D_test = np.loadtxt("./optdigits_test.dat")
    D_trial = np.loadtxt("./optdigits_trial.dat")

    m, n = D.shape[0], D.shape[1]-1

    X = D[:,:-1].reshape(m,n)
    y = D[:,-1].reshape(m,1)
    out_enc = LabelBinarizer() 
    y_ohe = out_enc.fit_transform(y)
    K = y_ohe.shape[1]

    m_test = D_test.shape[0]
    X_test = D_test[:,:-1].reshape(m_test,n) 
    y_test = D_test[:,-1].reshape(m_test,1)
    y_test_ohe = out_enc.transform(y_test)

    m_trial = D_trial.shape[0]
    X_trial = D_trial[:,:-1].reshape(m_trial,n)
    y_trial = D_trial[:,-1].reshape(m_trial,1) 
    y_trial_ohe = out_enc.transform(y_trial)

    def nnet_error_rate(y_true, y_pred):
        y_pred_label = np.argmax(y_pred, axis=0).reshape(-1, 1)
        y_true_label = out_enc.inverse_transform(y_true.T).reshape(-1, 1)
        return error_rate(y_true_label, y_pred_label)

    nnet_metric = NNetMetric(f=nnet_error_rate)

    if model_select == -1 or model_select == 0:
        # Perceptron
        learn_rates = [4**i for i in range(5)]
        min_err = float('inf')
        perceptron_cv_errors = []
        for lr in learn_rates:
            perceptron_nnet = NNet(nunits=[n,K]) 
            kf = KFold(n_splits=3)
            cv_errs = []
            for train_idx, test_idx in kf.split(X):
                X_train, X_cv = X[train_idx], X[test_idx]
                y_train, y_cv = y_ohe[train_idx], y_ohe[test_idx]
                perceptron_opt = NNetGDOptimizer(learn_rate=lr, metric=nnet_metric)
                perceptron_nnet.fit(X_train, y_train, optimizer=perceptron_opt)
                _, cv_err = perceptron_nnet.error(X_cv, y_cv, metric=nnet_metric.f)
                cv_errs.append(cv_err)
            mean_cv_err = np.mean(cv_errs)
            perceptron_cv_errors.append(mean_cv_err)
            if mean_cv_err < min_err:
                min_err = mean_cv_err
                best_lr = lr
        
        # CV error curve
        print("Perceptron - Average CV Test Misclassification Errors:")
        print(perceptron_cv_errors)

        plt.figure()
        plt.plot(learn_rates, perceptron_cv_errors, marker='o')
        plt.xscale('log', base=4)
        plt.xlabel('Learning Rate')
        plt.ylabel('Average CV Test Misclassification Error')
        plt.title('Perceptron - CV Error Curve')
        plt.show()

        # Perceptron GradientDescent Fitting Curves
        perceptron_opt = NNetGDOptimizer(learn_rate=best_lr, metric=nnet_metric, max_iters=1000)
        perceptron_nnet = NNet(nunits=[n,K])
        perceptron_nnet.fit(X, y_ohe, X_test, y_test_ohe, optimizer=perceptron_opt, verbose=1)

        perceptron_train_err, perceptron_test_err = np.array(perceptron_opt.train_err), np.array(perceptron_opt.test_err)

        print("Perceptron Final Training Error (MSE):", perceptron_train_err[-1,0])
        print("Perceptron Final Training Error Rate:", perceptron_train_err[-1,1]) 
        print("Perceptron Final Test Error (MSE):", perceptron_test_err[-1,0])
        print("Perceptron Final Test Error Rate:", perceptron_test_err[-1,1])

        plt.figure()
        plt.plot(perceptron_train_err[:,0], label='Training')
        plt.plot(perceptron_test_err[:,0], label='Test')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error') 
        plt.legend()
        plt.title('Perceptron Training and Test MSE Curves')
        plt.show()

        plt.figure()
        plt.plot(perceptron_train_err[:,1], label='Training')
        plt.plot(perceptron_test_err[:,1], label='Test')  
        plt.xlabel('Epochs')
        plt.ylabel('Misclassification Error Rate')
        plt.legend()
        plt.title('Perceptron Training and Test Error Rate Curves') 
        plt.show()

        # Evaluate on trial set
        perceptron_trial_outputs = perceptron_nnet.forwardprop(X_trial.T)
        perceptron_trial_pred = np.argmax(perceptron_trial_outputs, axis=0) 
        print("Trial Set Predictions:", perceptron_trial_pred)

        # Visualize Perceptron weights  
        fig, axes = plt.subplots(2, 5, figsize=(12,6))
        axes = axes.ravel()
        for i in range(10):
            img = perceptron_nnet.layer[-1].W[i,1:].reshape(32,32)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Digit {i}")
            axes[i].axis('off')
        plt.suptitle("Perceptron Weights as Images")    
        plt.show()

        # Limit training to these numbers of initial samples
        ms = [10, 40, 100, 200, 400, 800, 1600]

        perceptron_train_errs = []
        perceptron_test_errs = []

        for m in ms:
            perceptron_nnet = NNet(nunits=[n,K])
            perceptron_opt = NNetGDOptimizer(learn_rate=best_lr, metric=nnet_metric)
            perceptron_nnet.fit(X[:m], y_ohe[:m], X_test, y_test_ohe, optimizer=perceptron_opt)
            perceptron_train_err, _ = perceptron_nnet.error(X[:m], y_ohe[:m], metric=nnet_metric.f)
            perceptron_test_err, _ = perceptron_nnet.error(X_test, y_test_ohe, metric=nnet_metric.f)
            perceptron_train_errs.append(perceptron_train_err)
            perceptron_test_errs.append(perceptron_test_err)
            
        plt.figure()
        plt.plot(ms, perceptron_train_errs, label='Training')
        plt.plot(ms, perceptron_test_errs, label='Test')  
        plt.xlabel('Number of Training Examples')
        plt.ylabel('Misclassification Error Rate')
        plt.legend()
        plt.title('Perceptron Learning Curves')
        plt.show()
        
    if model_select == -1 or model_select == 1:
        hidden_units = [4**i for i in range(2,5)]
        depths = [1,2,3,4]
        learn_rates = [4**i for i in range(-2,3)]
        min_err = float('inf')
        
        # Create separate lists to store CV errors for each hidden_unit and depth combination
        DNN_cv_errors = {(hu, d): [] for hu in hidden_units for d in depths}
        
        for hu in hidden_units:
            for d in depths:
                for lr in learn_rates:
                    nunits = [n] + [hu]*d + [K]
                    DNN_nnet = NNet(nunits=nunits)
                    
                    kf = KFold(n_splits=3)
                    cv_errs = []
                    for train_idx, test_idx in kf.split(X):
                        X_train, X_cv = X[train_idx], X[test_idx]
                        y_train, y_cv = y_ohe[train_idx], y_ohe[test_idx]
                        DNN_opt = NNetGDOptimizer(learn_rate=lr, metric=nnet_metric)
                        DNN_nnet.fit(X_train, y_train, optimizer=DNN_opt)
                        _, cv_err = DNN_nnet.error(X_cv, y_cv, metric=nnet_metric.f)
                        cv_errs.append(cv_err)
                        
                    mean_cv_err = np.mean(cv_errs)
                    DNN_cv_errors[(hu, d)].append(mean_cv_err)
                    
                    if mean_cv_err < min_err:
                        min_err = mean_cv_err
                        best_nunits = nunits
                        best_lr = lr
        
        # Grid plotting all of the cv plots of hidden unit and depth combinations
        fig, axes = plt.subplots(len(depths), len(hidden_units), figsize=(12, 8), sharex=True, sharey=True)
        
        for i, d in enumerate(depths):
            for j, hu in enumerate(hidden_units):
                ax = axes[i, j]
                ax.plot(learn_rates, DNN_cv_errors[(hu, d)], marker='o')
                ax.set_xscale('log', base=4)
                ax.set_xlabel('Learning Rate')
                ax.set_ylabel('Average CV Test Misclassification Error')
                ax.set_title(f'Hidden Units: {hu}, Depth: {d}')
        
        plt.tight_layout()
        plt.show()

        # DNN GradientDescent Fitting Curves
        DNN_opt = NNetGDOptimizer(learn_rate=best_lr, metric=nnet_metric, max_iters=1000)
        DNN_nnet = NNet(nunits=best_nunits)
        DNN_nnet.fit(X, y_ohe, X_test, y_test_ohe, optimizer=DNN_opt, verbose=1)

        DNN_train_err, DNN_test_err = np.array(DNN_opt.train_err), np.array(DNN_opt.test_err)

        print("DNN Final Training Error (MSE):", DNN_train_err[-1,0])
        print("DNN Final Training Error Rate:", DNN_train_err[-1,1]) 
        print("DNN Final Test Error (MSE):", DNN_test_err[-1,0])
        print("DNN Final Test Error Rate:", DNN_test_err[-1,1])

        plt.figure()
        plt.plot(DNN_train_err[:,0], label='Training')
        plt.plot(DNN_test_err[:,0], label='Test')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.title('DNN Training and Test MSE Curves')
        plt.show()

        plt.figure()
        plt.plot(DNN_train_err[:,1], label='Training') 
        plt.plot(DNN_test_err[:,1], label='Test')
        plt.xlabel('Epochs')
        plt.ylabel('Misclassification Error Rate')
        plt.legend()
        plt.title('DNN Training and Test Error Rate Curves')
        plt.show()

        # Evaluate on trial set 
        DNN_trial_outputs = DNN_nnet.forwardprop(X_trial.T)
        DNN_trial_pred = np.argmax(DNN_trial_outputs, axis=0)
        print("Trial Set Predictions:", DNN_trial_pred)

        # Visualize random weights from first hidden layer
        num_hidden_units = DNN_nnet.layer[1].W.shape[0]
        random_units = np.random.choice(range(num_hidden_units), size=10, replace=False)

        fig, axes = plt.subplots(2, 5, figsize=(12,6)) 
        axes = axes.ravel()
        for i, unit in enumerate(random_units):
            img = DNN_nnet.layer[1].W[unit,1:].reshape(32,32)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Hidden Unit {unit}")
            axes[i].axis('off')
        plt.suptitle("Random First Hidden Layer Weights as Images")
        plt.show()

        # Limit training to these numbers of initial samples
        ms = [10, 40, 100, 200, 400, 800, 1600]

        DNN_train_errs = []
        DNN_test_errs = []

        for m in ms:
            DNN_nnet = NNet(nunits=best_nunits)
            DNN_opt = NNetGDOptimizer(learn_rate=best_lr, metric=nnet_metric)
            DNN_nnet.fit(X[:m], y_ohe[:m], X_test, y_test_ohe, optimizer=DNN_opt)
            DNN_train_err, _ = DNN_nnet.error(X[:m], y_ohe[:m], metric=nnet_metric.f)
            DNN_test_err, _ = DNN_nnet.error(X_test, y_test_ohe, metric=nnet_metric.f)
            DNN_train_errs.append(DNN_train_err)
            DNN_test_errs.append(DNN_test_err)
            
        plt.figure()
        plt.plot(ms, DNN_train_errs, label='Training')
        plt.plot(ms, DNN_test_errs, label='Test')  
        plt.xlabel('Number of Training Examples')
        plt.ylabel('Misclassification Error Rate')
        plt.legend()
        plt.title('DNN Learning Curves')
        plt.show()

    if model_select == -1 or model_select == 2:
        h1_units = [4**3, 4**4]
        h2_units = [4**2, 4**3]
        learn_rates = [4**i for i in range(-3,2)]

        min_err = float('inf')
        
        # Create separate lists to store CV errors for each h1_unit and h2_unit combination
        DNN2_cv_errors = {(h1, h2): [] for h1 in h1_units for h2 in h2_units}
        
        for h1 in h1_units:
            for h2 in h2_units:
                for lr in learn_rates:
                    nunits = [n, h1, h2, K] 
                    DNN2_nnet = NNet(nunits=nunits)
                    
                    kf = KFold(n_splits=3)
                    cv_errs = []
                    for train_idx, test_idx in kf.split(X):
                        X_train, X_cv = X[train_idx], X[test_idx]  
                        y_train, y_cv = y_ohe[train_idx], y_ohe[test_idx]
                        DNN2_opt = NNetGDOptimizer(learn_rate=lr, metric=nnet_metric)
                        DNN2_nnet.fit(X_train, y_train, optimizer=DNN2_opt)
                        _, cv_err = DNN2_nnet.error(X_cv, y_cv, metric=nnet_metric.f) 
                        cv_errs.append(cv_err)
                    
                    mean_cv_err = np.mean(cv_errs)
                    DNN2_cv_errors[(h1, h2)].append(mean_cv_err)
                    
                    if mean_cv_err < min_err:  
                        min_err = mean_cv_err
                        best_nunits = nunits
                        best_lr = lr
        
        # Grid plotting all of the cv plots of hidden unit and depth combinations
        fig, axes = plt.subplots(len(h2_units), len(h1_units), figsize=(12, 8), sharex=True, sharey=True)
        
        for i, h2 in enumerate(h2_units):
            for j, h1 in enumerate(h1_units):
                ax = axes[i, j]
                ax.plot(learn_rates, DNN2_cv_errors[(h1, h2)], marker='o')
                ax.set_xscale('log', base=4)
                ax.set_xlabel('Learning Rate')
                ax.set_ylabel('Average CV Test Misclassification Error')
                ax.set_title(f'Hidden 1 Units: {h1}, Hidden 2 Units: {h2}')
        
        plt.tight_layout()
        plt.show()

        # DNN2 GradientDescent Fitting Curves
        DNN2_opt = NNetGDOptimizer(learn_rate=best_lr, metric=nnet_metric, max_iters=1000)
        DNN2_nnet = NNet(nunits=best_nunits)
        DNN2_nnet.fit(X, y_ohe, X_test, y_test_ohe, optimizer=DNN2_opt, verbose=1)

        DNN2_train_err, DNN2_test_err = np.array(DNN2_opt.train_err), np.array(DNN2_opt.test_err)

        print("DNN2 Final Training Error (MSE):", DNN2_train_err[-1,0])
        print("DNN2 Final Training Error Rate:", DNN2_train_err[-1,1])
        print("DNN2 Final Test Error (MSE):", DNN2_test_err[-1,0]) 
        print("DNN2 Final Test Error Rate:", DNN2_test_err[-1,1])

        plt.figure()
        plt.plot(DNN2_train_err[:,0], label='Training')
        plt.plot(DNN2_test_err[:,0], label='Test')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.title('DNN2 Training and Test MSE Curves')
        plt.show()

        plt.figure()
        plt.plot(DNN2_train_err[:,1], label='Training')
        plt.plot(DNN2_test_err[:,1], label='Test')
        plt.xlabel('Epochs') 
        plt.ylabel('Misclassification Error Rate')
        plt.legend()
        plt.title('DNN2 Training and Test Error Rate Curves')
        plt.show()

        # Evaluate on trial set
        DNN2_trial_outputs = DNN2_nnet.forwardprop(X_trial.T) 
        DNN_2trial_pred = np.argmax(DNN2_trial_outputs, axis=0)
        print("DNN2 Trial Set Predictions:", DNN_2trial_pred)

        # Visualize random weights from first hidden layer 
        num_hidden_units = DNN2_nnet.layer[1].W.shape[0]
        random_units = np.random.choice(range(num_hidden_units), size=10, replace=False)

        fig, axes = plt.subplots(2, 5, figsize=(12,6))
        axes = axes.ravel()
        for i, unit in enumerate(random_units):
            img = DNN2_nnet.layer[1].W[unit,1:].reshape(32,32) 
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Hidden Unit {unit}")
            axes[i].axis('off')
        plt.suptitle("Random First Hidden Layer Weights as Images") 
        plt.show()

        # Limit training to these numbers of initial samples
        ms = [10, 40, 100, 200, 400, 800, 1600]

        DNN2_train_errs = []
        DNN2_test_errs = []

        for m in ms:
            DNN2_nnet = NNet(nunits=best_nunits)
            DNN2_opt = NNetGDOptimizer(learn_rate=best_lr, metric=nnet_metric)
            DNN2_nnet.fit(X[:m], y_ohe[:m], X_test, y_test_ohe, optimizer=DNN2_opt)
            DNN2_train_err, _ = DNN2_nnet.error(X[:m], y_ohe[:m], metric=nnet_metric.f)
            DNN2_test_err, _ = DNN2_nnet.error(X_test, y_test_ohe, metric=nnet_metric.f) 
            DNN2_train_errs.append(DNN2_train_err)
            DNN2_test_errs.append(DNN2_test_err)
            
        plt.figure()
        plt.plot(ms, DNN2_train_errs, label='Training')
        plt.plot(ms, DNN2_test_errs, label='Test')
        plt.xlabel('Number of Training Examples')
        plt.ylabel('Misclassification Error Rate')
        plt.legend()  
        plt.title('DNN2 Learning Curves')
        plt.show()