#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python framework for developing neural network emulators of 
RRTMGP gas optics scheme

This program takes existing input-output data generated with RRTMGP and
user-specified hyperparameters such as the number of neurons, 
scales the data if requested, and trains a neural network. 

Alternatively, an automatic tuning method can be used for
finding a good set of hyperparameters (expensive).

Right now just a placeholder, pasted some of the code I used in my paper

Contributions welcome!

@author: Peter Ukkonen
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import losses, optimizers
import tensorflow as tf
import tensorflow.keras.backend as K
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten,Input
import numpy as np
import h5py
# import optuna

from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops, control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.training.optimizer import Optimizer

import shlex
from subprocess import Popen, PIPE

# err_metrics_rrtmgp_lw = np.array([0.0792,   0.0630,   0.0499,  -0.1624,  -0.3475,  -0.4379,   0.0025])
# Heating rate (all exps), Heating rate (present), SFC forcing (pre-industrial to present), 
# SFC forcing (present to future), TOA forcing (present to future), 
# TOA forcing CO2 (pre-industrial to 8x), SFC forcing N2O (pre-industrial to present) 



def hybrid_loss_wrapper(alpha):
    def loss_expdiff(y_true, y_pred):

        err_tot =  K.mean(K.square(y_pred - y_true))

        err_diff =  expdiff(y_true, y_pred)

        err = (alpha) * err_diff + (1 - alpha)*err_tot
            
        return err
    return loss_expdiff

def expdiff(y_true, y_pred):

    diff_pred = y_pred[1::2,:] - y_pred[0::2,:] 
    diff_true = y_true[1::2,:] - y_true[0::2,:] 
    
    # err_diff =  K.mean(K.square(diff_pred - diff_true))
    err_diff =  K.mean(K.abs(diff_pred - diff_true))

    return err_diff

def get_stdout(cmd):
    """
    Execute the external command and get its exitcode, stdout and stderr.
    """
    args = shlex.split(cmd)

    proc = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    # exitcode = proc.returncode
    #
    out = out.decode("utf-8")

    return out, err

from tensorflow.keras.callbacks import Callback

class RunRadiationScheme(Callback):
    
    def __init__(self, cmd, modelpath, modelsaver, patience=5, interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.cmd = cmd
        self.modelpath = modelpath
        self.modelsaver = modelsaver
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.err_metrics_rrtmgp = None
        
    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf
        
        self.best_epoch = 0
        
        print("Using RunRadiationScheme earlystopper, fluxes are validated " \
        "against Line-By-Line benchmark (RFMIP),\nand training stopped when a "\
        "weighted mean of the metrics printed by the radiation program have\n"\
        "not improved for {} epochs".format(self.patience ))
                
        print("The temporary model is saved to {}".format(self.modelpath))
        
        # First run the RRTMGP code without NNs to get the reference errors
        cmd_ref = self.cmd[0:75]
        out,err = get_stdout(cmd_ref)
        outstr = out.split('--------')
        err_metrics_str = outstr[2].strip('\n')
        err_metrics_str = err_metrics_str.split(',')
        self.err_metrics_rrtmgp = np.float32(err_metrics_str)
        # print("Reference errors were: {}".format(err_metrics_str))
        # err_metrics_norm = err_metrics / err_metrics_rrtmgp_lw

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            # Shortwave or Longwave?
            # Weight heating rates more for SW
            # if 'sw' in self.cmd:
            #     sw = True
            #     weight_hr = 0.75
            # else:
            #     sw = False
            #     weight_hr = 0.5
            # y_pred = self.model.predict_proba(self.X_val, verbose=0)
            # score = roc_auc_score(self.y_val, y_pred)
            
            # SAVE MODEL
            # print("saving to {}".format(self.modelpath))
            self.modelsaver(self.modelpath, self.model)
            
            # RUN RADIATION CODE WITH MODEL
            # print("running: {}".format(self.cmd))
            # cmd = './rrtmgp_lw_eval_nn_rfmip 8 ../../rrtmgp/data/rrtmgp-data-lw-g128-210809.nc 1 1 ' + modelinput
            out,err = get_stdout(self.cmd)
            outstr = out.split('--------')
            metric_names = outstr[1].strip('\n')
            metric_names = metric_names.split(',')
            for i in range(len(metric_names)): metric_names[i] = metric_names[i].lstrip().rstrip()
            err_metrics_str = outstr[2].strip('\n')
            err_metrics_str = err_metrics_str.split(',')
            err_metrics = np.float32(err_metrics_str)
            # err_metrics_norm = err_metrics / err_metrics_rrtmgp_lw
            err_metrics = err_metrics / self.err_metrics_rrtmgp
            # find position where forcing errors start
            indices = [i for i, elem in enumerate(metric_names) if 'HR' in elem]
            ind_forc = indices[-1] + 1
            
            # score = err_metrics.mean()
            logs["mean_relative_heating_rate_error"] = err_metrics[0]
            # Construct "overall" accuracy score for radiation
            # forcing_err = np.abs(err_metrics[2:]).mean()
            forcing_err = np.sqrt(np.mean(np.square(err_metrics[ind_forc:])))
            
            logs["mean_relative_forcing_error"] = forcing_err
            # hr_err = np.abs(err_metrics[0:ind_forc]).mean()
            hr_err = np.sqrt(np.mean(np.square(err_metrics[0:ind_forc])))

            # weight_forcing = 1 - weight_hr
            # score = weight_hr * hr_err + weight_forcing * forcing_err
            score = np.sqrt(np.mean(np.square(err_metrics)))
            logs["radiation_score"] = score
            
            # print("SCORE  {:9} {:9} {:16} {:20} {:20} {:20} {:20}". format(*metric_names))
            # print("{:.2f}   {:.2f}      {:.2f}     {:.2f}             {:.2f}        "\
            #       "         {:.2f}                 {:.2f}        "\
            #       "         {:.2f}". format(score,*err_metrics))
            print("The RFMIP accuracy relative to RRTGMP was:   {:.2f}   (HR {:.2f}, FLUXES/FORCINGS {:.2f})".format(score, hr_err, forcing_err))
            for i in range(len(err_metrics)):
                if (i==len(err_metrics)-1):
                    print("{}: {:.2f} \n".format(metric_names[i],err_metrics[i]), end =" ")
                else:
                    print("{}: {:.2f}, ".format(metric_names[i],err_metrics[i]), end =" ")
            # hr_ref = 0.0711
            # forcing_ref = 0.2
            # print("LBL errors - heating rate {:.3f} (RRTMGP {:.3f}), "\
            #       "TOA/sfc forcings {:.2f} ({:.2f}): weighted metric: {:.6f}".format(hr_err, hr_ref, forcing_err, forcing_ref, score))

            current = logs.get("radiation_score")
            # if epoch >  30: 
            # A local/temporary minimum can be found quickly, don't want to 
            # get stuck there: only start considering early stopping after a while
            if np.less(current, self.best):
                self.best = current
                self.wait = 0
                # Record the best weights if current results is better (less).
                self.best_weights = self.model.get_weights()
                self.best_epoch = epoch

            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print("Early stopping, the best radiation score (comprised of LBL heating rate"\
                          " and forcing errors normalized by RRTGMP values) was {:.2f}".format(self.best))
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    print("Restoring model weights from the end of the best epoch ({})".format(self.best_epoch+1))
                    self.model.set_weights(self.best_weights)
                        
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
     
        

# 1. Define an objective function to be maximized.
def create_model_hyperopt(trial, nx, ny):
    model = Sequential()
    
    # We define our MLP.
    # number of hidden layers
    n_layers = trial.suggest_int("n_layers", 1, 3)
    model = Sequential()
    # Input layer
    activ0 = trial.suggest_categorical('activation', ['relu', 'softsign'])
    num_hidden0 = trial.suggest_int("n_neurons_l0_l", 64, 256)
    model.add(Dense(num_hidden0, input_dim=nx, activation=activ0))
     
    for i in range(1, n_layers):
         num_hidden = trial.suggest_int("n_neurons_l{}".format(i), 64, 256)
         activ =trial.suggest_categorical('activation', ['relu', 'softsign']),
         model.add(Dense(num_hidden, activation=activ))
         
    # output layer
    model.add(Dense(ny, activation='linear'))
    
    # We compile our model with a sampled learning rate.
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    lossfunc    = losses.mean_squared_error
    model.compile(
        loss=lossfunc, 
        optimizer=optimizers.Adam(learning_rate=lr),
        metrics   = ['mean_absolute_error'],
        )
    return model



def create_model_mlp(nx,ny,neurons=[40,40], activ=['softsign','softsign','linear'],
                 kernel_init='he_uniform'):
    model = Sequential()
    # input layer (first hidden layer)
    model.add(Dense(neurons[0], input_dim=nx, kernel_initializer=kernel_init, activation=activ[0]))
    # further hidden layers
    for i in range(1,np.size(neurons)):
      model.add(Dense(neurons[i], activation=activ[i],kernel_initializer=kernel_init))
    # output layer
    model.add(Dense(ny, activation=activ[-1],kernel_initializer=kernel_init))
    
    return model



def savemodel(kerasfile, model):
   model.summary()
   newfile = kerasfile[:-3]+".txt"
   # model.save(kerasfile)
   try:
    model.save(kerasfile)
   except Exception:
        pass
   print("saving to {}".format(newfile))
   h5_to_txt(kerasfile,newfile)
   
   
def get_available_layers(model_layers, available_model_layers=[b"dense"]):
    parsed_model_layers = []
    for l in model_layers:
        for g in available_model_layers:
            if g in l:
                parsed_model_layers.append(l)
    return parsed_model_layers

# # KERAS HDF5 NEURAL NETWORK MODEL FILE TO NEURAL-FORTRAN ASCII MODEL FILE
def h5_to_txt(weights_file_name, output_file_name=''):

    #check and open file
    with h5py.File(weights_file_name,'r') as weights_file:

        weights_group_key=list(weights_file.keys())[0]

        # activation function information in model_config
        model_config = weights_file.attrs['model_config']#.decode('utf-8') # Decode using the utf-8 encoding
        model_config = model_config.replace('true','True')
        model_config = model_config.replace('false','False')

        model_config = model_config.replace('null','None')
        model_config = eval(model_config)

        model_layers = list(weights_file['model_weights'].attrs['layer_names'])
        # model_layers = get_available_layers(model_layers)
        print("names of layers in h5 file: %s \n" % model_layers)

        # attributes needed for .txt file
        # number of model_layers + 1(Fortran includes input layer),
        #   dimensions, biases, weights, and activations
        num_model_layers = len(model_layers)+1

        dimensions = []
        bias = {}
        weights = {}
        activations = []

        print('Processing the following {} layers: \n{}\n'.format(len(model_layers),model_layers))
        if 'Input' in model_config['config']['layers'][0]['class_name']:
            model_config = model_config['config']['layers'][1:]
        else:
            model_config = model_config['config']['layers']

        for num,l in enumerate(model_layers):
            layer_info_keys=list(weights_file[weights_group_key][l][l].keys())

            #layer_info_keys should have 'bias:0' and 'kernel:0'
            for key in layer_info_keys:
                if "bias" in key:
                    bias.update({num:np.array(weights_file[weights_group_key][l][l][key])})

                elif "kernel" in key:
                    weights.update({num:np.array(weights_file[weights_group_key][l][l][key])})
                    if num == 0:
                        dimensions.append(str(np.array(weights_file[weights_group_key][l][l][key]).shape[0]))
                        dimensions.append(str(np.array(weights_file[weights_group_key][l][l][key]).shape[1]))
                    else:
                        dimensions.append(str(np.array(weights_file[weights_group_key][l][l][key]).shape[1]))

            if 'Dense' in model_config[num]['class_name']:
                activations.append(model_config[num]['config']['activation'])
            else:
                print('Skipping bad layer: \'{}\'\n'.format(model_config[num]['class_name']))

    if not output_file_name:
        # if not specified will use path of weights_file with txt extension
        output_file_name = weights_file_name.replace('.h5', '.txt')

    with open(output_file_name,"w") as output_file:
        output_file.write(str(num_model_layers) + '\n')

        output_file.write("\t".join(dimensions) + '\n')
        if bias:
            for x in range(len(model_layers)):
                bias_str="\t".join(list(map(str,bias[x].tolist())))
                output_file.write(bias_str + '\n')
        if weights:
            for x in range(len(model_layers)):
                weights_str="\t".join(list(map(str,weights[x].T.flatten())))
                output_file.write(weights_str + '\n')
        if activations:
            for a in activations:
                if a == 'softmax':
                    print('WARNING: Softmax activation not allowed... Replacing with Linear activation')
                    a = 'linear'
                output_file.write(a + "\n")
