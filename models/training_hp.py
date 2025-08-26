import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt 
import json
import os
import keras_tuner as kt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5100)]
        )
    except RuntimeError as e:
        print(e)

arg = argparse.ArgumentParser(description="Finetuning DNN")
arg.add_argument("--xdata")
arg.add_argument("--ydata")
#arg.add_argument("--batch_size",type=int)
arg.add_argument("--epochs",type=int)

args = arg.parse_args()

xdata = np.load(args.xdata)
ydata = np.load(args.ydata)
epochs = args.epochs
#batch_size = args.batch_size
print(f"Shape of x is :{xdata.shape}, Shape of y is:{ydata.shape}")

#X_train_full, X_test, y_train_full, y_test = train_test_split(xdata[92:8464-92,:], del_F_reshaped[92:8464-92,:],random_state=40)
X_train_full, X_test, y_train_full, y_test = train_test_split(xdata[92:,:], ydata[92:,:],random_state=40)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full,random_state=30)


import keras_tuner as kt
weight_init = keras.initializers.GlorotNormal()
bias_init = keras.initializers.Zeros()
weight_reg = keras.regularizers.L2(0.0001)
bias_reg = keras.regularizers.L1()
###Building keras Functional API model###
def build_model(hp):
    
    input_ = keras.layers.Input(shape=(440,))
    hidden1 = keras.layers.Dense(hp.Int('unit_1', min_value=100,max_value=300,step=50),activation=keras.layers.LeakyReLU(),
            kernel_regularizer=keras.regularizers.L2(hp.Float('l2_reg', 1e-5, 1e-2, sampling='log')))(input_)
#dropout1 = keras.layers.Dropout(0.2)(hidden1)
    hidden2 = keras.layers.Dense(hp.Int('unit_2', min_value=75,max_value=200,step=50),activation=keras.layers.LeakyReLU(),
            kernel_regularizer=keras.regularizers.L2(hp.Float('l2_reg', 1e-5, 1e-2, sampling='log')))(hidden1)#(dropout1)
#dropout2 = keras.layers.Dropout(0.2)(hidden2)
    hidden3 = keras.layers.Dense(hp.Int('unit_3', min_value=50,max_value=100,step=25),activation=keras.layers.LeakyReLU(),
            kernel_regularizer=keras.regularizers.L2(hp.Float('l2_reg', 1e-5, 1e-2, sampling='log')))(hidden2)#(dropout2)
    dropout = keras.layers.Dropout(hp.Float('dropout_rate', 0.1, 0.5, step=0.1))(hidden3)
    output = keras.layers.Dense(100)(dropout)#(dropout4)
    model = keras.Model(inputs=[input_],outputs=[output])
    model.summary()
    opt = keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5]))
    model.compile(loss='mse',optimizer=opt,metrics=['mse','mae'])
    return model
from kerastuner.tuners import BayesianOptimization

tuner = BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=10,
    directory='DNN_tunning',
    project_name='bayesian_tuning'
)

tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

best_params = {
    "units_1": best_hps.get('unit_1'),
    "units_2": best_hps.get('unit_2'),
    "units_3": best_hps.get('unit_3'),
    "dropout_rate": best_hps.get('dropout_rate'),
    "learning_rate": best_hps.get('learning_rate'),
    "l2_reg": best_hps.get('l2_reg'),
}

with open("best_hyperparameters.json", "w") as f:
    json.dump(best_params, f, indent=4)

print("Best Hyperparameters Saved to best_hyperparameters.json")

