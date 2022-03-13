# -*- coding: utf-8 -*-
"""
Created on Fri Aug 05 16:04:30 2021

@author: utku
"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt




def PSNRmetric(x_val, decoded,):
    psnr = tf.image.psnr(x_val, decoded, max_val=1.0)
    return psnr


def psnrs(x_in, x_out):
    if type(x_in) is list:
        img_in = x_in[0]
    else:
        img_in = x_in
    return tf.image.psnr(img_in, x_out, max_val=1.0)





def awgn(x, std):

    awgn = tf.random.normal(tf.shape(x), 0, std, dtype=tf.float32)
    r = x + awgn

    return r



def autoencoder_eval(k,c_snr,test_snr):
    std2 = np.sqrt(10 ** (-test_snr / 10))
    x_train_noisy = awgn(x_train,std2)
    x_test_noisy = awgn(x_test,std2)
    batch_size = 128
    c = int(k/32)


    input_img = Input(shape=(32, 32, 3))  

    #Encoder
    x = Conv2D(filters=16,
               kernel_size=(5,5),
               strides=2,
               padding='same')(input_img)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
     
    x = Conv2D(filters=32,
               kernel_size=(5,5),
               strides=2,
               padding='same')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
     
    x = Conv2D(filters=32,
               kernel_size=(5,5),
               strides=1,
               padding='same')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    
    x = Conv2D(filters=32,
               kernel_size=(5,5),
               strides=1,
               padding='same')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    
    x = Conv2D(filters= c,
               kernel_size=(5,5),
               strides=1,
               padding='same')(x)
    encoded_img = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    
    #Channel
    std = np.sqrt(10 ** (-c_snr / 10))
    shape = tf.shape(encoded_img)
    
    z = Flatten()(encoded_img)
    dim_z = tf.shape(z)[1]
    z_in = tf.sqrt(tf.cast(dim_z, dtype=tf.float32)) * tf.nn.l2_normalize(
                z, axis=1
            )
    

    z_out = awgn(z_in,std)

    z_out = tf.reshape(z_out, shape)

    
    #Decoder
    x = Conv2DTranspose(filters=32,
                        kernel_size=(5,5),
                        strides=1,
                        padding='same')(z_out)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    
    x = Conv2DTranspose(filters=32,
                        kernel_size=(5,5),
                        strides=1,
                        padding='same')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    
    x = Conv2DTranspose(filters= 32,
                        kernel_size=(5,5),
                        strides=1,
                        padding='same')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    
    x = Conv2DTranspose(filters=16,
                        kernel_size=(5,5),
                        strides=2,
                        padding='same')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    
    decoded_img = Conv2DTranspose(filters=3,
                              kernel_size= (5,5),
                              strides=2,
                              activation='sigmoid',
                              padding='same',
                              name='decoder_output')(x)
    

    autoencoder = Model(input_img,decoded_img)
    autoencoder.summary()
    
    
    
    
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss="mse",
                metrics = [PSNRmetric]
                )
    
    autoencoder.fit(x_train_noisy,
                    x_train,
                    epochs=30,
                    verbose = 1,
                    batch_size=batch_size,
                    validation_steps = 2000
                    )

    x_decoded = autoencoder.predict(x_test_noisy)

    avg_psnr = np.mean(psnrs(x_test, x_decoded))

    return avg_psnr

# load the CIFAR10 data
(x_train, _), (x_test, _) = cifar10.load_data()



# normalization
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train = x_train.reshape(x_train.shape[0], 32,32,3)
x_test = x_test.reshape(x_test.shape[0], 32,32,3)
#3
test_snr = [1,4,7,10,13,16,18,22,25]

psnr1 = []
psnr2 = []
psnr3 = []
psnr4 = []
psnr5 = []
for i in test_snr:
    psnr1.append(autoencoder_eval(512,1,i))
    psnr2.append(autoencoder_eval(512,4,i))
    psnr3.append(autoencoder_eval(512,7,i))
    psnr4.append(autoencoder_eval(512,13,i))
    psnr5.append(autoencoder_eval(512,19,i))


plt.plot(test_snr,psnr1,label = "SNR = 1dB") 
plt.plot(test_snr,psnr2,label = "SNR = 4dB")
plt.plot(test_snr,psnr3,label = "SNR = 7dB")
plt.plot(test_snr,psnr4,label = "SNR = 13dB")
plt.plot(test_snr,psnr5,label = "SNR = 19dB")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xlabel("SNRtest")

plt.ylabel("PSNR")
plt.title("AWGN Channel")
plt.show()
