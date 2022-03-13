# -*- coding: utf-8 -*-
"""
Created on Fri Aug 02 09:33:30 2021

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


def slow_rayleigh_awgn(x, stddev, h = None):

    if h is None:
        n1 = tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2), dtype=tf.float32)
        n2 = tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2), dtype=tf.float32)
        h = tf.sqrt(tf.square(n1) + tf.square(n2))

    awgn = tf.random.normal(tf.shape(x), 0, stddev / np.sqrt(2), dtype=tf.float32)

    return (h * x + awgn), h


def autoencoder_eval(k,c_snr,ch):

    prev_h = None
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
    
    if(ch == "awgn"):
        z_out = awgn(z_in,std)
    
    elif(ch == "slowfading"):
        z_out,h = slow_rayleigh_awgn(z_in, std, prev_h)
        prev_h = h
        
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
    
    autoencoder.fit(x_train,
                    x_train,
                    epochs=30,
                    verbose = 1,
                    validation_data= (x_test,x_test),
                    batch_size=batch_size,
                    validation_steps = 2000
                    )

    x_decoded = autoencoder.predict(x_test)

    avg_psnr = np.mean(psnrs(x_test, x_decoded))

    return avg_psnr

# load the CIFAR10 data
(x_train, _), (x_test, _) = cifar10.load_data()



# normalization
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train = x_train.reshape(x_train.shape[0], 32,32,3)
x_test = x_test.reshape(x_test.shape[0], 32,32,3)
autoencoder_eval(128,0,"slowfading")

#1
k = [128,256,512,768,1024,1280,1472]
psnr0 = []
for i in k:
    psnr0.append(autoencoder_eval(i,0,"awgn"))
    
psnr10 = []    
for j in k:
    psnr10.append(autoencoder_eval(j,10,"awgn"))

psnr20 = []    
for z in k:
    psnr20.append(autoencoder_eval(z,20,"awgn"))
      
   
kn = [0.042,0.083,0.167,0.25,0.33,0.417,0.48]
plt.plot(kn,psnr0,label = "SNR = 0dB") 
plt.plot(kn,psnr10,label = "SNR = 10dB")
plt.plot(kn,psnr20,label = "SNR = 20dB")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xlabel("k/n")

plt.ylabel("PSNR")
plt.title("AWGN Channel")
plt.show()  

#2
s = [128,256,512,768,1024,1280,1472]
psnrs0 = []
for i in s:
    psnrs0.append(autoencoder_eval(i,0,"slowfading"))
    
psnrs10 = []    
for j in s:
    psnrs10.append(autoencoder_eval(j,10,"slowfading"))

psnrs20 = []    
for z in s:
    psnrs20.append(autoencoder_eval(z,20,"slowfading"))


kn = [0.042,0.083,0.167,0.25,0.33,0.417,0.48]
plt.plot(kn,psnrs0,label = "SNR = 0dB") 
plt.plot(kn,psnrs10,label = "SNR = 10dB")
plt.plot(kn,psnrs20,label = "SNR = 20dB")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xlabel("k/n")

plt.ylabel("PSNR")
plt.title("AWGN+ Slow Rayleigh Fading Channel")
plt.show()


