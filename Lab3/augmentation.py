import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def XYaugmentGenerator(gen1, gen2, X1, y, seed, batch_size):
    genX1 = gen1.flow(X1, y, batch_size=batch_size, seed=seed)
    genX2 = gen2.flow(y, X1, batch_size=batch_size, seed=seed)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        
        yield X1i[0], X2i[0]
