# 12/09/2025

## Diferencia de las versiones de gpu:

Los gradientes ocupan posiciones fijas. Tienen su propio espacio. Por lo tanto los gradientes solicitados no pueden superar AutoFore.gradientes. 

AutoFore peso2id contiene la traducción de la posición a la variable que le corresponde.

Delta tiene una primera dimensión diferente:

self.delta=np.zeros((self.poblacion,self.gradientes),dtype=np.float32)

