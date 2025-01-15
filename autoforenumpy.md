# 12/09/2025

## Diferencia de las versiones de gpu:

Los gradientes ocupan posiciones fijas. Tienen su propio espacio. Por lo tanto los gradientes solicitados no pueden superar AutoFore.gradientes. 

AutoFore peso2id contiene la traducción de la posición a la variable que le corresponde.

Peso y Delta tiene una primera dimensión diferente, pasa de población a gradientes:

self.delta=np.zeros((self.poblacion,self.gradientes),dtype=np.float32)


Hay un instrumento complejo, const que ha tenido que ser creado para el reciclaje de variables.
