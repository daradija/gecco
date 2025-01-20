# 12/01/2025

## Diferencia de las versiones de gpu:

Los gradientes ocupan posiciones fijas. Tienen su propio espacio. Por lo tanto los gradientes solicitados no pueden superar AutoFore.gradientes. 

AutoFore peso2id contiene la traducción de la posición a la variable que le corresponde.

Peso y Delta tiene una primera dimensión diferente, pasa de población a gradientes:

self.delta=np.zeros((self.poblacion,self.gradientes),dtype=np.float32)


Hay un instrumento complejo, const que ha tenido que ser creado para el reciclaje de variables. Parece una constante, pero no es, una constante es algo que no cambia de valor, aquí una constante es algo que tiene una posición fija en memoria, es constante su puntero. Esta filosofía permite la reutilización de variables, dando una gestión muy eficaz de la memoria. A cambio de que el programador identifique que elementos se reutilizan o memorizan en ciertas estructuras.

Assign usa el concepto de préstamo de variable cuando se usa entre variables diferenciadas. De esta forma se permite la inicialización random.

# 19/01/2025

La versión actual provoca poca diversidad.

He intentado provocar diversidad la tasa de aprendizaje (learning rate) con AD.
Pero el resultado ha sido un empeoramiento del sistema, debido a que baja, provocando una lenta convergencia.

La convergencia es rápida.
Ha desaparecido la tasa de aprendizaje, o epsilon (ep):

```python
	def applyDelta(self,epsilon,id):
		ep=self.value[id]/len(self.peso2id)
		for peso,id in enumerate(self.peso2id):
			cte=self.sign(self.delta[:,peso])*np.abs(self.value[id])
			self.value[id]-=ep*cte
```

Pero contiene un componente anti-estandar.

```python
			cte=self.delta[:,peso]
```
La teoría de convergencia de las redes neuronales lo que nos indica es que de debiera añadir el gradiente (sumado en delta).

Hay una mejora que se puede aplicar que es hacer desaparecer delta. En concreto se calcula un error global y este contiene la suma de gradientes.

Se dejan las trazas y se deja la versión con convergencia rápida y como puede observarse hay poca diversidad.

Se pasa a una nueva versión tnato de roboticArm6 como de autoforenumpy y en ella se expone la nueva idea.

Nótese: De que el ajuste por intervalos (mínimo máximo del random) no ha ido muy bien y que el ángulo del brazo robótico se mueva mas o menos rápido para ciertos valores altos es algo que no me gusta, aunqeu soluciona el problema.