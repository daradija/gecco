# Genética de población
* Identificación de mejores por error
* Cruce

# Mejoras

* Mejora del error estereoscópico.
En el caso de la visión estereocópica se decide que 

'''python
	def error(self,c):
		pygame.draw.line(self.screen,c.color,(c.x.value(0),c.y.value(0)),self.focus,1)
		m=(c.y-self.focus[1])/(c.x-self.focus[0])
		# pendiente a ángulo
		angle=m.atan()	
		#angle=math.atan2(c.y.value(0)-focus_cam[1],c.x.value(0)-focus_cam[0])

		#if c==a:
		error=angle-angle.value(0)
		return error

    ...
  
    error=None
    for c in arm:
        for eye in eyes:
            errorAux=eye.error(c)
            if error is None:
                error=errorAux
            else:
                error+=errorAux
    error2=error*error
    error2.error2Delta()
    nn.applyDelta(learning_rate)

'''
Devuelva el error y computar fuera la suma de los errores. Es mas correcto.

* tanh aplicado al computo del error. Pero no funciona.
```python
    #error=angle-angle.value(0)
    error=angle.tanh()-math.tanh(angle.value(0))
```
En el acta dejo anotada una linea de trabajo futura.

# Representación gráfica 
Se dibujan los brazos en distintas posiciones para poderlos ver.
Se utiliza el color rojo para identificar los peores candidatos, que morirán.
En verde los mejores y en azul la referencia.

# 
Actualización según el random.
cada individuo tiene una tasa de actualización diferente.
