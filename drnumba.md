# ¿Cómo usar el Doctor Numba?

DrNumba facilita la programación en gpu usando numba.

¿Por qué incluir drnumba y no usar numba simplemente?

1. La programación con numba cuando se complica el kernel tiende a muchos parámetros, los cuales tienen muchas dimensiones, lo que conduce a un código ininteligible.
2. Numba no está pensado para depurar en gpu. 

La versión 3 es una orientación a objetos.
No se consigue una experiencia parecida a un array de objetos pero al menos el contendor y gestor de memoria/copia si.

Numba trabaja con estructuras de tipo numpy, que son tablas de tipos uniformes, de varias dimensiones (shape).

A pesar de ello, autoforegpu.py es un ejemplo claro de el azucar sintáctica que proporciona la autodiferenciación y la potencia de cuda.

Se incorpora la posibilidad de ejecutar una parte en gpu y cuando falle, ejecutar lo que falla en cpu.
Ya que la diferencia es abismal, entre cpu-gpu.

Doctor Numba da una experiencia ágil en la depuración de kernels. Permite el desarrollo de kernels como si de prototipos se tratase. Y evita parones de días de trabajo.

Está pensado para que toda la ejecución sea ejecutada en GPU.

Se encarga de los paso de parámetros.

Recuerda que:
1. dentro de los kernels no se puede pedir memoria (arrays), esta debe ser pasada como parámetro. Si se pueden crear registros, 
2. de hecho lo que es lento es escribir en arrays, no en registros.

# Manual de usuario.

```python
# incluye todo ya que tiene funciones de hack
from drnumba import * 

# Recuerda inclur numpy
import numpy as np
# Recuerda incluir este import en el 'kernel.py' o el que elijas si usas alguna función de np dentro de un kernel. 

# instancia una variable a nivel global
drnumba=DrNumba("kernel.py") 
# indica como parámetro en que fichero quieres que se guarden los kernels de cuda/numba y de CPU

# Ejemplo de clase
class Anillo:
	def __init__(self,x,y, dtype):
		# presenta la clase al doctor numba
		self.dr=drnumba.dr(self)

		# declara las variables
		# x,y pueden ser dimensiones
		self.x=x
		self.y=y
		self.data=np.array((y,x),dtype=np.int16)
		# x e y no son datos, son las dimensiones
		self.dr.data("y","x","data")

		# en tantas líneas como te sea cómodo
		self.pos=0
		self.dr.data("pos")

		# declara una variable insert, de tamaño param y se usará para una función llamada add
		# Si no existe y se añada param es un parámetro de la función add
		# nota: se pueden especificar varias funciones
		self.dr.data("x","insert",param=["add"])
		

		def post():
			self.pos+=1
			if self.pos>=self.y:
				self.pos=0

		# declara la función, indica que se ejecutará en paralelo x veces, e indica una función que ha de llamarse al final (post) (también existe pre). post es una función de cpu.
		dr.function("add","x",post=post)

	# la función add, al menos las primeras lineas la creará el dr al ser declarada, también declarará funciones en kernel.py, requiere varias llamadas
	def add(self):
		idx=cuda.grid(1)
		if idx>=self.data.shape[1]:
			return
		# fíjese que toda la manipulación se realiza como si de miembros se tratase
		self.data[self.pos][idx]=self.insert[idx]
```

# Uso

Tan sencillo como crear un objeto y llamar a las funciones.

El kernel reempaza las funciones por llamadas a GPU.

# Extensiones

Hay dos tipos de extansiones, modulares y de tipos.

Las modulares son como de herencia, si bien no se usa la herencia si consigue el ejecto de tener un objeto y en tiempo de ejecución darle nuevas capacidades, declarar nuevas funciones y datos, el código queda separado modularmente, sin interferencia entre los distintos módulos.

Una vez creado un objeto de tipo doctor puede ser usado como parámetros. La migración no es completa, se ha ido desarrollando el core a medida que se requería. 

# Depuración avanzada.

Basta con incluir un parámetro llamado cpu=True en la llamada al kernel.
Puedes depurar poniendo puntos de ruptura en el fichero de kernels declarado.