# deriva de autoforegpu.py
# paso de variable a una población, se usa la paralelización de numpy
import numpy as np
import time
np.__config__.show()


# Es una copia de autofore.py
# migración a gpu

# Es una copia de dinamic_prunning_in_forward_mode2

import random,math,time
import numpy as np
from autofore import ejemplo_red_neuronal_polinomios

outTime={}
inTime={}
lastTime={}
lastPrint=time.time()
def Dentro(name):
	start=time.time()
	if name in lastTime:
		if name not in outTime:
			outTime[name]=0
		outTime[name]+=start-lastTime[name]
	def fuera():
		global lastPrint
		end=time.time()		
		enlapsed=end-start
		if name not in inTime:
			inTime[name]=0
		inTime[name]+=enlapsed
		lastTime[name]=end
		if (end - lastPrint) > 1 and name in outTime:
			porcentaje=100*inTime[name]/(inTime[name]+outTime[name])
			print("Porcentaje de tiempo en",name,round(porcentaje,2),"%")
			lastPrint=end
	return fuera


class AutoFore:
	def __init__(self,gaf=None,pruning=0):
		self.variables=20 # en función de la memoria
		self.poblacion=128 # en función de la gpu
		self.gradientes=8 # número de variables, las menos significativas seran eliminadas

		self.nextVar=0 # Siguiente variable a usar
		self.nextPeso=0 # Siguiente peso a usar

		self.firma=np.zeros(self.variables) # firma para detectar fallo en el reuso de variables
		self.peso=np.zeros(self.variables,dtype=np.int8) # incica si la variable es un peso del sistema
		# -1 si no es, n si es el peso n, donde se almacena el gradiente
		self.peso2id=np.zeros(self.gradientes,dtype=np.int16) # indica el id de la variable que es el peso

		self.value=np.zeros((self.variables,self.poblacion),dtype=np.float32)
		self.delta=np.zeros((self.poblacion,self.gradientes),dtype=np.float32)
		self.g=np.zeros((self.variables,self.poblacion,self.gradientes),dtype=np.float32)
	
	def assign2(self,id_var,v2):	
		self.value[id_var] = v2
		self.g[id_var] = 0
		
	def applyDelta(self):
		idx,idy=cuda.grid(2)
		if idx>=self.value.shape[1] or idy>=self.g.shape[2]:
			return
		idaux=self.id[self.dest, idx, idy]
		if idaux==-1:
			return
		self.value[idaux,idx] -= self.delta[idaux,idx] * self.epsilon
		self.delta[idaux,idx] = 0

	def error2Delta(self,id):
		self.delta += self.g[id]


	def mul(self,dest,src1,src2):
		self.value[dest] = self.value[src1] * self.value[src2] #
		
		for idx in range(self.value.shape[1]):
			self.g[dest, idx] = self.g[src1, idx]*self.value[src2, idx]+self.g[src2, idx]*self.value[src1, idx] #

	def add(self,dest,src1,src2):
		self.value[dest] = self.value[src1] + self.value[src2]
		self.g[dest] = self.g[src1] + self.g[src2]

	def sub(self,dest,src1,src2):
		self.value[dest] = self.value[src1] - self.value[src2]	
		self.g[dest] = self.g[src1] - self.g[src2]

	def differentiable(self,id_var):
		self.g[id_var] = 0
		self.nextPeso+=1
		self.g[id_var, :, id_var] = 1
		self.delta[id_var, idx] = 0

	def assign(self):
		idx=cuda.grid(1)
		if idx>=self.value.shape[1]:
			return		
		self.value[self.id_var, idx] = self.v
		for i in range(self.g.shape[2]):
			self.g[self.id_var, idx, i] = 0
			self.id[self.id_var, idx,i] = -1


	def vector(self,x,y):
		if isinstance(x,Variable):
			xx=x
		else:
			xx=self.val(x)
		if isinstance(y,Variable):
			yy=y
		else:
			yy=self.val(y)
		return (xx,yy)

	def nominativeComparison(self,ref,epsilon):
		
		for n,p in enumerate(self.nominative):
			error=-p+ref.nominative[n].value
			if n==1:
				print("error",error.value)
			for key,grad in enumerate(p.forward):
				if grad!=0:
					self.id2var[key].delta+=error.value*grad*epsilon
				p.forward[key]=0
			p.value=ref.nominative[n].value
		self.applyDelta()

	def get(self,name):
		return self.params[name]

	def derivable(self):
		for k in self.keys:
			p=self.params[k]
			p.name=k
			p.derivable()

	def val(self,value):
		v=Variable(self)
		v.assign(value)
		return v
	
	def var(self):
		v=Variable(self)
		v.assign(0)
		return v
	
	def random(self,valueFrom,valueTo):
		v=Variable(self)
		dados=np.random.uniform(valueFrom,valueTo,self.poblacion).astype(np.float32)
		self.assign2(v.id2,dados)
		return v
	
	def param(self,valueFrom, valueTo):
		v=self.val(random.uniform(valueFrom,valueTo))
		v.valueFrom=valueFrom
		v.valueTo=valueTo
		return v.differentiable()
	
	def control(self,valueFrom, valueTo):
		v=self.val(random.uniform(valueFrom,valueTo))
		v.valueFrom=valueFrom
		v.valueTo=valueTo
		return v.nomination()

	def midVar(self):
		v=Variable(self)
		#v.forward=[0]*len(self.var2id)
		return v
	


# @cuda.jit
# def af_assing(matrix, row_idx, new_row, g, id):
# 	col = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
# 	if col < matrix.shape[1]:
# 		matrix[row_idx, col] = new_row[col]
# 		for i in range(g.shape[2]):
# 			g[row_idx, col, i] = 0
# 			id[row_idx, col] = -1



class Variable:
	def __init__(self, nn):
		self.nn=nn
		self.id2=nn.nextVar
		nn.nextVar+=1
		self.idPeso=-1
		if nn.nextVar==nn.variables:
			nn.nextVar=0
		while nn.peso[nn.nextVar]==1:
			nn.nextVar+=1
		# if nn.nextVar==nn.variables:
		# 	min=nn.referencia[0]
		# 	i=0
		# 	for j in range(1,nn.variables):
		# 		if nn.peso[j]==1:
		# 			continue
		# 		if nn.referencia[j]<min:
		# 			min=nn.referencia[j]
		# 			i=j
		# 	self.id2=i
		# else:
		# 	nn.nextVar+=1
		#nn.referencia[self.id2]=nn.operacion
		self.firma=np.random.randint(0, 2**16)
		nn.firma[self.id2]=self.firma

	def checkFirma(self):
		if self.firma!=self.nn.firma[self.id2]:
			raise Exception("Firma incorrecta")

	def _printGrad(self):
		nn=self.nn
		for id in nn.id[self.id2,0]:
			if id==-1:
				break
			print(id,end=" ")
		print()
	
	def value(self,id):
		return self.nn.value[self.id2,id]
	
	def error2Delta(self):
		self.nn.error2Delta(self.id2)

	def applyDelta(self,epsilon):
		#self.nn.applyDelta(self.id2,epsilon)
		for peso,id in enumerate(self.nn.peso2id):
			self.nn.value[id]+=self.nn.delta[:,peso]*epsilon
		self.nn.delta=0
	
	def minId(self):
		#fuera=Dentro("minId")
		i=0
		min=self.nn.value[self.id2,0]
		for j in range(1,self.nn.poblacion):
			if self.nn.value[self.id2,j]<min:
				min=self.nn.value[self.id2,j]
				i=j
		#fuera()
		return i
	
	def pruning(self):
		if self.nn.pruning==0:
			return 
		topDelta=[0]*self.nn.pruning
		for delta in self.forward:
			adelta=abs(delta)
			for m,td in enumerate(topDelta):
				if td<adelta:
					aux=topDelta[m]
					topDelta[m]=adelta
					adelta=aux
		for i,delta in enumerate(self.forward):
			adelta=abs(delta)
			if adelta<topDelta[-1]:
				self.forward[i]=0

	def clone(self):
		v=Variable(self.nn)
		v.value=self.value
		v.forward=list(self.forward)
		return v

	def assign(self,v):
		self.nn.value[self.id2]=v
		self.nn.g[self.id2]=0

	def set(self,v):
		if self.valueFrom<=v.value and v.value<=self.valueTo:
			self.value=v.value
			self.forward=v.forward
		return self

	def nomination(self):
		if self.nn.gaf.population[0]!=self.nn:
			self.value=self.nn.gaf.population[0].nominative[len(self.nn.nominative)].value
		self.nn.nominative.append(self)
		return self

	def get(self,v,pob):
		return self.nn.g[self.id2,pob,v.idPeso]

	def differentiable(self):
		if self.nn.nextPeso==self.nn.gradientes:
			raise Exception("No hay más gradientes")
		self.idPeso=self.nn.nextPeso
		self.nn.nextPeso+=1
		#self.nn.differentiable(self.id2,self.idPeso)
		self.nn.peso[self.id2]=1
		# self.id=len(self.nn.var2id)
		# self.nn.var2id[self]=self.id
		# self.nn.id2var[self.id]=self
		# self.forward=[0]*len(self.nn.var2id)
		# self.forward[self.id]=1
		# self.delta=0
		self.nn.g[self.id2]=0
		self.nn.g[self.id2,:,self.idPeso]=1
		return self
		
	def __add__(self, other):
		self.checkFirma
		v=self.nn.midVar()
		if not isinstance(other, Variable):
			aux=self.nn.midVar()
			aux.assign(other)
			other=aux
		else:
			other.checkFirma()

		# self.nn.referencia[self.id2]=self.nn.operacion
		# self.nn.referencia[other.id2]=self.nn.operacion
		# self.nn.operacion+=1
		
		# else:
		# 	v.value=self.value+other.value
		#fuera=Dentro("add")
		self.nn.add(v.id2,self.id2,other.id2)
		#fuera()

		# for child in (self, other):
		# 	if isinstance(child, Variable):
		# 		for name,value in enumerate(child.forward):
		# 			v.forward[name]+=value
		return v

	def __radd__(self, other):
		return self.__add__(other)

	def __mul__(self, other):
		self.checkFirma()
		v=self.nn.midVar()
		if not isinstance(other, Variable):
			aux=self.nn.midVar()
			aux.assign(other)
			other=aux
		else:
			other.checkFirma()

		# self.nn.referencia[self.id2]=self.nn.operacion
		# self.nn.referencia[other.id2]=self.nn.operacion
		# self.nn.operacion+=1

		#fuera=Dentro("mul")
		self.nn.mul(v.id2,self.id2,other.id2)
		#fuera()
		return v
	
	def __pow__(self, exponent):
		# Crear una nueva variable para el resultado
		v = self.nn.midVar()
		
		# Calcular el valor de la potencia
		v.value = self.value ** exponent
		
		# Calcular la pasada forward para los gradientes
		for name, value in enumerate(self.forward):
			v.forward[name] = exponent * (self.value ** (exponent - 1)) * value
		
		return v

	def __neg__(self):
		v=self.nn.midVar()
		v.value=-self.value
		child=self
		for name,value in enumerate(child.forward):
			link=-1
			v.forward[name]+=link*value 
		return v

	def sin(self):
		v=self.nn.midVar()
		v.value=math.sin(self.value)
		child=self
		for name,value in enumerate(child.forward):
			link=math.cos(child.value)
			v.forward[name]+=link*value 
		return v

	def cos(self):
		v=self.nn.midVar()
		v.value=math.cos(self.value)
		child=self
		for name,value in enumerate(child.forward):
			link=-math.sin(child.value)
			v.forward[name]+=link*value 
		return v

	def sigmoid(self):
		v=self.nn.midVar()
		v.value=1 / (1 + math.exp(-self.value))
		for name,value in enumerate(self.forward):
			
			link=(4 * math.cosh(v.value / 2)**2)
			v.forward[name]+=value / link
		return v

	def __sub__(self, other):
		self.checkFirma()
		v=self.nn.midVar()
		if not isinstance(other, Variable):
			aux=self.nn.midVar()
			aux.assign(other)
			other=aux
		else:
			other.checkFirma()

		# self.nn.referencia[self.id2]=self.nn.operacion
		# self.nn.referencia[other.id2]=self.nn.operacion
		# self.nn.operacion+=1
				
		self.nn.sub(v.id2,self.id2,other.id2)
		return v

	def __truediv__(self, other):
		v=self.nn.midVar()
		if isinstance(other, Variable):
			v.value=self.value/other.value
		else:
			v.value=self.value/other
		for i,child in enumerate((self, other)):
			if isinstance(child, Variable):
				for name,value in enumerate(child.forward):
					if i==0:
						if isinstance(other, Variable):
							link=1/other.value
						else:
							link=1/other
					else:
						link=-v.value/(other.value**2)
					v.forward[name]+=link*value 
		return v


class GeneticAutoFore:
	def __init__(self,populationSize):
		self.population=[AutoFore(gaf=self) for i in range(populationSize)]

class Contable:
	def __init__(self):
		self.g=ejemplo_red_neuronal_polinomios()
		self.A=next(self.g)	
		self.B=next(self.g)

	def check(self,v1):
		if isinstance(v1,Variable):
			v1=v1.value(0)
		v2=next(self.g)
		if hasattr(v2,"value"):
			v2=v2.value
		epsilon=0.01
		if abs(v1-v2)>epsilon:
			print(v1,v2)
			# throw exception
			raise Exception("No cuadra la contabilidad")

def ejemplo_red_neuronal_polinomios2():

	nn=AutoFore()

	# SISTEMA DE ECUACIONES y dimensiones

	# A   * B   = C
	# z*x * x*y = z*y
	x=2
	y=4
	z=100
	
	def f0(*args):
		return sum(args)
	def f1(a,b):
		return 1*a+2*b
	def f3(a,b):
		return 10*a+2*b
	def f4(a,b):
		return 2*a+5*b
	
	fs=[f0,f1,f3,f4]
	assert len(fs)==y
	
	contable=Contable()
	A=contable.A
	#A=[[random.random() for j in range(x)] for i in range(z)]

	Ct=[[fs[yy](*A[zz]) for zz in range(z)] for yy in range(y)]

	C=[[Ct[j][i] for j in range(y)] for i in range(z)]

	B=[[nn.val(contable.B[i][j].value).differentiable() for j in range(y)] for i in range(x)]

	#B=[[nn.random(0,1).differentiable() for j in range(y)] for i in range(x)]
	
	totalPendientes=y
	completado=[-1]*y

	# A   * B   = C
	# z*x * x*y = z*y
	ronda=0
	while True:
		ronda+=1
		print("ronda",ronda)
		for yy in range(y):
			if completado[yy]>-1:
				continue
			for b1 in B:
				b1[yy].delta=0
			errorTotal=nn.val(0)

			for zz in range(z):
				c=C[zz][yy]
				a=A[zz]
				cp=0
				
				for xx in range(x):
					contable.check(B[xx][yy])
					contable.check(a[xx])
					#B[xx][yy]._printGrad()
					aux=B[xx][yy]*a[xx]
					#aux._printGrad()
					cp+=aux
					#cp._printGrad()
					contable.check(cp)
					contable.check(cp.get(B[xx][yy],0))	
				#print("c",c.value)
				error=cp-c
				#error._printGrad()
				error2=error*error
				#error2._printGrad()
				errorTotal+=error2


				error2.error2Delta()

				# nn.dr.to_host("delta")
				for b1 in B:
					b=b1[yy]
					contable.check(nn.delta[b.id2][0])
					#b.delta+=error2.get(b)



			
			epsilon=0.01
			errorTotal.applyDelta(epsilon)
			# for b1 in B:
			# 	b=b1[yy]
			# 	b.value-=b.delta*epsilon

			# 	print(b.value,end=" ")
			# print()

			#print("errorTotal",errorTotal)	

			idmin=errorTotal.minId()

			print("errorTotal",errorTotal.value(idmin))

			if errorTotal.value(idmin)<0.0001:
				completado[yy]=idmin
				totalPendientes-=1
				break
		if totalPendientes==0:
			# print B transpuesta
			for yy in range(y):
				for xx in range(x):
					print(round(B[xx][yy].value(completado[yy]),1),end=" ")
				print()
			break
		
def exampleNumpy():
	poblacion=1

	for i in range(0,1000):
		x=np.random.randint(0,10,poblacion)
		y=np.random.randint(0,10,poblacion)

		start=time.time()
		z=x*y
		print(poblacion,"Tiempo de ejecución: ",time.time()-start)
		print("Tiempo unitario: ",(time.time()-start)/poblacion)
		poblacion*=10
	#z=x*y
	#z=np.cos(x)

	print(x)
	print(y)
	print(z)


def ejemplo_simple(nn):
	# Basado en el blog de colah
	# https://colah.github.io/posts/2015-08-Backprop/
	

	a=nn.val(2)
	b=nn.val(1)
	
	b.differentiable()

	#print("db/db",b.get(b,0))

	c=a+b
	print("dc/db",c.get(b,0))

	d=b+1
	e=c*d

	#print("e value",e.value(0))
	#print("de/db",e.get(b,0))


if __name__ == '__main__':
	start=time.time()
	nn=AutoFore()
	for i in range(1):
		#ejemplo_simple(nn)
		ejemplo_red_neuronal_polinomios2()
		#exampleNumpy()
	print("Tiempo de ejecución: ",time.time()-start)
