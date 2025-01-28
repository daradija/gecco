# deriva de autoforegpu.py
# paso de variable a uoa población, se usa la paralelización de numpy
import numpy as np
import time
#np.__config__.show()


# Es una copia de autofore.py
# migración a gpu

# Es una copia de dinamic_prunning_in_forward_mode2

import math,time
import numpy as np
from autofore import ejemplo_red_neuronal_polinomios



class AutoFore:
	def __init__(self,gaf=None,pruning=0,variables=20,gradientes=8,poblacion=128*10,seed=0):
		self.variables=variables # en función de la memoria
		self.poblacion=poblacion # en función de la gpu
		self.gradientes=gradientes # número de variables, las menos significativas seran eliminadas

		self.nextVar=0 # Siguiente variable a usar
		self.nextPeso=0 # Siguiente peso a usar

		self.firma=np.zeros(self.variables) # firma para detectar fallo en el reuso de variables
		self.peso=np.zeros(self.variables,dtype=np.int8) # incica si la variable es un peso del sistema
		# -1 si no es, n si es el peso n, donde se almacena el gradiente
		self.peso2id=np.zeros(self.gradientes,dtype=np.int16) # indica el id de la variable que es el peso

		self.value=np.zeros((self.variables,self.poblacion),dtype=np.float32)
		self.valueFrom=np.zeros(self.variables,dtype=np.float32)
		self.valueTo=np.zeros(self.variables,dtype=np.float32)
		self.g=np.zeros((self.variables,self.poblacion,self.gradientes),dtype=np.float32)
		self.prohibitedConst=False


		self.seed=seed
		# if seed!=0:
		self.seeder()
		#random.seed(self.seed.randint())
		self.learning_rate = (np.random.random((self.gradientes,self.poblacion)) * (0.01 - 0.00001) + 0.00001).astype(np.float32)

	def seeder(self):
		np.random.seed(self.seed.randint(0,2**32-1))

	def assign2(self,id_var,v2):	
		self.value[id_var] = v2
		self.g[id_var] = 0
		
	def sign(self,x):
		return np.where(x > 0, 1, np.where(x < 0, -1, 0))#+x
		#return np.tanh(x)
	
	

	def mul(self,dest,src1,src2):
		self.value[dest] = self.value[src1] * self.value[src2] #
		
		# for idx in range(self.value.shape[1]):
		# 	self.g[dest, idx] = self.g[src1, idx]*self.value[src2, idx]+self.g[src2, idx]*self.value[src1, idx] #

		self.g[dest, :,:] = (
				self.g[src1, :,:] * self.value[src2, :,np.newaxis] +
				self.g[src2, :,:] * self.value[src1, :,np.newaxis]
		)

	def div(self, dest, src1, src2):
		# División elemento a elemento en self.value (2D)
		self.value[dest] = self.value[src1] / self.value[src2]

		# for idx in range(self.value.shape[1]):
		# 	self.g[dest, idx] = self.g[src1, idx]/self.value[src2, idx]-self.value[src1, idx]*self.g[src2, idx]/(self.value[src2, idx]**2)

		# Actualización de gradientes en self.g (3D)
		self.g[dest, :, :] = (
			self.g[src1, :, :] / self.value[src2, :, np.newaxis] -
			(self.value[src1, :, np.newaxis] * self.g[src2, :, :]) /
			(self.value[src2, :, np.newaxis] ** 2)
		)


	def add(self,dest,src1,src2):
		self.value[dest] = self.value[src1] + self.value[src2]
		self.g[dest] = self.g[src1] + self.g[src2]

	def sub(self,dest,src1,src2):
		self.value[dest] = self.value[src1] - self.value[src2]	
		self.g[dest] = self.g[src1] - self.g[src2]

	# def cos(self,dest,src):
	# 	self.value[dest] = np.cos(self.value[src])
	# 	for idx in range(self.value.shape[1]):
	# 		self.g[dest, idx] = -np.sin(self.value[src, idx])*self.g[src, idx]

	def cos(self, dest, src):
		# Calcular el coseno para value (2D)
		self.value[dest] = np.cos(self.value[src])

		# Actualizar el gradiente (3D)
		self.g[dest, :, :] = -np.sin(self.value[src, :, np.newaxis]) * self.g[src, :, :]
			

	# def atan(self,dest,src):
	# 	self.value[dest] = np.arctan(self.value[src])
	# 	for idx in range(self.value.shape[1]):
	# 		self.g[dest, idx] = 1/(1+self.value[src, idx]**2)*self.g[src, idx]

	def atan(self, dest, src):
		# Calcular la arctangente para value (2D)
		self.value[dest] = np.arctan(self.value[src])

		# Actualizar el gradiente (3D)
		self.g[dest, :, :] = (
			1 / (1 + self.value[src, :, np.newaxis] ** 2) * self.g[src, :, :]
		)


	# def tanh(self,dest,src):
	# 	self.value[dest] = np.tanh(self.value[src])
	# 	for idx in range(self.value.shape[1]):
	# 		self.g[dest, idx] = (1-self.value[dest, idx]**2)*self.g[src, idx]

	def tanh(self, dest, src):
		# Calcular la tangente hiperbólica para value (2D)
		self.value[dest] = np.tanh(self.value[src])

		# Actualizar el gradiente (3D)
		self.g[dest, :, :] = (
			(1 - self.value[dest, :, np.newaxis] ** 2) * self.g[src, :, :]
		)

	# def sin(self,dest,src):
	# 	self.value[dest] = np.sin(self.value[src])
	# 	for idx in range(self.value.shape[1]):
	# 		self.g[dest, idx] = np.cos(self.value[src, idx])*self.g[src, idx]

	def sin(self, dest, src):
		# Calcular el seno para value (2D)
		self.value[dest] = np.sin(self.value[src])

		# Actualizar el gradiente (3D)
		self.g[dest, :, :] = np.cos(self.value[src, :, np.newaxis]) * self.g[src, :, :]



	def neg(self,dest,src):
		self.value[dest] = -self.value[src]
		self.g[dest] = -self.g[src]

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

	def val(self,value):
		v=Variable(self)
		v.assign(value)
		return v
	
	def const(self,value):
		if self.prohibitedConst:
 			raise Exception("No se pueden crear más constantes")
		v=Variable(self)
		v.assign(value)
		self.peso[v.id2]=1
		return v
	
	def noMoreConst(self):
		self.prohibitedConst=True
	
	def var(self):
		v=Variable(self)
		v.assign(0)
		return v
	
	def random(self,valueFrom,valueTo):
		v=Variable(self)
		self.seeder()
		dados=np.random.uniform(valueFrom,valueTo,self.poblacion).astype(np.float32)
		self.assign2(v.id2,dados)
		self.valueTo[v.id2]=valueTo
		self.valueFrom[v.id2]=valueFrom
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
		self.g[v.id2]=0
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
			#print("Reciclando variables")
			nn.nextVar=0
		while nn.peso[self.id2]==1:
			self.id2=nn.nextVar
			nn.nextVar+=1
			if nn.nextVar==nn.variables:
				nn.nextVar=0
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
		# if self.id2==22:
		# 	print("id",self.id2)
		self.nn.seeder()
		self.firma=np.random.randint(0, 2**16)
		nn.firma[self.id2]=self.firma

	def learn(self):
		#ep=self.nn.value[self.id2]/len(self.nn.peso2id)
		for peso,id in enumerate(self.nn.peso2id):
			#cte=self.nn.sign(self.nn.g[self.id2,:,peso])*np.abs(self.nn.value[id]) 
			cte=self.nn.g[self.id2,:,peso]*np.abs(self.nn.value[id]) 
			# _var pob _gra * _var pob		
			if cte[0]!=0:
				print(self.nn.g[self.id2,0,peso])
				print(self.nn.value[id,0])
				print("cte",cte)
			epsilon=self.nn.learning_rate[peso]
			self.nn.value[id]-=epsilon*cte

			self.nn.value[id] = np.where(
				self.nn.valueFrom[id] != self.nn.valueTo[id],
				np.clip(self.nn.value[id], self.nn.valueFrom[id], self.nn.valueTo[id]),
				self.nn.value[id]
			)



	def geneticAlgorithm(self,killdown=1,doit=False,mutate=0.2):
		self.id2
		self.nn.value
		# clone variable
		clon=self.nn.value[self.id2].copy()
		# remove 0 index
		clon=clon[1:]
		# create sort index
		index=np.argsort(clon)+1

		parents=index[:int(len(index)-killdown)]
		children=index[int(len(index)-killdown):]

		if len(parents)==0 or len(children)==0:
			return []

		if doit:
			for id in children:
				father=parents[0]
				self.nn.seeder()
				mother=np.random.choice(parents)
				#mother=father
				for peso,id2 in enumerate(self.nn.peso2id):
					self.nn.seeder()
					# if np.random.random()<mutate and self.nn.valueFrom[id2]!=self.nn.valueTo[id2]:
					# 	self.nn.value[id2,id]=np.random.uniform(self.nn.valueFrom[id2],self.nn.valueTo[id2])
					# else:
					if np.random.random()<0.5:
						self.nn.value[id2,id]=self.nn.value[id2,father]
						# _gra pob 
					else:
						self.nn.value[id2,id]=self.nn.value[id2,mother]
					self.nn.seeder()
					if np.random.random()<mutate:
						self.nn.learning_rate[peso,id]=np.random.uniform(0.00001,0.01)
					else:
						if np.random.random()<0.5:
							self.nn.learning_rate[peso,id]=self.nn.learning_rate[peso,father]#*np.random.uniform(0.9,1.1)
						else:
							self.nn.learning_rate[peso,id]=self.nn.learning_rate[peso,mother]#*np.random.uniform(0.9,1.1)
				
		return children


	def checkFirma(self):
		if self.firma!=self.nn.firma[self.id2]:
			raise Exception("Firma incorrecta en la variable",self.id2)

	def _printGrad(self):
		nn=self.nn
		for id in nn.id[self.id2,0]:
			if id==-1:
				break
			print(id,end=" ")
		print()
	
	def value(self,id):
		return self.nn.value[self.id2,id]
	
	def applyDelta(self,epsilon):
		#self.nn.applyDelta(self.id2,epsilon)
		for peso,id in enumerate(self.nn.peso2id):
			self.nn.value[id]-=self.nn.delta[:,peso]*epsilon
		self.nn.delta=0
	
	def minId(self):
		#fuera=Dentro("minId")
		i=1
		min=self.nn.value[self.id2,0]
		for j in range(2,self.nn.poblacion):
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
		if isinstance(v,Variable):
			self.nn.value[self.id2]=self.nn.value[v.id2]
			self.nn.g[self.id2]=self.nn.g[v.id2]
			# si tiene peso se presta
			if v.idPeso!=-1:
				self.idPeso=v.idPeso
				self.nn.peso2id[self.idPeso]=self.id2
				self.nn.peso[self.id2]=1
				
				# Free the other variable
				v.idPeso=-1
		else:
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
		self.nn.peso2id[self.idPeso]=self.id2
		self.nn.peso[self.id2]=1
		self.nn.g[self.id2]=0
		self.nn.g[self.id2,:,self.idPeso]=1
		return self
		
	def __add__(self, other):
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
		self.checkFirma()
		# Crear una nueva variable para el resultado
		v = self.nn.midVar()
		
		# Calcular el valor de la potencia
		v.value = self.value ** exponent
		
		# Calcular la pasada forward para los gradientes
		for name, value in enumerate(self.forward):
			v.forward[name] = exponent * (self.value ** (exponent - 1)) * value
		
		return v

	def __neg__(self):
		self.checkFirma()
		v=self.nn.midVar()
		self.nn.neg(v.id2,self.id2)
		return v

	def sin(self):
		self.checkFirma()
		v=self.nn.midVar()
		self.nn.sin(v.id2,self.id2)
		return v

	def cos(self):
		self.checkFirma()
		v=self.nn.midVar()
		self.nn.cos(v.id2,self.id2) 
		return v
	
	def atan(self):
		self.checkFirma()
		v=self.nn.midVar()
		self.nn.atan(v.id2,self.id2)
		return v

	def tanh(self):
		self.checkFirma()
		v=self.nn.midVar()
		self.nn.tanh(v.id2,self.id2)
		return v

	def sigmoid(self):
		self.checkFirma()
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
		self.checkFirma()
		v=self.nn.midVar()
		self.nn.div(v.id2,self.id2,other.id2)
		return v

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
	
	# contable=Contable()
	# A=contable.A
	A=[[random.random() for j in range(x)] for i in range(z)]

	Ct=[[fs[yy](*A[zz]) for zz in range(z)] for yy in range(y)]

	C=[[Ct[j][i] for j in range(y)] for i in range(z)]

	# B=[[nn.val(contable.B[i][j].value).differentiable() for j in range(y)] for i in range(x)]
	B=[[nn.random(0,1).differentiable() for j in range(y)] for i in range(x)]
	
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
					# contable.check(B[xx][yy])
					# contable.check(a[xx])
					#B[xx][yy]._printGrad()
					aux=B[xx][yy]*a[xx]
					#aux._printGrad()
					cp+=aux
					#cp._printGrad()
					# contable.check(cp)
					# contable.check(cp.get(B[xx][yy],0))	
				#print("c",c.value)
				error=cp-c
				#error._printGrad()
				error2=error*error
				#error2._printGrad()
				errorTotal+=error2


				error2.error2Delta()

				# nn.dr.to_host("delta")
				# for b1 in B:
				# 	b=b1[yy]
				# 	contable.check(nn.delta[0][b.idPeso])



			
			epsilon=0.01
			errorTotal.applyDelta(epsilon)
			# for b1 in B:
			# 	b=b1[yy]
			# 	contable.check(b.value(0))

			# 	print(b.value,end=" ")
			# print()

			#print("errorTotal",errorTotal)	

			idmin=errorTotal.minId()

			print("errorTotal",errorTotal.value(idmin))
			# contable.check(errorTotal.value(idmin))

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
