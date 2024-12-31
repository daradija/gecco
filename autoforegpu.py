# Es una copia de autofore.py
# migración a gpu

# Es una copia de dinamic_prunning_in_forward_mode2

import random,math,time
from drnumba import*
import numpy as np

drnumba=DrNumba("kernelAutofore.py")
class AutoFore:
	def __init__(self,gaf=None,pruning=0):
		# self.var2id={}
		# self.id2var={}
		# self.nominative=[]
		# self.gaf=gaf
		# self.pruning=pruning



		self.dr=drnumba.dr(self)
		self.variables=1000 # en función de la memoria
		self.poblacion=4608 # en función de la gpu
		self.gradientes=32 # número de variables, las menos significativas seran eliminadas

		self.nextVar=0 # Siguiente variable a usar

		self.operacion=0 # Lleva la cuenta de las operaciones realizadas
		self.referencia=np.zeros(self.variables,dtype=np.int16) # marca los datos usados en cada operación, para su reuso
		self.peso=np.zeros(self.variables,dtype=np.int8) # incica si la variable es un peso del sistema
		
		# self.operador=-1 # Código donde se programa la operación a realizar
		# self.r1=-1
		# self.r2=-1
		# self.r3=-1

		self.value=np.zeros((self.variables,self.poblacion),dtype=np.float16)
		self.dr.data("variables","poblacion","value")
		self.delta=np.zeros((self.variables,self.poblacion),dtype=np.float16)
		self.dr.data("variables","poblacion","delta")
		self.g=np.zeros((self.variables,self.poblacion,self.gradientes),dtype=np.float16)
		self.dr.data("variables","poblacion","gradientes","g")
		self.id=np.zeros((self.variables,self.poblacion,self.gradientes),dtype=np.int16) # posición que ocupa la variable
		self.dr.data("variables","poblacion","gradientes","id")

		#self.id_var=np.int16(0)
		self.dr.data("id_var",param=["assign","differentiable"])
		#self.v=np.float16(0)
		self.dr.data("v",param=["assign"])
		self.dr.function("assign","poblacion")

		self.dr.function("differentiable","poblacion")

		self.dr.data("dest",param=["add","sub","mul","div","pow","sin","cos","sigmoid"])
		self.dr.data("src1",param=["add","sub","mul","div","pow","sin","cos","sigmoid"])
		self.dr.data("src2",param=["add","sub","mul","div","pow"])
		self.dr.function("add","poblacion")


	def add(self):
		idx=cuda.grid(1)
		if idx>=self.value.shape[1]:
			return
		self.value[self.dest, idx] = self.value[self.src1, idx] + self.value[self.src2, idx]
		for i in range(self.g.shape[2]):
			self.g[self.dest, idx, i] = self.g[self.src1, idx, i] 
			self.id[self.dest, idx, i] = self.id[self.src1, idx, i]
			if self.id[self.src2, idx, i]!=-1:
				break
		for i in range(self.g.shape[2]):
			i=-1
			min=0
			id1=self.id[self.src1, idx, i]
			for j in range(self.g.shape[2]):
				id2=self.id[self.src2, idx, j]
				if id2==-1:
					break
				if id2==id1:
					i=-1
					self.g[self.dest, idx, i] +=  self.g[self.src2, idx, i]
					break
				g2=np.abs(self.g[self.src2, idx, i])
				if min<g2:
					min=g2
					i=id2
			if i!=-1:
				self.g[self.dest, idx, i] = min
				self.id[self.dest, idx, i] = i

	def differentiable(self):
		idx=cuda.grid(1)
		if idx>=self.value.shape[1]:
			return
		self.g[self.id_var, idx, 0] = 1
		self.id[self.id_var, idx, 0] = self.id_var
		self.delta[self.id_var, idx] = 0

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

	def applyDelta(self):
		for p in self.id2var.values():
			p.value+=p.delta
			p.delta=0

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

		self.nn.assign(self.id2,v)

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

	def get(self,v):
		id=self.nn.var2id[v]
		return self.forward[id]

	def differentiable(self):
		self.nn.differentiable(self.id2)
		# self.id=len(self.nn.var2id)
		# self.nn.var2id[self]=self.id
		# self.nn.id2var[self.id]=self
		# self.forward=[0]*len(self.nn.var2id)
		# self.forward[self.id]=1
		# self.delta=0
		return self
		
	def __add__(self, other):
		v=self.nn.midVar()
		if not isinstance(other, Variable):
			aux=self.nn.midVar()
			aux.assign(other)
			other=aux
		# else:
		# 	v.value=self.value+other.value
		self.nn.add(v.id2,self.id2,other.id2,cpu=True)

		# for child in (self, other):
		# 	if isinstance(child, Variable):
		# 		for name,value in enumerate(child.forward):
		# 			v.forward[name]+=value
		return v

	def __radd__(self, other):
		return self.__add__(other)

	def __mul__(self, other):
		v=self.nn.midVar()
		if not isinstance(other, Variable):
			v.value=self.value*other
		else:
			v.value=self.value*other.value

		children=(self, other)
		for i,child in enumerate(children):
			if isinstance(child, Variable):
				for name,value in enumerate(child.forward):
					if isinstance(children[1-i], Variable):
						link=children[1-i].value
					else:
						link=children[1-i]
					v.forward[name]+=link*value 
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
		v=self.nn.midVar()
		if isinstance(other, Variable):
			v.value=self.value-other.value
		else:
			v.value=self.value-other
		for i,child in enumerate((self, other)):
			if isinstance(child, Variable):
				for name,value in enumerate(child.forward):
				
					link=1-2*i
					v.forward[name]+=link*value 
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

def ejemplo_red_neuronal_polinomios():

	nn=AutoFore()

	# SISTEMA DE ECUACIONES y dimensiones

	# A   * B   = C
	# z*x * x*y = z*y
	x=2
	y=4
	z=4
	
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
	
	A=[[random.random() for j in range(x)] for i in range(z)]

	Ct=[[fs[yy](*A[zz]) for zz in range(z)] for yy in range(y)]

	C=[[Ct[j][i] for j in range(y)] for i in range(z)]

	B=[[nn.val(random.random()).differentiable() for j in range(y)] for i in range(x)]
	
	totalPendientes=y
	completado=[False]*y

	# A   * B   = C
	# z*x * x*y = z*y
	while True:
		for yy in range(y):
			if completado[yy]:
				continue
			for b1 in B:
				b1[yy].delta=0
			errorTotal=0

			for zz in range(z):
				c=C[zz][yy]
				a=A[zz]
				cp=0
				for xx in range(x):
					cp+=B[xx][yy]*a[xx]

				#print("c",c.value)
				error=cp-c
				error2=error*error
				errorTotal+=error2.value

				for b1 in B:
					b=b1[yy]
					b.delta+=error2.get(b)

			
			epsilon=0.01
			for b1 in B:
				b=b1[yy]
				b.value-=b.delta*epsilon
			# 	print(b.value,end=" ")
			# print()

			#print("errorTotal",errorTotal)	

			if errorTotal<0.0001:
				completado[yy]=True
				totalPendientes-=1
				break
		if totalPendientes==0:
			# print B transpuesta
			for yy in range(y):
				for xx in range(x):
					print(round(B[xx][yy].value,1),end=" ")
				print()
			break
		

def ejemplo_simple():
	# Basado en el blog de colah
	# https://colah.github.io/posts/2015-08-Backprop/
	nn=AutoFore()

	a=nn.val(2)
	b=nn.val(1)
	
	b.differentiable()

	c=a+b
	d=b+1
	e=c*d

	print("e value",e.value)
	print("de/db",e.get(b))


if __name__ == '__main__':
	start=time.time()
	ejemplo_simple()
	#ejemplo_red_neuronal_polinomios()
	
	print("Tiempo de ejecución: ",time.time()-start)