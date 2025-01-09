# Deriva de dinamic_prunning_in_forward_mode2.py y fue incorporado el 29/12/2024
# Deriva del poyecto pybolsa y fue incorporado el 16/09/2024

# Deriva del poyecto pybolsa y fue incorporado el 16/09/2024

import numpy as np
import re
import ast
import os
import itertools
import inspect
import importlib
import sys

onNumba=True
np.seterr(over='raise')

if onNumba:
	from numba import jit, int32, prange
	from numba import cuda
	import numba

class CudaKernelWrapper:
	def __init__(self, kernel,cuda):
		self.kernel = kernel
		self.blocks = None
		self.threads = None
		self.cuda=cuda

	def __getitem__(self, config):
		blocks, threads = config
		if isinstance(blocks, int):
			blocks = (blocks,)
		if isinstance(threads, int):
			threads = (threads,)
		def r(*args):
			# Generar el producto cartesiano de los valores de las dimensiones de los bloques
			for b in itertools.product(*[range(dim) for dim in blocks]):
				for t in itertools.product(*[range(dim) for dim in threads]):
			# for b in range(blocks):
			# 	for t in range(threads):
					self.cuda.blockIdx.x=b
					self.cuda.threadIdx.x=t
					if len(b)==2:
						idx=(b[0]*threads[0]+t[0],b[1]*threads[1]+t[1])
					else:
						idx=b[0]*threads[0]+t[0]
					self.cuda.set(idx)
					self.kernel(*args)
		return r

	def __call__(self, *args, **kwargs):
		# Aquí puedes hacer cualquier preprocesamiento de los argumentos si es necesario
		if self.blocks is None or self.threads is None:
			self.kernel(*args, **kwargs)
		else:
			self.kernel[self.blocks, self.threads](*args, **kwargs)
class X:
	def __init__(self):
		self.x=0
class Atomic:
	def add(self,stop_,idx,num):
		stop_[idx]+=num
	def max(self,stop_,idx,num):
		stop_[idx]=max(stop_[idx],num)
	def min(self,stop_,idx,num):
		stop_[idx]=min(stop_[idx],num)
class Cuda:
	def __init__(self):
		self.threadIdx=X()
		self.blockIdx=X()
		self.atomic=Atomic()
	def jit(self, func):
		return CudaKernelWrapper(func,self)
	def to_device(self, x):
		return x
	def grid(self, x):
		return self.idx
	def set(self, x):
		self.idx=x
	def syncthreads(self):
		pass
	def synchronize(self):
		pass
cpu=Cuda()
if not onNumba:
	cuda=cpu

if onNumba:
	from numba import jit, int32, prange
	from numba import cuda
	import numba

class LineParse:
	def __init__(self, tabs, name, dosPuntos, tipo, functions):
		self.tabs = tabs
		self.name = name
		self.dosPuntos = dosPuntos
		self.tipo = tipo
		self.functions = functions
		self.children = []
	def addChildren(self, child):
		self.children.append(child)

	def __repr__(self):
		return f"LineParse(tabs={self.tabs}, name='{self.name}', dosPuntos='{self.dosPuntos}', tipo='{self.tipo}', functions={self.functions})"

class Util:
	def signo(a):
		return 1 if a>=0 else -1
	
	def angle(origin,xy):
		a= np.arctan2(xy[1]-origin[1],xy[0]-origin[0])
		if a<0:
			a+=2*np.pi
		return a
	
	def resta(a, b):
		return a[0]-b[0], a[1]-b[1]

	def productoEscalar(a, b):
		return a[0]*b[0]+a[1]*b[1]

	def distancia(a,b):
		return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
	def distancia1(a):
		return np.sqrt(a[0]**2+a[1]**2)
	
	def proyeccion(p, a, b):
		d1 = Util.resta(b, a)
		dd1=Util.distancia1(d1)
		d2 = Util.resta(p, a)
		x = Util.productoEscalar(d1, d2)/dd1

		# z del producto vectorial entre d1 y d2
		z = d1[0]*d2[1]-d1[1]*d2[0]
		
		if x<0:
			return a,(x,Util.distancia(a,p),z),True,True
		if x>dd1:
			return b,(x,Util.distancia(b,p),z),True,False
		pp=(a[0]+d1[0]/dd1*x, a[1]+d1[1]/dd1*x)
		y=Util.distancia(pp,p)
		return pp,(x,y,z),False,False

class Programemory:
	def __init__(self, p, fileName=__file__):
		self.p=p
		self.fileName = fileName
		self.function={}
		self.allObjects=[]

	def param(self,obj,name,device=True):
		p=[]
		for a in self.function[name]:
			if onNumba and device:
				p.append(getattr(obj,"d_"+a))
			else:
				p.append(getattr(obj,a))
		return p

	def all_to_device(self,obj=None):
		if obj is None:
			obj=self
		total=0
		for a in self.allObjects:
			a2=getattr(obj,a)
			a3=np.array(a2)
			total+=a3.size* a3.itemsize
			if onNumba:
				d=cuda.to_device(a2)
				setattr(obj,"d_"+a,d)
		print(f"Total: {total/1024/1024} MB")

	def createDS(self,obj=None):
		if obj is None:
			obj=self
		ds=self.dataStructure(self.p.get("dataStructure"))
		# Jerarquización
		lower=10
		tabs=[None]*10
		root=[]
		for line in ds:
			if line.tabs<lower:
				root=[]
				lower=line.tabs
			if lower==line.tabs:
				root.append(line)
			for i in range(line.tabs+1,10):
				tabs[i]=None
			tabs[line.tabs]=line
			for i in range(line.tabs-1,-1,-1):
				if tabs[i]:
					tabs[i].addChildren(line)
					break
		# Crea los shapes en un recorrido recursivo
		for r in root:
			self.createObject(r,obj,[])
		# crea los objetos
		# crea lo necesario por función
		# crea los dos tipos de llamadas a funciones
	def createObject(self,line,obj,shape):
		if line.dosPuntos:
			shape2=shape.copy()
			shape2.append(self.p[line.name])
			for c in line.children:
				self.createObject(c,obj,shape2)
		else:
			if line.tipo=="int32":
				npo=np.zeros(shape,dtype=np.int32)
			if line.tipo=="int64":
				npo=np.zeros(shape,dtype=np.int64)
			if line.tipo=="int16":
				npo=np.zeros(shape,dtype=np.int16)
			if line.tipo=="int8":
				npo=np.zeros(shape,dtype=np.int8)
			if line.tipo=="float16":
				npo=np.zeros(shape,dtype=np.float16)
			if line.tipo=="float32":
				npo=np.zeros(shape,dtype=np.float32)
			if line.tipo=="float64":	
				npo=np.zeros(shape,dtype=np.float64)
			setattr(obj,line.name,npo)
			self.allObjects.append(line.name)
			# create functions call
			for f in line.functions:
				if not f in self.function:
					self.function[f]=[]
				self.function[f].append(line.name)


	def dataStructure(self,text):
		lines = text.strip().split('\n')
		parsed_lines = [self.parse_line(line) for line in lines]
		return parsed_lines

	def parse_line(self,line):
		pattern = re.compile(r"^(?P<tabs>\t*)(?P<name>\w+):?(?P<tipo>\s\w+)?(?P<functions>(?:\s\w+)*)$")
		match = pattern.match(line)
		if match:
			tabs = match.group('tabs').count('\t')
			name = match.group('name')
			dosPuntos = ':' if line.strip().endswith(':') else ''
			tipo = match.group('tipo').strip() if match.group('tipo') else ''
			functions = match.group('functions').strip().split() if match.group('functions') else []
			return LineParse(tabs, name, dosPuntos, tipo, functions)
		return None
	
	def store(self, json, kvmap):
		# Leer todo el contenido del archivo
		with open(self.fileName, 'r', encoding='utf-8') as file:
			content = file.read()

		# Usar una expresión regular para encontrar el bloque que contiene el diccionario Python
		pattern = re.compile(rf'{re.escape(json)}\s*=\s*\{{[^\}}]*\}}', re.DOTALL)
		match = pattern.search(content)
		if match:
			# Evaluar el diccionario encontrado
			data = ast.literal_eval(match.group(0)[match.group(0).find('{'):])  # Extraer el diccionario
			# Modificar el diccionario con la nueva clave-valor
			for key, value in kvmap.items():
				data[key] = value
			# Convertir el diccionario actualizado de nuevo a un string en formato de diccionario de Python
			updated_dict_text = f'{json} = {data}'
			# Reemplazar el bloque antiguo en el contenido con el nuevo bloque
			updated_content = content[:match.start()] + updated_dict_text + content[match.end():]
		else:
			print(
					f'No se encontró el diccionario {json} en el archivo {self.fileName}')
			return
		# Sobrescribir el archivo con el nuevo contenido
		#print(updated_content)
		with open(self.fileName, 'w', encoding='utf-8') as file:
			file.write(updated_content)


class Data:
	def __init__(self, drnumba,args,param,dtype):
		self.drnumba=drnumba
		self.index=args[:-1]
		self.name=args[-1]
		self.param=param
		self.dtype=dtype
		for i,index in enumerate(self.index):
			if not index in self.drnumba._index:
				self.drnumba._index[index]=[]
			self.drnumba._index[index].append((self.name,i))

		if param==None:
			obj=getattr(self.drnumba.obj,self.name)
			if dtype==None:
				d=cuda.to_device(obj)
				setattr(self.drnumba.obj,"d_"+self.name,d)

	def to_device(self):
		if onNumba:
			obj=getattr(self.drnumba.obj,self.name)
			if hasattr(obj,"d_"+self.name):
				gpu=getattr(obj,"d_"+self.name)
				gpu.copy_to_device(obj)
			else:				
				d=cuda.to_device(obj)
				setattr(self.drnumba.obj,"d_"+self.name,d)

	def to_host(self):
		if onNumba:
			gpu=getattr(self.drnumba.obj,"d_"+self.name)
			obj=getattr(self.drnumba.obj,self.name)
			gpu.copy_to_host(obj)
			# d=gpu.copy_to_host()
			# setattr(self.drnumba.obj,self.name,d)

class EditFile:
	def __init__(self,name):
		self.name=name
		# read content.
		self.error=False
		with open(self.name,'r') as file:
			try:
				self.content=file.read().split("\n")
			except:
				self.error=True
	def editClass(self,name):
		return EditClass(self,name)
	
	def save(self):
		if self.error:
			return 
		with open(self.name, 'w') as file:
			file.write("\n".join(self.content))
		module_name = self.name.replace(".py", "")
		if module_name in sys.modules:
			importlib.reload(sys.modules[module_name])

class EditClass:
	def __init__(self,file,name):
		self.editFile=file
		if file.error:
			return 
		self.name=name
		# parase python file to get class strings
		self.content=[]
		self.start=-1
		for i,l in enumerate(file.content):
			if "class "+name in l:
				self.content=copyBlock(file.content,i)
				self.start=i
				break
			# # exit when tabs or spaces is 0
			# # count number of tabs or spaces in the l 
			# tabs=len(l)-len(l.lstrip())
			# if i==self.start:
			# 	self.tabs=tabs
			# else:
			# 	if self.start>=0 and tabs==self.tabs and l.lstrip()!="":
			# 		break
			# if self.start>=0:
			# 	self.content.append(l)

	def editMethod(self,name):
		return EditMethod(self,name)
	
	def save(self):
		content=self.editFile.content
		# Primero busca si está
		for i,l in enumerate(content):
			if l==self.content[0]:
				fromText=copyBlock(self.editFile.content,i)
				context2=content[:i-1]+self.content+content[i+len(fromText):]
				self.editFile.content=context2
				return 
				
		raise Exception(f"Please complete the implementation of class {self.name} in {self.editFile.name}")
		# context2=content[:inLine]+self.content+content[inLine:]
		# self.editFile.content=context2
	

def copyBlock(content, start):
	l=content[start]
	r=[l]
	tabs=len(l)-len(l.lstrip())
	for i in range(start+1,len(content)):
		l=content[i]
		tabs2=len(l)-len(l.lstrip())
		if tabs2<=tabs and l.lstrip()!="":
			break
		r.append(l)
	return r
	
class EditMethod:
	def __init__(self,editClass,name):
		self.editClass=editClass
		if editClass.editFile.error:
			return
		self.name=name
		# parase python file to get class strings
		self.content=[]
		self.start=-1
		for i,l in enumerate(self.editClass.content):
			if "def "+name+"(" in l:
				self.start=i

			self.content=copyBlock(self.editClass.content,self.start)
			# exit when tabs or spaces is 0
			# count number of tabs or spaces in the l 
			# tabs=len(l)-len(l.lstrip())
			# if i==self.start:
			# 	self.tabs=tabs
			# else:
			# 	if self.start>=0 and tabs<=self.tabs and l.lstrip()!="":
			# 		break
			# if self.start>=0:
			# 	self.content.append(l)

	def toCpu(self,module,name):
		k=EditMethod(self.editClass,module+"_"+name)
		k.content=list(self.content)
		k.editFile=self.editFile
		for i,l in enumerate(k.content):
			k.content[i]=l.replace("cuda.","cpu.").replace(module,module+"_CPU")
		#k.content[1]=k.content[1].replace("def "+self.name,"def "+module+"_CPU_"+name)
		k.name=name
		return k

	def toKernel(self,name):
		# Rename
		k=EditMethod(self.editClass,self.name)
		k.content[0]=k.content[0].replace("def "+self.name,"def "+name)
		self.name=name

		tabs=len(k.content[0])-len(k.content[0].lstrip())
		# remove initial tabs
		for i,l in enumerate(k.content):
			k.content[i]=l[tabs:]

		# add @cuda.jit
		k.content.insert(0,f"@cuda.jit")
		k.name=name

		return k

	def save(self,inLine):
		if hasattr(self,"editFile"):
			content=self.editFile.content
		else:
			content=self.editClass.content
		# Primero busca si está
		for i,l in enumerate(content):
			if l==self.content[0]:
				i+=1
				l1p=content[i].split("(")[0]
				sl1p=self.content[1].split("(")[0]
				if l1p==sl1p:
					fromText=copyBlock(self.editFile.content,i)
					context2=content[:i-1]+self.content+content[i+len(fromText):]
					if hasattr(self,"editFile"):
						self.editFile.content=context2
					else:	
						self.editClass.content=context2
					return 
		# si no está lo guarda en el punto recomendado
		context2=content[:inLine]+self.content+content[inLine:]
		if hasattr(self,"editFile"):
			self.editFile.content=context2
		else:	
			self.editClass.content=context2
				

class DrNumba:
	def __init__(self, file):
		self.file=file
		# if not exists create file
		if not os.path.exists(file):
			with open(file, 'w') as file:
				file.write("from drnumba import *\n")
		self.kernelFile=EditFile(self.file)
	def dr(self,obj):
		if hasattr(obj,"drnumba"):
			return obj.drnumba
		dr2=DrNumba2(self,obj)
		obj.dr=dr2
		return dr2

simple_types = (int, float, str, bool, type(None))

def is_simple(var):
    return isinstance(var, simple_types)

class DrNumba2:
	def __init__(self,gestor, obj):
		self.gestor=gestor
		self.obj=obj
		# capture caller file name
		fileName = inspect.stack()[2].filename
		# read file name
		self.editFile=EditFile(fileName)
		self._data=[]
		self._index={}

	def data(self,*args,param=None,dtype=None):
		d=Data(self,args,param,dtype)
		self._data.append(d)

	def to_device(self,*lista):
		for d in self._data:
			if d.name in lista:
				d.to_device()

	def to_host(self,*lista):
		for d in self._data:
			if d.name in lista:
				d.to_host()

	def prepareReplaces2(self,objName,fName,replace,pre="",includeParam=True):
		data=[]
		data2=[]
		for d in self._data:
			if d.dtype==None:
				si=d.param==None
				if not si and includeParam:
					if fName in d.param:
						si=True
				if si:
					if replace!=None:
						replace.append((objName+"."+d.name,pre+d.name))	
					data.append(pre+d.name)
					data2.append(d)
			else:
				raise Exception("Programed, but not understood")
				d2=getattr(self.obj,d.name)
				data3,data4=d2.dr.prepareReplaces(objName+"."+d.name,fName,replace,pre+d.name+"_",className=d.dtype)	
				data.extend(data3)
				data2.extend(data4)
		return data,data2

	def prepareReplaces(self,objName,fName,replace,pre="",className=None):
		data,data2=self.prepareReplaces2(objName,fName,replace,pre)
		replace.append((objName+")",",".join(data)+")"))
		replace.append((objName+",",",".join(data)+","))
		def esUnaFuncion(s):
			primerParentesisEn=s.find("(")
			fNameIn=s[:primerParentesisEn]
			dataIn,data2In=self.prepareReplaces2(objName,fNameIn,None,pre,includeParam=False)
			r=className+"_"+s[:primerParentesisEn]+"("+",".join(dataIn)+","+s[primerParentesisEn+1:]
			return r
		replace.append((objName+".",esUnaFuncion))
		return data,data2
		
	def function(self,name,*findex,pre=None,post=None):
		# dn.function("add","y",post=self.postAdd)
		# self.obj is a class, gets his name
		objName=self.obj.__class__.__name__

		frame = inspect.currentframe().f_back
		nombre_llamador = frame.f_code.co_name

		editClass=self.editFile.editClass(objName)
	
		editMethod=editClass.editMethod(name)

		if editMethod.start==-1:	
			editMethod.content[0]="\tdef "+name+"(self):"
			field0,dim0=self._index[findex[0]][0]
			if len(findex)==1:	
				editMethod.content.append("\t\tidx=cuda.grid(1)")
				editMethod.content.append(f"\t\tif idx>=self.{field0}.shape[{dim0}]:")
			elif len(findex)==2:
				field1,dim1=self._index[findex[1]][0]
				editMethod.content.append("\t\tidx,idy=cuda.grid(2)")
				editMethod.content.append(f"\t\tif idx>=self.{field0}.shape[{dim0}] or idy>=self.{field1}.shape[{dim1}]:")
			else:
				raise Exception("Not implemented")
			editMethod.content.append("\t\t\treturn")
			editMethod.content.append("")	

			# busca una funcion que contenga def nombre_llamador
			for i,l in enumerate(editClass.content):
				if "def "+nombre_llamador in l:
					bloque=copyBlock(editClass.content,i)
					break
				
			editMethod.save(i+len(bloque))
			editMethod.editClass.save()
			editMethod.editClass.editFile.save()
			raise Exception(f"Please complete the implementation of {objName}.{name} in {self.editFile.name}")


		# Bloque que cambia para guardar la función en un fichero externo
		editKernel=editMethod.toKernel(objName+"_"+name)
		replace=[]
		data,data2=self.prepareReplaces("self",name,replace,className=editClass.name)
		for i,c in enumerate(editKernel.content):
			for r in replace:
				if isinstance(r[1],str):
					c=c.replace(r[0],r[1])
				else:
					while True:
						inPos=c.find(r[0])
						if inPos>=0:
							c=c[:inPos]+r[1](c[inPos+len(r[0]):])
						else:
							break
			editKernel.content[i]=c
		editKernel.editFile=self.gestor.kernelFile
		#editKernel.save(editKernel.editClass.start+len(editKernel.editClass.content))
		editKernel.save(len(editKernel.editFile.content))

		editCPU=editKernel.toCpu(objName,name)
		editCPU.save(len(editKernel.editFile.content))

		editKernel.editFile.save()


		fname=f"{objName}_{name}"

		def f(*args,cpu=False):
			nonlocal fname
			if cpu:
				fname=f"{objName}_CPU_{name}"
			if pre:
				pre()

			args2=[]
			iparam=0
			for d in data2:
				if d.param and name in d.param:
					aux=args[iparam]
					if is_simple(aux):
						d2=aux
					else:
						d2=cuda.to_device(args[iparam])
					iparam+=1
					args2.append(d2)
				else:
					# Se supone transferido a cuda.
					if len(d.index)==0:
						d2=getattr(d.drnumba.obj,d.name)
					else:
						if cpu:
							d.drnumba.to_host(d.name)
							d2=getattr(d.drnumba.obj,d.name)
						else:
							d2=getattr(d.drnumba.obj,"d_"+d.name)
					args2.append(d2)
			# get global function by name
			# ruta, archivo = os.path.split(self.editFile.name)
			# im=__import__(archivo.replace(".py",""))

			module_name = self.gestor.file.replace(".py", "")
			im = __import__(module_name)
			if not hasattr(im,fname):
				if module_name in sys.modules:
					im = importlib.reload(sys.modules[module_name])
				else:
					raise Exception(f"Error in {objName}.{name} in {self.gestor.file}")
			f=getattr(im,fname)


			# Hay que hallar la dimensión o dimensiones que ejecuta

			dim=[]
			mul=1
			for a in findex:
				for name2,index in self._index[a]:
					encontrado=False
					for d2 in self._data:
						if d2.name==name2 and hasattr(d2.drnumba.obj,d2.name):
							d3=getattr(d2.drnumba.obj,d2.name)
							aux=d3.shape[index]
							dim.append(aux)
							mul*=aux
							encontrado=True
							break
					if encontrado:
						break

			if len(dim)==1:
				if onNumba:
					threadsperblock = 1024
				else:
					threadsperblock = 1
				while True:
					blocks_per_grid = (dim[0] - 1) // threadsperblock+1
					if blocks_per_grid>=128:
						break
					threadsperblock//=2
			elif len(dim)==2:
				if onNumba:
					threadsperblock = (32,32)
				else:
					threadsperblock = (1,1)
				while True:
					blocks_per_grid = ((dim[0] - 1) // threadsperblock[0]+1, (dim[1] - 1) // threadsperblock[1]+1)
					if blocks_per_grid[0]*blocks_per_grid[1]>=128:
						break
					threadsperblock=(threadsperblock[0]//2,threadsperblock[1]//2)
			else:
				raise Exception("Not implemented")

			from autoforegpu import Dentro
			fuera=Dentro(name)
			f[blocks_per_grid,threadsperblock](*args2)
			fuera()
			#cuda.synchronize()

			if post:
				post()
			
		setattr(self.obj,name,f)
		return
		
		self.content+=line
		#write content in self.fileName
		with open(self.fileName, 'w') as file:
			file.write(self.content)
		
		raise Exception(f"Please complete the implementation of {objName}_{name} in {self.fileName}")
		
