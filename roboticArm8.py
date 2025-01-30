import pygame
import math
from autoforenumpy2 import AutoFore
import time
import random
import numpy as np
import matplotlib
import os
import csv
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import itertools
from collections import defaultdict

class Parameters:
	def __init__(self,graphic=True,population=10,segments=5,seed=0):
		self.width = 600
		self.height = 600
		self.white = (255, 255, 255)
		self.red=(255,0,0)
		self.green=(0,255,0)
		self.blue=(0,0,255)
		self.yellow=(255,255,0)
		self.black=(0,0,0)
		self.circle_radius = 5  # Radio del círculo
		self.max_angle_velocity = 0.01  
		self.graphic=graphic
		self.stereo=False # True to use stereo vision, faster convergence
		self.population=population
		self.convergence=0.03
		self.changePositionEach=100
		self.changePopulationEach=200
		self.checkExitEach=100
		self.population=population
		self.segments=segments
		self.segment_min_size=50
		self.segment_max_size=200
		self.random_seed=seed


class Transform:
	def __init__(self,nn):
		self.nn=nn
		self.matrix = None
	def rotate(self, angle):
		nn=self.nn
		if self.matrix==None:
			self.matrix= [
				[nn.const(0), nn.const(0), nn.const(0)],
				[nn.const(0), nn.const(0), nn.const(0)],
				[nn.const(0), nn.const(0), nn.const(1)]
			]	
		
		self.matrix[0][0].assign(angle.cos())
		self.matrix[0][1].assign(-angle.sin())
		self.matrix[1][0].assign(angle.sin())
		self.matrix[1][1].assign(angle.cos())
	def translate(self, translation):
		nn=self.nn
		self.matrix = [
			[nn.const(1), nn.const(0), translation[0]],
			[nn.const(0), nn.const(1), translation[1]],
			[nn.const(0), nn.const(0), nn.const(1)]
		]

class Arm:
	def __init__(self,p,nn,segment_length):
		self.p=p
		self.nn=nn
		#self.color=color
		self.size=Transform(nn)
		self.segment_length=segment_length
		self.size.translate((0,self.segment_length))
		self.rota=Transform(nn)
		self.children=[]
		self.angle=nn.val(0)

	def setAngle(self,angle):
		#nn=self.nn
		self.angle.assign(angle)
		self.rota.rotate(self.angle)

	def draw(self,screen,center,id,color):
		b=self.matrix_multiplication(center,self.rota.matrix)
		c= self.matrix_multiplication(b,self.size.matrix)
		
		self.x=c[0][2]
		self.y=c[1][2]
		
		if self.p.graphic:
			pygame.draw.line(screen, color, self._fromPoint(center,id), self._fromPoint(c,id) , 5)

		for child in self.children:
			child.draw(screen,c,id,color)

		
	
	def _fromPoint(self,point,id):
		x=point[0][2].value(id)
		y=point[1][2].value(id)
		inicial=100
		altura=inicial-5*id
		# for i in range(id):
		# 	altura/=2
		#self.p.width
		#self.p.height
		return (x+inicial-altura,y-inicial+altura)

	def matrix_multiplication(self,A, B):
		if len(A[0]) != len(B):
			raise ValueError("Number of columns in A should be equal to the number of rows in B")

		# Obtener dimensiones
		rows_A, cols_A = len(A), len(A[0])
		rows_B, cols_B = len(B), len(B[0])
		
		# Inicializar matriz de resultado con ceros
		result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

		for i in range(rows_A):
			for j in range(cols_B):
				for k in range(cols_A): 
					result[i][j] += A[i][k] * B[k][j]

		return result

	def addChildren(self,child):
		self.children.append(child)

class Eye:
	def __init__(self,screen,x,y):
		self.screen=screen
		self.radius = 100
		self.focus=(x,y)

	def draw(self):
		black=(0,0,0)
		pygame.draw.circle(self.screen, black, self.focus, 5)
		pygame.draw.circle(self.screen, black, self.focus, self.radius, 1)

	def error(self,c):
		if self.screen!=None:
			blue=(0,0,255)
			pygame.draw.line(self.screen,blue,(c.x.value(0),c.y.value(0)),self.focus,1)
		m=(c.y-self.focus[1])/(c.x-self.focus[0])
		# pendiente a ángulo
		angle=m.atan()	
		#angle=math.atan2(c.y.value(0)-focus_cam[1],c.x.value(0)-focus_cam[0])

		error=angle-angle.value(0)
		#error=angle.tanh()-np.tanh(angle.value(0))
		return error


class RoboticArm:
	def __init__(self, p):
		self.p = p
		if p.graphic:
			pygame.init()
			screen = pygame.display.set_mode((p.width, p.height))
			pygame.display.set_caption("Robotic Arm")
		else:
			screen=None

		if p.stereo:
			eyes=[Eye(screen,p.width,p.height//3),Eye(screen,p.width,p.height//3*2)]
		else:
			eyes=[Eye(screen,p.width,p.height//2)]

		changePositionEach=p.changePositionEach
		changePopulationEach=p.changePopulationEach
		pulation=p.population
		segments=p.segments

		ronda=0
		seed=random.Random(p.random_seed)
		nn=AutoFore(gradientes=2*segments+1,variables=int(200*segments),poblacion=pulation,seed=seed)

		center=Transform(nn)
		center.translate((nn.const(p.width//3),nn.const(p.height//2)))

		arm=[]
		for i in range(segments):
			ma=nn.random(p.segment_min_size,p.segment_max_size).differentiable()
			aa=nn.random(0,math.pi*2).differentiable()
			a=Arm(p,nn,ma)
			a.setAngle(aa)
			if len(arm)>0:
				arm[-1].addChildren(a)
			arm.append(a)		

		a=arm[0]
		b=arm[-1]
		
		circle_position = (100,100)
		error=None

		running = True
		willDie=[]
		since=time.time()
		while running:
			if p.graphic:
				for event in pygame.event.get():
					if event.type == pygame.QUIT:
						running = False
					elif event.type == pygame.MOUSEBUTTONDOWN:
						click_x, click_y = event.pos
						circle_position = (click_x, click_y)  # Guardar posición del clic para dibujar el círculo
					
			ronda+=1
			if ronda%changePositionEach==0:
				circle_position = (nn.random(0,p.width).value(0),nn.random(0,p.height).value(0))

			if p.graphic:
				screen.fill(p.white)


			for pob in range(pulation-1,-1,-1):
				if pob in willDie:
					color=p.red
				else:
					color=p.green
				if pob==0:
					color=p.blue
				a.draw(screen,center.matrix,pob,color=color)
				if not error is None:
					error=error+0

			if p.graphic:
				for eye in eyes:
					eye.draw()
			
			suberror=nn.val(0)
			for c in arm:
				for eye in eyes:
					errorAux=eye.error(c)
					error2=errorAux*errorAux
					suberror+=error2
			if error is None:
				error=suberror
			else:
				error+=suberror


			suberror.learn()
			#nn.applyDelta(learning_rate,suberror.id2)
			if ronda%p.checkExitEach==0:
				minId=error.minId()
				minErrorPerSegment=error.value(minId)/segments
				print("Segments:",segments,"Population:",pulation,"Seed:",p.random_seed,"Time:",round(time.time()-since,2),"Rounds:",ronda,"Segment Error:",minErrorPerSegment)
				#print( "Time:",round(time.time()-since,2),"Rounds:",ronda,"Segment Error:",minErrorPerSegment)
				if minErrorPerSegment<p.convergence:
					self.time=time.time()-since
					self.rounds=ronda
					self.segmentError=minErrorPerSegment
					break

			doit=ronda%changePopulationEach==0
			willDie=error.geneticAlgorithm(doit=doit,killdown=(self.p.population-1)//2)

			if ronda%changePopulationEach==0:
				error=None
			# if doit:
			# 	minId=error.minId()
			# 	minErrorPerSegment=error.value(minId)/segments
			# 	print("Time:",round(time.time()-since,2),"Rounds:",ronda,"Segment Error:",minErrorPerSegment)
			# 	if minErrorPerSegment<p.convergence:
			# 		break
			# 	error=None

			if circle_position:
				# halla el vector normalizado
				x_n=circle_position[0]-b.x.value(0)
				y_n=circle_position[1]-b.y.value(0)
				norm=math.sqrt(x_n**2+y_n**2)
				#norm=20000
				x_n=x_n/norm
				y_n=y_n/norm
				# lo dibuja
				#pygame.draw.line(screen, p.black, (b.x.value(0),b.y.value(0)), (b.x.value(0)+x_n*30,b.y.value(0)+y_n*30) , 1)

				# calcula el producto escalar 
				for c in arm:
					angle_grad_y=b.y.get(c.angle,0)
					angle_grad_x=b.x.get(c.angle,0)

					producto_escalar=x_n*angle_grad_x+y_n*angle_grad_y
					angle_velocity=p.max_angle_velocity*norm/100
					#angle_velocity=p.max_angle_velocity
					if producto_escalar>angle_velocity:
						producto_escalar=angle_velocity
					if producto_escalar<-angle_velocity:
						producto_escalar=-angle_velocity
					c.setAngle(c.angle+producto_escalar)

				if p.graphic:
					pygame.draw.circle(screen, p.black, circle_position, p.circle_radius)

			# Actualizar la ventana
			if p.graphic:
				pygame.display.flip()
			nn.noMoreConst()

			

		pygame.quit()


# Suponiendo que RoboticArm y Parameters ya están definidos
def run_robotic_arm(population):
    """Función para inicializar el RoboticArm con el parámetro population."""
    ra = RoboticArm(Parameters(graphic=False, population=population, segments=3))
    return population, ra.time, ra.rounds, ra.segmentError

# Suponiendo que RoboticArm y Parameters ya están definidos
def run_robotic_arm(params):
	"""Función para inicializar el RoboticArm con múltiples parámetros."""
	population, segment, seed = params
	ra = RoboticArm(Parameters(graphic=False, population=population, segments=segment, seed=seed))
	return population, segment, seed, ra.time, ra.rounds


def save_results_to_file(results, filename="results.csv"):
	"""Guardar los resultados en un archivo CSV."""
	with open(filename, mode='w', newline='') as file:
		writer = csv.writer(file)
		# Escribir encabezados
		writer.writerow(["Population", "Segments", "Seed", "Time", "Rounds"])
		
		# Escribir filas
		writer.writerows(results)

# Configuración para fuentes en PDF y PS
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def plot_results(filename="results.csv", metric="times"):
	"""
	Plot results with the option to select the metric ("times" or "rounds").
	"""
	if not os.path.exists(filename):
		print(f"The file {filename} does not exist.")
		return

	if metric not in ["times", "rounds"]:
		print("Invalid metric. Use 'times' or 'rounds'.")
		return

	# Data organized by population and segments
	data = defaultdict(lambda: defaultdict(list))

	# Read the file
	with open(filename, mode='r') as file:
		reader = csv.DictReader(file)
		for row in reader:
			population = int(row["Population"]) - 1
			segment = int(row["Segments"])
			time = float(row["Time"]) / float(row["Population"])
			rounds = float(row["Rounds"]) / 100
			# Group by population and segment
			data[segment][population].append((time, rounds))

	plt.figure(figsize=(12, 8))

	# Plot for each segment
	for segment, segment_data in data.items():
		populations = sorted(segment_data.keys())
		means = []
		stds = []

		for population in populations:
			# Extract data for all seeds for this segment and population
			values = segment_data[population]
			if metric == "times":
				values_metric = [v[0] for v in values]
			else:  # metric == "rounds"
				values_metric = [v[1] for v in values]

			# Calculate mean and standard deviation
			means.append(np.mean(values_metric))
			stds.append(np.std(values_metric))

		# Plot the selected metric (mean and interval)
		plt.errorbar(
			populations, means, yerr=stds, label=f"{segment}", marker='o'
		)

	# Configure titles and labels based on the metric
	if metric == "times":
		#plt.title("Average Time by Population and Segments")
		plt.ylabel("Time (Average per Population)")
	else:  # metric == "rounds"
		#plt.title("Average Rounds by Population and Segments")
		plt.ylabel("Rounds (Average per Population)")

	plt.xlabel("Population")
	plt.xticks(populations)  # Ensure only integer values appear on the x-axis
	plt.legend(title="Segments")
	plt.grid(True)
	plt.show()



if __name__ == '__main__':
	fileName="results3.csv"
	plot_results(filename=fileName, metric="times")
	plot_results(filename=fileName, metric="rounds")

	# Obtener el número de CPUs disponibles
	num_cpus = os.cpu_count()

	# Crear un rango de valores para population
	populations = range(2, 7)
	segments = range(1, 7)
	seeds=[123,456,789]

	# populations = range(2, 4)
	# segments = range(1, 3)
	# seeds=[123,456]

	# Generar todas las combinaciones de parámetros
	parameter_combinations = list(itertools.product(populations, segments, seeds))

	RoboticArm(Parameters(graphic=True, population=10, segments=4, seed=123))

	# Ejecutar en paralelo con todos los núcleos disponibles
	results = []
	# Ejecutar en paralelo con todos los núcleos disponibles
	with ProcessPoolExecutor(max_workers=num_cpus) as executor:
		results = list(executor.map(run_robotic_arm, parameter_combinations))


	# Change the file name to no overwrite the previous results
	fileName="results1.csv"
	# Guardar resultados en un archivo CSV
	save_results_to_file(results, filename=fileName)

	# Graficar los resultados
	plot_results(filename=fileName, metric="times")
	plot_results(filename=fileName, metric="rounds")
