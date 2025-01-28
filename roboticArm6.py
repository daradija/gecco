# copiada de kinematics el 29/12/2024
# adaptada a autofore
# deriva de clock2 y es backpropagation2->neuronalprogrammig4

import pygame
import math
from autoforenumpy import AutoFore
import time
import random
import numpy as np

class Parameters:
	def __init__(self):
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

class Transform:
	def __init__(self,nn):
		self.nn=nn
		self.matrix = None
		# [
		# 	[nn.const(1), nn.const(0), nn.const(0)],
		# 	[nn.const(0), nn.const(1), nn.const(0)],
		# 	[nn.const(0), nn.const(0), nn.const(1)]
		# ]
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
				# Inicializar pygame
		pygame.init()
		screen = pygame.display.set_mode((p.width, p.height))
		pygame.display.set_caption("Robotic Arm")

		#eyes=[Eye(screen,p.width,p.height//3),Eye(screen,p.width,p.height//3*2)]
		eyes=[Eye(screen,p.width,p.height//2)]

		changePositionEach=100
		round=0
		poblacion=2
		segments=5

		nn=AutoFore(gradientes=2*segments+1,variables=int(500/3*segments),poblacion=poblacion)



		center=Transform(nn)
		center.translate((nn.const(p.width//3),nn.const(p.height//2)))

		learning_rate=nn.random(0.001,0.001).differentiable()

		arm=[]
		for i in range(segments):
			ma=nn.random(50,200).differentiable()
			aa=nn.random(0,math.pi*2).differentiable()
			color=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
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
		while running:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					running = False
				elif event.type == pygame.MOUSEBUTTONDOWN:
					click_x, click_y = event.pos
					circle_position = (click_x, click_y)  # Guardar posición del clic para dibujar el círculo
					
			round+=1
			if round%changePositionEach==0:
				circle_position = (nn.random(0,p.width).value(0),nn.random(0,p.height).value(0))

			screen.fill(p.white)

			since=time.time()

			for pob in range(poblacion-1,-1,-1):
				if pob in willDie:
					color=p.red
				else:
					color=p.green
				if pob==0:
					color=p.blue
				a.draw(screen,center.matrix,pob,color=color)
				if not error is None:
					error=error+0

			for eye in eyes:
				eye.draw()
			
			suberror=nn.val(0)
			for c in arm:
				for eye in eyes:
					errorAux=eye.error(c)
					error2=errorAux*errorAux
					suberror+=error2
					error2.error2Delta()
			if error is None:
				error=suberror
			else:
				error+=suberror
			nn.applyDelta(learning_rate,suberror.id2)
			doit=round%changePositionEach==0
			willDie=error.geneticAlgorithm(kill=0.5, doit=doit)
			if doit:
				error=None

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

				pygame.draw.circle(screen, p.black, circle_position, p.circle_radius)

			print("Tiempo:",time.time()-since)
			# Actualizar la ventana
			pygame.display.flip()
			nn.noMoreConst()

		pygame.quit()

if __name__ == '__main__':
	RoboticArm(Parameters())
