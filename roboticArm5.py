# copiada de kinematics el 29/12/2024
# adaptada a autofore
# deriva de clock2 y es backpropagation2->neuronalprogrammig4

import pygame
import math
from autoforenumpy import AutoFore
import time

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
	def __init__(self,p,nn,segment_length,color):
		self.p=p
		self.nn=nn
		self.color=color
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

	def draw(self,screen,center,id,tono=1):
		b=self.matrix_multiplication(center,self.rota.matrix)
		c= self.matrix_multiplication(b,self.size.matrix)
		
		self.x=c[0][2]
		self.y=c[1][2]
		
		pygame.draw.line(screen, (self.color[0]*tono,self.color[1]*tono,self.color[2]*tono), self._fromPoint(center,id), self._fromPoint(c,id) , 5)

		for child in self.children:
			child.draw(screen,c,id,tono)

		
	
	def _fromPoint(self,point,id):
		return [point[0][2].value(id),point[1][2].value(id)]

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


class RoboticArm:
	def __init__(self, p):
		self.p = p
		nn=AutoFore(gradientes=6,variables=500,poblacion=2)

		# Inicializar pygame
		pygame.init()
		screen = pygame.display.set_mode((p.width, p.height))
		pygame.display.set_caption("Robotic Arm")


		center=Transform(nn)
		center.translate((nn.const(p.width//3),nn.const(p.height//2)))

		radio_ojo=100
		focus_cam=(p.width,p.height//2)

		ma=nn.random(50,200).differentiable()
		aa=nn.random(0,math.pi*2).differentiable()
		md=nn.random(50,200).differentiable()
		ad=nn.random(0,math.pi*2).differentiable()
		mb=nn.random(50,200).differentiable()
		ab=nn.random(0,math.pi*2).differentiable()

		a=Arm(p,nn,ma,p.red)
		a.setAngle(aa)

		d=Arm(p,nn,md,p.green)
		d.setAngle(ad)
		a.addChildren(d)

		b=Arm(p,nn,mb,p.blue)
		b.setAngle(ab)
		d.addChildren(b)

		
		circle_position = (100,100)
		

		running = True
		while running:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					running = False
				elif event.type == pygame.MOUSEBUTTONDOWN:
					click_x, click_y = event.pos

					if focus_cam==None:
						focus_cam=(click_x,click_y)
					else:
						circle_position = (click_x, click_y)  # Guardar posición del clic para dibujar el círculo
					
			screen.fill(p.white)

			since=time.time()

			a.draw(screen,center.matrix,0,tono=1)
			a.draw(screen,center.matrix,1,tono=0.5)

			if focus_cam:
				pygame.draw.circle(screen, p.black, focus_cam, 5)
				pygame.draw.circle(screen, p.black, focus_cam, radio_ojo, 1)

			for c in [a,d,b]:
				angle_grad_y=b.y.get(c.angle,0)
				angle_grad_x=b.x.get(c.angle,0)

				norm=math.sqrt(angle_grad_x**2+angle_grad_y**2)/20
				norm=1
				#print(angle_grad_x,angle_grad_y)

				#pygame.draw.line(screen, c.color, (b.x.value(0),b.y.value(0)), (b.x.value(0)+angle_grad_x/norm,b.y.value(0)+angle_grad_y/norm) , 1)


				if focus_cam:
					pygame.draw.line(screen,c.color,(c.x.value(0),c.y.value(0)),focus_cam,1)
					m=(c.y-focus_cam[1])/(c.x-focus_cam[0])
					# pendiente a ángulo
					angle=m.atan()	
					#angle=math.atan2(c.y.value(0)-focus_cam[1],c.x.value(0)-focus_cam[0])

					#if c==a:
					error=angle-angle.value(0)
					error2=error*error
					error2.error2Delta()

					# calcula el vector normalizado focus->c
					x_n=c.x.value(0)-focus_cam[0]
					y_n=c.y.value(0)-focus_cam[1]
					# Normaliza el vector
					norm=math.sqrt(x_n**2+y_n**2)
					x_n=x_n/norm*radio_ojo+focus_cam[0]
					y_n=y_n/norm*radio_ojo+focus_cam[1]


					# calcula el punto medio
					#middle=((c.x.value(0)+focus_cam[0])//2,(c.y.value(0)+focus_cam[1])//2)
					# draw a label m in the middle
					# font = pygame.font.Font(None, 36)
					# text = font.render(str(round(angle.value(0),2)), True, c.color)
					# screen.blit(text, middle)

					derivate=angle.get(c.angle,0)*100
					#derivate=angle.get(c.segment_length)*10000
					
					# draw a ortogonal line from the middle with module derivate
					# Calcula el vector ortogonal (derivada perpendicular)
					orthogonal_angle = angle.value(0) - math.pi / 2  # Ángulo perpendicular
					length = derivate  # Longitud del vector ortogonal
					end_point = (
						x_n + length * math.cos(orthogonal_angle),
						y_n + length * math.sin(orthogonal_angle)
					)
					
					# Dibuja la línea ortogonal desde el punto medio
					pygame.draw.line(screen, c.color, (x_n,y_n), end_point, 1)


			nn.applyDelta(0.001)

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
				# mide el tiempo
				# star=time.time()
				for c in [a,d,b]:
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

				# calcula el tiempo que tardó
				# end=time.time()
				# print(end-star)
				pygame.draw.circle(screen, p.black, circle_position, p.circle_radius)

			print("Tiempo:",time.time()-since)
			# Actualizar la ventana
			pygame.display.flip()
			nn.noMoreConst()

		pygame.quit()

if __name__ == '__main__':
	RoboticArm(Parameters())
