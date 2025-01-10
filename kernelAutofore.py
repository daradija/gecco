from drnumba import *
import numpy as np

@cuda.jit
def AutoFore_assign(value,delta,g,id,id_var,v):
	idx=cuda.grid(1)
	if idx>=value.shape[1]:
		return		
	value[id_var, idx] = v
	for i in range(g.shape[2]):
		g[id_var, idx, i] = 0
		id[id_var, idx,i] = -1


@cpu.jit
def AutoFore_CPU_assign(value,delta,g,id,id_var,v):
	idx=cpu.grid(1)
	if idx>=value.shape[1]:
		return		
	value[id_var, idx] = v
	for i in range(g.shape[2]):
		g[id_var, idx, i] = 0
		id[id_var, idx,i] = -1


@cuda.jit
def AutoFore_differentiable(value,delta,g,id,id_var):
	idx=cuda.grid(1)
	if idx>=value.shape[1]:
		return
	g[id_var, idx, 0] = 1
	id[id_var, idx, 0] = id_var
	delta[id_var, idx] = 0

@cpu.jit
def AutoFore_CPU_differentiable(value,delta,g,id,id_var):
	idx=cpu.grid(1)
	if idx>=value.shape[1]:
		return
	g[id_var, idx, 0] = 1
	id[id_var, idx, 0] = id_var
	delta[id_var, idx] = 0

@cuda.jit
def AutoFore_add(value,delta,g,id,dest,src1,src2):
	idx=cuda.grid(1)
	if idx>=value.shape[1]:
		return
	value[dest, idx] = value[src1, idx] + value[src2, idx] #
	for i in range(g.shape[2]):
		g[dest, idx, i] = g[src1, idx, i] #
		id[dest, idx, i] = id[src1, idx, i]
		if id[src1, idx, i]==-1:
			break
	for k in range(g.shape[2]):
		id2=id[src2, idx, k]
		if id2==-1:
			break
		i=-1
		g2=g[src2,idx,k] #
		min=g2 
		
		for j in range(g.shape[2]):
			idd=id[dest, idx, j]
			if idd==-1:
				i=j
				if i+1<g.shape[2]:
					id[dest, idx, i+1] = -1
				break
			if id2==idd:
				i=-1
				g[dest, idx, j] +=  g2
				break
			gd=g[dest, idx, j] 
			if abs(min)>abs(gd):
				min=gd
				i=idd
		if i!=-1:
			g[dest, idx, i] = g2
			id[dest, idx, i] = id2

@cpu.jit
def AutoFore_CPU_add(value,delta,g,id,dest,src1,src2):
	idx=cpu.grid(1)
	if idx>=value.shape[1]:
		return
	value[dest, idx] = value[src1, idx] + value[src2, idx] #
	for i in range(g.shape[2]):
		g[dest, idx, i] = g[src1, idx, i] #
		id[dest, idx, i] = id[src1, idx, i]
		if id[src1, idx, i]==-1:
			break
	for k in range(g.shape[2]):
		id2=id[src2, idx, k]
		if id2==-1:
			break
		i=-1
		g2=g[src2,idx,k] #
		min=g2 
		
		for j in range(g.shape[2]):
			idd=id[dest, idx, j]
			if idd==-1:
				i=j
				if i+1<g.shape[2]:
					id[dest, idx, i+1] = -1
				break
			if id2==idd:
				i=-1
				g[dest, idx, j] +=  g2
				break
			gd=g[dest, idx, j] 
			if abs(min)>abs(gd):
				min=gd
				i=idd
		if i!=-1:
			g[dest, idx, i] = g2
			id[dest, idx, i] = id2

@cuda.jit
def AutoFore_mul(value,delta,g,id,dest,src1,src2):
	idx=cuda.grid(1)
	if idx>=value.shape[1]:
		return
	value[dest, idx] = value[src1, idx] * value[src2, idx] #
	for i in range(g.shape[2]):
		g[dest, idx, i] = g[src1, idx, i]*value[src2, idx] #
		id[dest, idx, i] = id[src1, idx, i]
		if id[src1, idx, i]==-1:
			break
	for k in range(g.shape[2]):
		id2=id[src2, idx, k]
		if id2==-1:
			break
		i=-1
		g2=g[src2,idx,k] * value[src1, idx] #
		min=g2 
		
		for j in range(g.shape[2]):
			idd=id[dest, idx, j]
			if idd==-1:
				i=j
				if i+1<g.shape[2]:
					id[dest, idx, i+1] = -1
				break

				break
			if id2==idd:
				i=-1
				g[dest, idx, j] +=  g2
				break
			gd=g[dest, idx, j] 
			if abs(min)>abs(gd):
				min=gd
				i=idd
		if i!=-1:
			g[dest, idx, i] = g2
			id[dest, idx, i] = id2



@cpu.jit
def AutoFore_CPU_mul(value,delta,g,id,dest,src1,src2):
	idx=cpu.grid(1)
	if idx>=value.shape[1]:
		return
	value[dest, idx] = value[src1, idx] * value[src2, idx] #
	for i in range(g.shape[2]):
		g[dest, idx, i] = g[src1, idx, i]*value[src2, idx] #
		id[dest, idx, i] = id[src1, idx, i]
		if id[src1, idx, i]==-1:
			break
	for k in range(g.shape[2]):
		id2=id[src2, idx, k]
		if id2==-1:
			break
		i=-1
		g2=g[src2,idx,k] * value[src1, idx] #
		min=g2 
		
		for j in range(g.shape[2]):
			idd=id[dest, idx, j]
			if idd==-1:
				i=j
				if i+1<g.shape[2]:
					id[dest, idx, i+1] = -1
				break

				break
			if id2==idd:
				i=-1
				g[dest, idx, j] +=  g2
				break
			gd=g[dest, idx, j] 
			if abs(min)>abs(gd):
				min=gd
				i=idd
		if i!=-1:
			g[dest, idx, i] = g2
			id[dest, idx, i] = id2



@cuda.jit
def AutoFore_sub(value,delta,g,id,dest,src1,src2):
	idx=cuda.grid(1)
	if idx>=value.shape[1]:
		return
	value[dest, idx] = value[src1, idx] - value[src2, idx] #
	for i in range(g.shape[2]):
		g[dest, idx, i] = g[src1, idx, i] #
		id[dest, idx, i] = id[src1, idx, i]
		if id[src1, idx, i]==-1:
			break
	for k in range(g.shape[2]):
		id2=id[src1, idx, k]
		if id2==-1:
			break
		i=-1
		g2=-g[src2,idx,k] #
		min=g2 
		
		for j in range(g.shape[2]):
			idd=id[dest, idx, j]
			if idd==-1:
				i=j
				if i+1<g.shape[2]:
					id[dest, idx, i+1] = -1
				break

				break
			if id2==idd:
				i=-1
				g[dest, idx, j] +=  g2
				break
			gd=g[dest, idx, j] 
			if abs(min)>abs(gd):
				min=gd
				i=idd
		if i!=-1:
			g[dest, idx, i] = g2
			id[dest, idx, i] = id2

@cpu.jit
def AutoFore_CPU_sub(value,delta,g,id,dest,src1,src2):
	idx=cpu.grid(1)
	if idx>=value.shape[1]:
		return
	value[dest, idx] = value[src1, idx] - value[src2, idx] #
	for i in range(g.shape[2]):
		g[dest, idx, i] = g[src1, idx, i] #
		id[dest, idx, i] = id[src1, idx, i]
		if id[src1, idx, i]==-1:
			break
	for k in range(g.shape[2]):
		id2=id[src1, idx, k]
		if id2==-1:
			break
		i=-1
		g2=-g[src2,idx,k] #
		min=g2 
		
		for j in range(g.shape[2]):
			idd=id[dest, idx, j]
			if idd==-1:
				i=j
				if i+1<g.shape[2]:
					id[dest, idx, i+1] = -1
				break

				break
			if id2==idd:
				i=-1
				g[dest, idx, j] +=  g2
				break
			gd=g[dest, idx, j] 
			if abs(min)>abs(gd):
				min=gd
				i=idd
		if i!=-1:
			g[dest, idx, i] = g2
			id[dest, idx, i] = id2

@cuda.jit
def AutoFore_error2Delta(value,delta,g,id,dest):
	idx,idy=cuda.grid(2)
	if idx>=value.shape[1] or idy>=g.shape[2]:
		return
	idaux=id[dest, idx, idy]
	if idaux==-1:
		return
	delta[idaux, idx] += g[dest, idx, idy]
	#cuda.atomic.add(delta[idaux],idx, g[dest, idx, idy])


@cpu.jit
def AutoFore_CPU_error2Delta(value,delta,g,id,dest):
	idx,idy=cpu.grid(2)
	if idx>=value.shape[1] or idy>=g.shape[2]:
		return
	idaux=id[dest, idx, idy]
	if idaux==-1:
		return
	delta[idaux, idx] += g[dest, idx, idy]
	#cpu.atomic.add(delta[idaux],idx, g[dest, idx, idy])


@cuda.jit
def AutoFore_applyDelta(value,delta,g,id,dest,epsilon):
	idx,idy=cuda.grid(2)
	if idx>=value.shape[1] or idy>=g.shape[2]:
		return
	idaux=id[dest, idx, idy]
	if idaux==-1:
		return
	value[idaux,idx] -= delta[idaux,idx] * epsilon
	delta[idaux,idx] = 0

@cpu.jit
def AutoFore_CPU_applyDelta(value,delta,g,id,dest,epsilon):
	idx,idy=cpu.grid(2)
	if idx>=value.shape[1] or idy>=g.shape[2]:
		return
	idaux=id[dest, idx, idy]
	if idaux==-1:
		return
	value[idaux,idx] -= delta[idaux,idx] * epsilon
	delta[idaux,idx] = 0

@cuda.jit
def AutoFore_assign2(value,delta,g,id,id_var,v2):
	idx=cuda.grid(1)
	if idx>=value.shape[1]:
		return		
	value[id_var, idx] = v2[idx]
	for i in range(g.shape[2]):
		g[id_var, idx, i] = 0
		id[id_var, idx,i] = -1

@cpu.jit
def AutoFore_CPU_assign2(value,delta,g,id,id_var,v2):
	idx=cpu.grid(1)
	if idx>=value.shape[1]:
		return		
	value[id_var, idx] = v2[idx]
	for i in range(g.shape[2]):
		g[id_var, idx, i] = 0
		id[id_var, idx,i] = -1

@cuda.jit
def AutoFore_execute2(value,delta,g,id,ins,flo):
	for i,x in enumerate(ins):
		op = (x >> 48) & 0xFFFF
		dest = (x >> 32) & 0xFFFF
		src1 = (x >> 16) & 0xFFFF
		src2 = x & 0xFFFF

		if op==0:
			AutoFore_add(value,delta,g,id,dest,src1,src2)
		elif op==1:
			AutoFore_mul(value,delta,g,id,dest,src1,src2)	
		elif op==2:
			AutoFore_sub(value,delta,g,id,dest,src1,src2)
		elif op==3:
			AutoFore_assign(value,delta,g,id,dest,flo[i])
		#cuda.syncthreads()
	
@cpu.jit
def AutoFore_CPU_execute2(value,delta,g,id,ins,flo):
	for i,x in enumerate(ins):
		op = (x >> 48) & 0xFFFF
		dest = (x >> 32) & 0xFFFF
		src1 = (x >> 16) & 0xFFFF
		src2 = x & 0xFFFF

		if op==0:
			AutoFore_CPU_add(value,delta,g,id,dest,src1,src2)
		elif op==1:
			AutoFore_CPU_mul(value,delta,g,id,dest,src1,src2)	
		elif op==2:
			AutoFore_CPU_sub(value,delta,g,id,dest,src1,src2)
		elif op==3:
			AutoFore_CPU_assign(value,delta,g,id,dest,flo[i])
		#cpu.syncthreads()
	