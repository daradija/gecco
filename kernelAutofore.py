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
	value[dest, idx] = value[src1, idx] + value[src2, idx]
	for i in range(g.shape[2]):
		g[dest, idx, i] = g[src1, idx, i] 
		id[dest, idx, i] = id[src1, idx, i]
		if id[src2, idx, i]!=-1:
			break
	for i in range(g.shape[2]):
		i=-1
		min=0
		id1=id[src1, idx, i]
		for j in range(g.shape[2]):
			id2=id[src2, idx, j]
			if id2==-1:
				break
			if id2==id1:
				i=-1
				g[dest, idx, i] +=  g[src2, idx, i]
				break
			g2=np.abs(g[src2, idx, i])
			if min<g2:
				min=g2
				i=id2
		if i!=-1:
			g[dest, idx, i] = min
			id[dest, idx, i] = i

@cpu.jit
def AutoFore_CPU_add(value,delta,g,id,dest,src1,src2):
	idx=cpu.grid(1)
	if idx>=value.shape[1]:
		return
	value[dest, idx] = value[src1, idx] + value[src2, idx]
	for i in range(g.shape[2]):
		g[dest, idx, i] = g[src1, idx, i] 
		id[dest, idx, i] = id[src1, idx, i]
		if id[src2, idx, i]!=-1:
			break
	for i in range(g.shape[2]):
		i=-1
		min=0
		id1=id[src1, idx, i]
		for j in range(g.shape[2]):
			id2=id[src2, idx, j]
			if id2==-1:
				break
			if id2==id1:
				i=-1
				g[dest, idx, i] +=  g[src2, idx, i]
				break
			g2=np.abs(g[src2, idx, i])
			if min<g2:
				min=g2
				i=id2
		if i!=-1:
			g[dest, idx, i] = min
			id[dest, idx, i] = i
