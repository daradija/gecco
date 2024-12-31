from drnumba import *




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
def AutoFore_add(value,delta,g,id,r1,r2,r3):
	idx=cuda.grid(1)
	if idx>=value.shape[1]:
		return

@cpu.jit
def AutoFore_CPU_add(value,delta,g,id,r1,r2,r3):
	idx=cpu.grid(1)
	if idx>=value.shape[1]:
		return
