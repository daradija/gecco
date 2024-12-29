# ¿Cómo usar el Doctor Numba?

DrNumba facilita la programación en gpu usando numba.

¿Por qué incluir drnumba y no usar numba simplemente?

1. La programación con numba cuando se complica el kernel tiende a muchos parámetros, los cuales tienen muchas dimensiones, lo que conduce a un código ininteligible.
2. Numba no está pensado para depurar en gpu. 

# Manual de usuario.

'''python
# incluye todo ya que tiene funciones de hack
from drnumba import *

# instancia una variable a nivel global
drnumba=DrNumba("kernel.py")




'''