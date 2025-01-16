# 9 Enero 2020
* No incluir GPU, son solo 8 páginas máximo a dos columnas (excluida bibliografía), a pesar de estar en uno de los epígrafes
* Optar por la modalidad Hibrido de congreso gecco 2025 (https://gecco-2025.sigevo.org/Tracks#GECH%20-%20General%20Evolutionary%20Computation%20and%20Hybrids).
* Reunión de seguimiento el jueves 16 de enero siguiente por la mañana.
* Pasar a trabajar a overleaf con la plantilla.
* Incluir estado del arte y abstract.
* [Programar la calibración por cámaras](roboticArm4.md)


# Ideas futuras
* Me queda por hacer la parte genética
* Evitar la colisión en la ruta: planificar la ruta, si hay una violación física, se identifican los elementos que la han producido y se halla otra ruta. La identificación puede ser como el aprendizaje por refuerzo. Identifica las últimas acciones por gradientes (media exponencial) y las evitas, generas cierta repursión. El efecto sería como soñar una pesadilla recurrente. 
* Las familias de GPU quedan invalidadas en cuestión del ciclo de vida de variables y su reutilización tras la últimas correcciones de las variables efectuadas en autoforenumpy. Tienen que copiar su forma de gestión.
