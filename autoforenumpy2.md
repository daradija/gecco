# 19/02/2025
* Idea: Usar la genética con diferente tasa de aprendizaje por cada parámetro.
* Se elimina error2delta. Ahora directamente desde el error se produce el aprendizaje.

```python
    suberror=nn.val(0)
    for c in arm:
        for eye in eyes:
            errorAux=eye.error(c)
            error2=errorAux*errorAux
            suberror+=error2
    suberror.learn()
````
Ahora desde el punto de vista del usuario la idea es muy sencilla. Basta acumular los errores cuadraticos del lote que se decida y llamar a learn.

Nótese que no hay tasa de aprendizaje. Es algo que decide el propio sistema con su población genética.
Ahora sí, dotamos a la problación de sentido y Gecco tiene sentido. Se espera que provoque una diversidad genética.