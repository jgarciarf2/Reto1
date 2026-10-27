# Reto 1: Seguidor de línea con señales PARE / SIGA

Programa de visión artificial para un robot móvil que sigue una línea negra sobre la pista, corrige su trayectoria y obedece dos señales: una tarjeta roja (PARE) y una verde (SIGA). Las tarjetas pueden ser octágonos, que son las que vamos a usar con el profesor, o cuadrados girados como los de los videos de prueba.

La idea de la prueba es que el profesor pone la tarjeta PARE en cualquier momento mientras el robot recorre la pista. El robot se detiene y se queda quieto mientras la tarjeta siga ahí. Cuando la quitan, sigue. Si le muestran SIGA, también sigue. No usa redes neuronales ni modelos entrenados. Todo está hecho con operaciones vistas en el curso: recortes de ROI, espacios de color, umbralización, morfología, contornos, aproximación poligonal y K-Means básico.

## Cómo correrlo

```bash
pip install opencv-python numpy
python main.py                              # video por defecto (uploads/ideal/video1.mp4)
python main.py uploads/ideal/video3.mp4     # otro video
python main.py 0                            # cámara 0
```

El frame de la cámara se gira 90 grados en sentido horario para corregir el
stream del móvil cuando está en vertical. Si la imagen queda girada hacia el
lado contrario, cambia `ROTACION_CAMARA` en `main.py` a
`cv2.ROTATE_90_COUNTERCLOCKWISE`.

Se abren tres ventanas: el video procesado (con la máscara de color al lado) y dos ventanas de trackbars para calibrar en vivo. El significado de cada trackbar se imprime en consola al arrancar. Con `q` se cierra.

## Comunicación con el robot

El robot recibe una letra por cada acción. Las letras y el puerto están en **`config_robot.py`**, que es el único archivo que hay que tocar cuando nos den las definitivas:

```python
LETRAS = {
    "ADELANTE": "F",
    "IZQUIERDA": "I",
    "DERECHA": "D",
    "PARAR": "S",
}
PUERTO_SERIAL = None   # "COM3", "/dev/ttyUSB0"...
BAUDIOS = 9600
TERMINADOR = ""        # "\n" si el robot lee por líneas
```

Con `PUERTO_SERIAL = None` las letras solo se imprimen en consola, que es lo que sirve para probar con los videos. Al poner un puerto se envían por serial (hace falta `pip install pyserial`). La letra solo se manda cuando la acción cambia, para no saturar al robot con la misma letra en cada frame. Al cerrar el programa se envía PARAR.

El resto del código trabaja con los nombres de las acciones (`Accion.ADELANTE`, etc.) y nunca con las letras. Si el robot resulta usar otro protocolo, basta con escribir otra clase de salida con los métodos `escribir()` y `cerrar()` en `comunicacion_robot.py`.

## Cómo está organizado

| Archivo                 | Qué hace                                                                      |
| ----------------------- | ----------------------------------------------------------------------------- |
| `main.py`               | Abre el video o la cámara y conecta todas las piezas en el loop principal.    |
| `seguidor_linea.py`     | Encuentra la línea y propone ADELANTE / IZQUIERDA / DERECHA.                  |
| `preprocesador.py`      | Máscara de píxeles rojo o verde saturados para buscar las señales.            |
| `detector_senales.py`   | Filtra por forma los contornos de la máscara (octágonos y cuadrados girados). |
| `clasificador_color.py` | K-Means sobre el interior de la figura para decidir si es rojo o verde.       |
| `control_senales.py`    | Máquina de estados: SIGUIENDO / DETENIDO.                                     |
| `comunicacion_robot.py` | Traduce la acción a su letra y la envía.                                      |
| `config_robot.py`       | Letras y puerto del robot.                                                    |
| `interfaz_control.py`   | Ventanas y trackbars.                                                         |
| `visualizador.py`       | Arma el panel que se ve en pantalla.                                          |

## Seguidor de línea

La cámara ve el frente del robot en la parte de abajo de la imagen, así que la línea se lee en dos franjas horizontales por encima de él: una **cercana** (justo delante del robot) y una **lejana** (lo que viene después).

1. **ROI**: se recortan las dos bandas. Su posición y alto se dan en porcentaje del frame, así funciona igual con cualquier resolución.
2. **Segmentación HSV**: la línea es oscura y poco saturada. Se filtra por valor y saturación máximos, con lo que los carteles rojos y verdes (muy saturados) no se confunden con la línea aunque estén en sombra.
3. **Morfología**: un cierre rellena los huecos del trazo (la línea está pintada a mano y tiene rayas claras). Después viene una apertura con un kernel del grosor mínimo de la línea, que borra todo lo que sea más delgado que la pista, como cables o rayas.
4. **Contornos y centroide**: en cada banda se toma el contorno más grande que la cruce de arriba a abajo, y se calcula su centroide con momentos. La exigencia de cruzar la banda descarta la cinta negra horizontal que sostiene los carteles.
5. **Error respecto al centro del robot**: el punto objetivo es un promedio ponderado de los dos centroides. El lejano hace que el robot empiece a girar antes de llegar a la curva. El error es la distancia entre ese punto y el centro de la imagen.
6. **Continuidad**: la línea no salta de un lado al otro entre una banda y otra ni de un frame al siguiente. Si una detección rompe esa continuidad (una sombra o un pedazo de cinta), se descarta.
7. **Decisión**: si el error está dentro de la tolerancia, ADELANTE; si no, IZQUIERDA o DERECHA según el signo.
8. **Línea perdida**: se repite la última acción que se tomó con la línea a la vista. Si se salió por la derecha, el robot sigue girando a la derecha hasta reencontrarla, que es justo el caso de los videos `uploads/descarriado/`. Si nunca vio la línea, se queda quieto.
9. **Filtro temporal**: una acción nueva solo reemplaza a la actual si se repite 3 frames seguidos, para que un frame con ruido no haga zigzaguear al robot.

## Señales PARE y SIGA

1. **ROI**: se buscan solo en la parte de arriba del frame. Abajo está el robot, que tiene pilas verdes y piezas rojas y amarillas.
2. **Segmentación por color**: blur, conversión a HSV y umbral por matiz (el rojo en los dos extremos del rango H, el verde en el medio) con saturación mínima. Salen dos máscaras, una roja y una verde, que se limpian con apertura y cierre y se juntan con un OR.
3. **Forma**: un contorno se acepta si tiene el área mínima (o sea, el cartel está cerca), no toca el borde de la imagen ni el límite de la ROI, su relación de aspecto es cercana a 1, es compacto (circularidad = 4π·área / perímetro² del polígono ≥ 0.70: el cuadrado da 0.785 y el octágono 0.95) y su `approxPolyDP` corresponde a una de las formas válidas:
   - **octágono**: 8 vértices, aceptando de 7 a 9 porque a veces approxPolyDP junta o parte una esquina;
   - **cuadrado girado**: 4 vértices.

   Las formas están en `DetectorSenales.FORMAS_VALIDAS`, así que agregar otra es cuestión de una línea.

4. **Color con K-Means**: se agrupan en dos clusters los píxeles de adentro de la figura (el fondo y las letras blancas), y el matiz del cluster más grande decide PARE o SIGA.

### Máquina de estados

```
SIGUIENDO --(PARE visto 3 frames seguidos)---------> DETENIDO   (envía PARAR)
DETENIDO  --(la tarjeta roja no se ve en 10 frames)-> SIGUIENDO  (la quitaron)
DETENIDO  --(SIGA visto 3 frames seguidos)---------> SIGUIENDO
```

Frenar es más exigente que seguir frenado. Para entrar a DETENIDO hace falta un PARE con forma válida durante 3 frames (trackbar `Confirmar`), así un falso positivo suelto no frena al robot. Para quedarse detenido basta con que haya una mancha roja de buen tamaño, porque cuando el profesor sostiene la tarjeta su mano puede tapar una parte y la forma deja de verse completa. Cuando la mancha roja lleva 10 frames sin aparecer (trackbar `Liberar`), el robot sigue. Si en un frame aparecen varias señales, manda la más grande, que es la más cercana.

### Pruebas

- **Videos de `uploads/ideal/`** (cuadrados girados): el robot ve SIGA y sigue, llega a PARE y se detiene, sin falsos positivos por la cartulina roja que aparece a un lado de la pista en `video1`.
- **Octágonos**: como no hay videos con las tarjetas octagonales, se pegaron las tarjetas PARE y SIGA sobre los nueve videos, con posición, tamaño y rotación al azar, y con una "mano" tapando parte de la tarjeta. En todos el robot frena al tercer frame de aparecer PARE, se mantiene quieto aunque la mano tape la tarjeta y sigue 10 frames después de que la quitan. Con SIGA a la vista mientras avanza, no cambia nada.

## Limitaciones y posibles mejoras

- La segmentación depende de la luz. Con sombras fuertes o reflejos hay que recalibrar los umbrales (`Umbral Val`, `Sat Min`, `Val Min`). Una mejora sería normalizar el brillo de cada banda antes de umbralizar, o usar el canal L de CIELab.
- El control es de tipo encendido/apagado: izquierda, derecha o adelante, sin velocidades intermedias. Si el robot aceptara comandos con intensidad, el mismo error en píxeles serviría para un control proporcional y los giros serían más suaves.
- El centro del robot se asume en la mitad de la imagen. Si la cámara queda corrida, se corrige con el trackbar `Centro`.
- Cuando un cartel tapa la línea justo delante del robot, el seguidor depende de la banda lejana o de la memoria. En tramos largos tapados podría desviarse.
- Mientras está detenido, cualquier objeto rojo grande a la vista lo mantiene quieto (por la histéresis). Si en la pista hay otras cosas rojas, conviene subir `Area Min` o achicar la ROI.
- Si el profesor pone la tarjeta muy cerca de la cámara o pegada al borde de la imagen, la forma no se puede confirmar y el robot no frena hasta que la tarjeta quede entera a la vista.
- Las letras y el puerto del robot todavía son provisionales.

## Comparación con otros equipos

_Pendiente: completar después de la socialización en clase._
