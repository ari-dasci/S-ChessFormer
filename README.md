# S-ChessFormer
Repositorio sobre el TFM ''Análisis e interpretabilidad de Transformers aplicados al ajedrez'', de Jorge López Remacho.

## Abstract

Tras la invención y el auge de la arquitectura *Transformer*, su aplicación se ha extendido a un amplio abanico de disciplinas. En 2024, *Google DeepMind* incorporó el ajedrez a esta lista con sus modelos *Searchless Chess*, basados en *Large Scale Transformers* y capaces de alcanzar fuerza de gran maestro en su versión más potente. Este avance abre la puerta al uso de modelos *Transformer* en problemas de planificación y razonamiento a medio y largo plazo.

Este trabajo analiza las capacidades y limitaciones de los modelos *Searchless Chess* a través de tres líneas experimentales. En primer lugar, se evalúan sobre un conjunto de problemas de ajedrez de alta dificultad, diferenciando entre problemas tácticos —que requieren cálculo de secuencias— y estratégicos —donde predominan patrones posicionales y objetivos a largo plazo—. En segundo lugar, se estudia su desempeño en partidas reales contra jugadores humanos y contra distintos niveles del motor Stockfish, mediante bots desplegados en Lichess. Este análisis incluye también partidas de ajedrez *Random-Fisher*, en las que la disposición inicial de las piezas de la primera y última fila se reordena aleatoriamente, lo que permite evaluar la capacidad de generalización de los modelos ante escenarios atípicos. Se analizan métricas como la precisión en partida, la pérdida media de probabilidad de victoria por movimiento y el estilo de juego. Finalmente, se realiza un análisis de interpretabilidad sobre el modelo más pequeño (9 millones de parámetros) mediante técnicas basadas en capas de atención (*Attention Rollout*, *Attention Flow* y diferencias de activaciones), visualización de *embeddings*, métodos basados en gradientes y técnicas agnósticas como SHAP y LIME, adaptadas al contexto del ajedrez.

Los resultados muestran que la principal limitación de estos modelos reside en su falta de profundidad táctica, obteniendo un rendimiento inferior en posiciones que requieren cálculo exhaustivo. En las partidas, se obtiene una caracterización clara del estilo de juego de los modelos, y se observa una degradación en su rendimiento al jugar partidas *Random-Fisher*, aunque en general mantienen un buen nivel de desempeño. Por el contrario, exhiben un comportamiento sólido en contextos estratégicos. Asimismo, se identifican patrones estructurales en las capas de atención que ofrecen pistas sobre los mecanismos internos del modelo, y se constata una notable robustez ante perturbaciones menores en el tablero mediante técnicas como LIME. Estos resultados aportan una caracterización precisa de los límites y fortalezas actuales de los modelos *Transformer* aplicados al ajedrez, y sientan las bases para futuras mejoras en su capacidad de razonamiento, generalización e interpretabilidad.

## Contenido del repositorio

El repositorio cuenta con la siguiente estructura:

- `analisis_partidas`: Contiene los archivos PGN de las partidas jugadas por los *bots* en Lichess, divididas por modelo, oponente (humano o bot) y variante (clásica o *chess960*).
- `lichess_bot`: Código relativo al despliegue de los *bots* de Lichess.
    - `homemade.py`: Código donde se implementan los motores para su uso por los bots.
    - `run_bot_{server}.sh`: *Script* para ejecutar el server de un *bot* de Lichess. Recibe como parámetro el nombre del modelo a ejecutar: `ThinkLess_9M`, `ThinkLess_136M` o `ThinkLess_270M`. Alternativamente, también se puede utilizar `python3 lichess-bot.py --engine_name ENGINE_NAME`, asegurándose de estar en un entorno con las dependencias instaladas.
- `memoria`: Memorias del TFM, en formato completo y con reducción de calidad.
- `notebooks`: *Notebooks* de Jupyter donde se desarrollan las diferentes partes del trabajo.
    - `analisis_partidas.ipynb`: Código asociado al análisis de partidas.
    - `analisis_puzzles.ipynb`: Código asociado al análisis de puzzles.
    - `xai.ipynb`: Código asociado al análisis interpretativo y explicativo del modelo.
- `problemas`: Archivos asociados a la gestión de problemas/puzzles de ajedrez. 
    - `puzzles_texto`: Archivos de puzzles en formato texto, los cuales necesitan preprocesamiento.
    - `scripts`: Conjunto de programas de utilidad para el preprocesamiento y manejo de archivos de puzzles.
        - `chess_problem_parser.py`: Convierte archivos de problemas en formato texto en formato CSV, manejable por el *script* `my_puzzles.py`.
        - `conversion_uci_pgn.py`: Muestra del código necesario para convertir movimientos de formato UCI a PGN y viceversa.
        - `copy_col_csv_within_other_csv.py`: Copia una columna de un archivo CSV y la pega al final de otro. Útil para lidiar con archivos de resultados.
    - `solved_puzzles`: Archivos de puzzles resueltos.
        - `all_engines`: Contiene los puzzles resueltos por todos los motores (9M, 136M, 270M y dos configuraciones de Stockfish) en formato CSV.
    - `unsolved_puzzles`: Conjuntos de puzzles base, sin resolver.
        - `CBP_HARD.csv`: Puzzles de táctica difíciles.
        - `SBP_HARD.csv`: Puzzles de estrategia difíciles.
        - `SBP_MEDIUM.csv`: Puzzles de estrategia media.
- `searchless_chess`: Código relativo a los motores de *Searchless Chess* 9M, 136M y 270M.
    - `checkpoints`: Pesos entrenados de los modelos.
    - `data`: Localización de los datos de entrenamiento (descargables con el *script* `data/download.sh`).
    - `src`: Código relativo a los *Transformer*. A continuación, mencionamos los *scripts* que hemos creado nosotros:
        - `my_puzzles.py`: Programa que, dado un conjunto de puzzles en formato CSV, evalúa cada puzzle con uno de los modelos y adjunta los resultados en una nueva columna, creando un nuevo CSV con los datos originales más la nueva predicción. Para ejecutarlo, debe seguirse la siguiente sintaxis: 
        ```
        python3 my_puzzles.py --agent <nombre_modelo> --input_file <input_csv> --output_file <output_csv> [--depth <profundidad (si se usa Stockfish)>] [--time <tiempo (si se usa Stockfish)>]
        ```
        Si se usa Stockfish, es necesario instalarlo previamente:
        ```
        cd searchless_chess
        git clone https://github.com/official-stockfish/Stockfish.git
        cd Stockfish/src
        make -j profile-build ARCH=x86-64-avx2
        cd ../../..
        ```
        - `old_my_puzzles.py`: Versión antigua del programa anterior, la cual ejecuta un modelo y Stockfish sobre cada puzzle y guarda las valoraciones de ambos modelos para dos jugadas: la jugada correcta según el CSV y la jugada del modelo elegido. 
        - `parallel_my_puzzles.py`: Inclusión en `my_puzzles.py` de un mecanismo de *dataloader* para realizar las predicciones de forma más rápida.
        - `transformer_xai.py`: Modificación del código del *transformer* para la implementación de ciertas técnicas de interpretabilidad, como la extracción de las capas de atención o el *embedding override*.
- `xai`: Resultados (principalmente imágenes) relativos a la parte de interpretabilidad.
    - `distancias_embeddings`: Archivos de distancias entre *embeddings*, guardados en disco para no tener que recalcularse.
    - `experiment1_batched`: Resultados de los experimentos con la posición inicial.
    - `experiment2_batched`: Resultados de los experimentos con la posición inicial de negras, tras `1. e4`.
    - `experiment3_batched`: Resultados de los experimentos con la posición *Carrera de peón*.
    - `experiment4_batched`: Resultados de los experimentos con la posición *Mates del pasillo*.
    - `experiment5_batched`: Resultados de los experimentos con la posición *Mates cruzados*.
    - `old_experiment1` y `old_experiment2`: Resultados antiguos de experimentos, donde se consideran las capas de atención crudas de tamaño 79x79.
- `install.sh`: Instala las dependencias necesarias para ejecutar el proyecto. Este instala `miniconda` y un entorno de `conda` llamado `lichess_bot`, donde se instalan todas las dependencias del proyecto.
- `README.md`: Información sobre el proyecto (es este archivo).
- `requirements.txt`: Librerías de *python* necesarias para ejecutar el proyecto.
- `run_script_{server}.sh`: *Script* utilizado para ejecutar una de las variantes del programa `my_puzzles.py`. Uso: `run_script_{server}.sh <script> <engine_name> <input_file> <output_file> [depth] [time]`

