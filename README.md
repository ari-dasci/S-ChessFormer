# S-ChessFormer
Repositorio sobre ajedrez, transformers y razonamiento vs memorización. TFM de Jorge Remacho y Álvaro Rodríguez

## Ejecución

Para instalar las dependencias que lanzan el bot, utiliza el siguiente script. Este creará un entorno conda `lichess_bot` con las dependencias necesarias (instalando conda en caso de que no esté presente en el sistema) y creará las variables de entorno precisas para que todo funcione.

```
./install.sh
```

Previo lanzamiento del script asegúrate de dar permisos de ejecución.

```
chmod +x install.sh
```


Para lanzar el bot en lichess, simplemente debes ejecutar los siguientes comandos:

```
cd lichess-bot
```

Antes de lanzar el script principal debes asegurarte de:

- Estás en el entorno conda adecuado. Esto es, en `lichess_bot`. En caso contrario, ejecuta
```
conda activate lichess_bot
```

- Tienes la variable de entorno PYTHONPATH exportada. Si no es así (te saldrá como error que no encuentra el módulo `searchless_chess` al intentar desplegar el bot), ejecuta

```
export PYTHONPATH="$PWD/engines:$PYTHONPATH"
```

Finalmente, lanza el bot con el siguiente comando:

```
python3 lichess-bot.py --engine_name ENGINE_NAME
```
Donde ENGINE\_NAME puede ser `ThinkLess_9M`, `ThinkLess_136M` o `ThinkLess_270M`.

Alternativamente, si se quiere ejecutar el proceso como un batch en una cola SLURM, se puede emplear el script `run_bot.sh`.
