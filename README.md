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
python3 lichess-bot.py
```
