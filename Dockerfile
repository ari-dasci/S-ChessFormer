# Usa una imagen de Miniconda basada en Python 3.10
FROM continuumio/miniconda3:latest

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia los archivos de la aplicación al contenedor
COPY . /app/

# Crea un nuevo entorno Conda llamado "lichess_env" con Python 3.10
RUN conda create --name lichess_env python=3.10 --yes

# Establece el entorno activado como predeterminado
ENV PATH="/opt/conda/envs/lichess_env/bin:$PATH"

# Instala las dependencias dentro del entorno "lichess_env"
RUN conda run -n lichess_env pip install -r requirements.txt

# Establece PYTHONPATH
RUN export PYTHONPATH="/app/lichess-bot/engines:$PYTHONPATH"
ENV PYTHONPATH=/app/lichess-bot/engines:$PYTHONPATH

WORKDIR /app/lichess-bot

# Comando por defecto para ejecutar el bot
# Cambiar ThinkLess_9M por el motor a usar
CMD ["conda", "run", "-n", "lichess_env", "python", "lichess-bot.py", "--engine_name", "ThinkLess_9M"]	
