#!/bin/bash

#SBATCH --job-name=ChessBot                 # Nombre del proceso
#SBATCH --partition=dios                    # Cola para ejecutar
#SBATCH --gres=gpu:1                         # Número de GPUs a usar

# Verifica que se haya pasado un argumento
if [ "$#" -ne 1 ]; then
    echo "Uso: $0 <engine_name>"
    exit 1
fi

ENGINE_NAME="$1"

cd /mnt/homeGPU/jorgelerre/lichess-bot

# Inicializa Conda para bash
source /opt/anaconda/etc/profile.d/conda.sh

# Verifica si el entorno 'lichess_bot' ya existe
if ! conda env list | grep -q "lichess_bot"; then
    echo "Creando entorno Conda 'lichess_bot'..."
    conda create --name lichess_bot python=3.10 --yes
else
    echo "El entorno Conda 'lichess_bot' ya existe."
fi

# Activa el entorno Conda
conda activate lichess_bot

# Instala pip si es necesario
conda install pip -y

# Instala las dependencias desde el archivo requirements.txt
pip install -r requirements.txt

# Exporta la variable de entorno PYTHONPATH
export PYTHONPATH="$PWD/engines:$PYTHONPATH"

# Ejecuta el script principal con el nombre del motor pasado como argumento
python3 lichess-bot.py --engine_name "$ENGINE_NAME"

# Envía un correo cuando termine
MAIL -s "Proceso lichess-bot.py finalizado" CORREO@gmail.com <<< "El proceso ha finalizado"

