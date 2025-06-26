#!/bin/bash

#SBATCH --job-name=ChessBot                 # Nombre del proceso
#SBATCH --partition=dios                    # Cola para ejecutar
#SBATCH --nodelist=titan                    # Servidor para ejecutar
#SBATCH --gres=gpu:1                       # Número de GPUs a usar

# Verifica que se haya pasado un argumento
if [ "$#" -ne 1 ]; then
    echo "Uso: $0 <engine_name>"
    exit 1
fi

ENGINE_NAME="$1"

cd "$(pwd)"

# Inicializa Conda para bash
source /opt/anaconda/etc/profile.d/conda.sh

conda activate /mnt/homeGPU/jorgelerre/conda_envs/lichess_bot

# Exporta la variable de entorno PYTHONPATH
export PYTHONPATH="$PWD/..:$PYTHONPATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Ejecuta el script principal con el nombre del motor pasado como argumento
python3 lichess-bot.py --engine_name "$ENGINE_NAME"

# Envía un correo cuando termine
MAIL -s "Proceso lichess-bot.py finalizado" CORREO@gmail.com <<< "El proceso ha finalizado"
