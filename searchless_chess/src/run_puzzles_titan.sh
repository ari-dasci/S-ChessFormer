#!/bin/bash

#SBATCH --job-name=run_puzzles                 # Nombre del proceso
#SBATCH --partition=dios                    # Cola para ejecutar
#SBATCH --nodelist=titan                             # Servidor para ejecutar
#SBATCH --gres=gpu:1                         # Número de GPUs a usar

# Verifica que se haya pasado un argumento
if [ "$#" -ne 3 ]; then
    echo "Uso: $0 <engine_name> <input_file> <output_file>"
    exit 1
fi

ENGINE_NAME="$1"
INPUT_FILE="$2"
OUTPUT_FILE="$3"
cd "$(pwd)"

# Inicializa Conda para bash
source /opt/anaconda/etc/profile.d/conda.sh

# Verifica si el entorno 'lichess_bot' ya existe
if ! conda env list | grep -q "lichess_bot"; then
    echo "Creando entorno Conda 'lichess_bot'..."
    conda create --name lichess_bot python=3.10 --yes
        
    # Activa el entorno Conda
    conda activate lichess_bot

    # Instala pip si es necesario
    conda install pip -y
    
    # Instala las dependencias desde el archivo requirements.txt
    pip install -r /mnt/homeGPU/jorgelerre/S-ChessFormer/requirements.txt

    # Usa la GPU
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

else
    echo "El entorno Conda 'lichess_bot' ya existe."
fi

conda activate lichess_bot
# Instala pip si es necesario
conda install pip -y

# Instala las dependencias desde el archivo requirements.txt
pip install -r /mnt/homeGPU/jorgelerre/S-ChessFormer/requirements.txt

# Usa la GPU
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Exporta la variable de entorno PYTHONPATH
export PYTHONPATH="/mnt/homeGPU/jorgelerre/S-ChessFormer/:$PYTHONPATH"
export PYTHONPATH="/mnt/homeGPU/jorgelerre/S-ChessFormer/lichess_bot:$PYTHONPATH"

# Ejecuta el script principal con el nombre del motor pasado como argumento
python3 my_puzzles.py --agent "$ENGINE_NAME" --input_file "$INPUT_FILE" --output_file "$OUTPUT_FILE" 

# Envía un correo cuando termine
MAIL -s "Proceso lichess-bot.py finalizado" CORREO@gmail.com <<< "El proceso ha finalizado"
