#!/bin/bash
cd /mnt/homeGPU/jorgelerre/lichess-bot

# Verificar si conda ya está instalado
if ! command -v conda &> /dev/null
then
    echo "Conda no está instalado. Iniciando la instalación..."
    
    # Descargar el instalador de Miniconda (ajusta la URL según la arquitectura de tu sistema)
    MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
    wget https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER
    
    # Instalar Miniconda
    bash $MINICONDA_INSTALLER -b -p $HOME/miniconda3
    
    # Eliminar el instalador después de la instalación
    rm $MINICONDA_INSTALLER
    
    # Inicializar Conda
    source $HOME/miniconda3/bin/activate
    conda init bash
    
    echo "Conda ha sido instalado y configurado."
else
    echo "Conda ya está instalado."
fi

# Inicializa Conda para bash
source /opt/anaconda/etc/profile.d/conda.sh

# Verifica si el entorno 'lichess_bot' ya existe
if ! conda env list | grep -q "lichess_bot"; then
    # Si el entorno no existe, lo crea
    echo "Creando entorno Conda 'lichess_bot'..."
    conda create --name lichess_bot python=3.10 --yes
else
    echo "El entorno Conda 'lichess_bot' ya existe."
fi

# Activa el entorno Conda
conda activate lichess_bot

# Instala pip si es necesario
conda install pip

# Instala las dependencias desde el archivo requirements.txt
pip install -r requirements.txt

# Exporta la variable de entorno PYTHONPATH
# Cambiar /mnt/homeGPU/jorgelerre por la ubicacion en tu sistema
export PYTHONPATH="$PWD/engines:$PYTHONPATH"

