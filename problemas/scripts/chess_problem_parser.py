"""
Este script permite analizar un archivo de texto que contiene problemas de ajedrez en formato PNG (notación descriptiva española) y convertirlo a formato CSV con notación UCI (Universal Chess Interface) para facilitar su procesamiento y análisis.

El script realiza las siguientes operaciones:
1. Extrae las posiciones FEN (Forsyth-Edwards Notation) de un archivo de texto, junto con las jugadas correspondientes.
2. Traduce las jugadas en notación descriptiva española (Ej: "Rey", "Dama", "Torre", "Caballo") a notación estándar en inglés (Ej: "K", "Q", "R", "N").
3. Convierte las jugadas de formato PNG (notación descriptiva) a formato UCI, que es ampliamente utilizado en motores de ajedrez.
4. Elimina entradas duplicadas basadas en la notación FEN.
5. Guarda los resultados en un archivo CSV con las posiciones FEN y las jugadas convertidas.

Uso:
    python chess_problem_parser.py [input_file.txt] [output_file.csv]

Parámetros:
    input_file (opcional): Ruta al archivo de texto de entrada que contiene los problemas de ajedrez.
    output_file (opcional): Ruta al archivo CSV de salida para guardar los resultados.

Si no se proporcionan parámetros, se usan los archivos predeterminados "SBP_HARD_FINAL.txt" y "SBP_HARD_FINAL.csv".

Requiere las siguientes librerías:
    - chess: para trabajar con notaciones de ajedrez y movimientos.
    - pandas: para manejar y almacenar los datos en un DataFrame.
    - re: para realizar búsquedas de patrones en el texto.
    - sys y os: para manejar los parámetros de entrada y salida.

"""

import chess
import re
import pandas as pd
import sys
import os

# Diccionario de traducción (Español → Inglés)
translation_dict = {
    "R": "K",  # Rey (King)
    "D": "Q",  # Dama (Queen)
    "T": "R",  # Torre (Rook)
    "A": "B",  # Alfil (Bishop)
    "C": "N"   # Caballo (Knight)
}

# Función para traducir jugadas en el DataFrame
def translate_move(move):
    if pd.isna(move):  # Evita errores con valores NaN
        return move
    for esp, eng in translation_dict.items():
        move = move.replace(esp, eng)
    return move

# Funcion para traducir jugadas de PNG a UCI
def convert_PNG_to_UCI(board_fen, move_png):
	# Convertir a objeto 'Board'
	board = chess.Board(board_fen)
	# Convertir a objeto `Move`
	move = board.parse_san(move_png)
	# Convertir a UCI
	move_uci = move.uci()
	
	return move_uci


def parse_chess_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Expresión regular para encontrar posiciones FEN (líneas que cumplen el formato FEN)
    fen_pattern = r"^([\d\w\/]+ [wb] [KQkq\-]+ [\-\w\d]+ \d+ \d+)$"
    
    # Expresión regular para encontrar la primera jugada con "!"
    move_pattern = r"(\d+\.\s*[\w\-+#=]+\s*!|\.\.\.[\w\-+#=]+\s*!)"
    
    # Dividir en líneas y analizar
    lines = content.split("\n")
    fen_list = []
    move_list = []
    
    for i, line in enumerate(lines):
        match_fen = re.match(fen_pattern, line)
        if match_fen:
            fen = match_fen.group(1)
            
            # Buscar la mejor jugada en las siguientes líneas
            moves_text = " ".join(lines[i+1:])  # Tomamos las líneas siguientes para buscar la jugada
            match_move = re.search(move_pattern, moves_text)
            
            move = match_move.group(1) if match_move else None  # Extraer jugada si existe
            
            # Limpiar jugada (remover número de jugada como "24." si es necesario)
            if move:
                move = re.sub(r"\d+\.\s*", "", move)
                move = re.sub(r'\.+', "", move)
                move = re.sub(r"!", "", move).strip()
            
            fen_list.append(fen)
            move_list.append(move)
    
    # Crear DataFrame de Pandas
    df = pd.DataFrame({"FEN": fen_list, "Moves": move_list})
    return df

# Main
if __name__ == '__main__':
	# Lectura de parametros
	n = len(sys.argv)
	# Si no se pasan parámetros, se usan los archivos "por defecto"
	if n == 1:
		input_file = "SBP_HARD_FINAL.txt"
		output_file = "SBP_HARD_FINAL.csv"
		print("Usando ficheros de entrada/salida por defecto.")
		print("Si quieres usar otros ficheros, usa python chess_problem_parser.py [input_file.txt] [output_file.csv]")
	# Si se pasan parametros al programa, los usamos
	elif n == 2 or n == 3:
		input_file = sys.argv[1]
		if n == 3:
			output_file = sys.argv[2]
		else:
			output_file = os.path.splitext(input_file)[0] + ".csv"
	else:
		print("Modo de uso: python chess_problem_parser.py [input_file.txt] [output_file.csv]")
		
	print("Input file:", input_file)
	print("Output file:", output_file)
		
	# Extraccion de problemas del fichero de entrada
	df = parse_chess_file(input_file) 
	
	# Eliminar tableros repetidos (pasa, debido a la repeticion 
	# de FENs de manera consecutiva en el .txt)
	df = df.drop_duplicates(subset=["FEN"])  
	
	# Traduccion del español al ingles
	df["Moves"] = df["Moves"].apply(translate_move) 
	
	# Conversion de PNG a UCI
	df["Moves_UCI"] = df.apply(lambda row: convert_PNG_to_UCI(row["FEN"], row["Moves"]), axis = 1)
	
	# Guardamos el dataframe en formato CSV
	df.to_csv(output_file, index=False)

	#print(df)



