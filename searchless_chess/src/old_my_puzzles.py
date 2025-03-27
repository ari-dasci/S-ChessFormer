# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluates engines on the puzzles dataset from lichess."""

from collections.abc import Sequence
import io
import os
import sys

from absl import app
from absl import flags
import chess
import chess.engine
import chess.pgn
import pandas as pd

from searchless_chess.src.engines import constants
from searchless_chess.src.engines import engine as engine_lib

import engines.stockfish_engine as stock_eng

import numpy as np
from utils import centipawns_to_win_probability, win_probability_to_centipawns


_INPUT_FILE = flags.DEFINE_string(
    name='input_file',
    default="../../problemas/unsolved_puzzles/SBP_HARD.csv",
    help='The input file name containing the puzzles to solve, in .csv.',
    required=False,
)

_OUTPUT_FILE = flags.DEFINE_string(
    name='output_file',
    default=None,
    help='The output file name where the solutions will be stored.',
    required=False,
)

_NUM_PUZZLES = flags.DEFINE_integer(
    name='num_puzzles',
    default=None,
    help='The number of puzzles to evaluate. If None, it processes all the puzzles in the file.',
    required=False,
)

_AGENT = flags.DEFINE_enum(
    name='agent',
    default='9M',
    enum_values=[
        'local',
        '9M',
        '136M',
        '270M',
        '9M_Depth',
        '136M_Depth',
        '270M_Depth',
        'stockfish',
        'stockfish_all_moves',
        'leela_chess_zero_depth_1',
        'leela_chess_zero_policy_net',
        'leela_chess_zero_400_sims',
    ],
    help='The agent to evaluate.',
    required=False,
)

_ORACLE = flags.DEFINE_enum(
    name='oracle',
    default='stockfish_all_moves',
    enum_values=[
        'local',
        '9M',
        '136M',
        '270M',
        'stockfish',
        'stockfish_all_moves',
        'leela_chess_zero_depth_1',
        'leela_chess_zero_policy_net',
        'leela_chess_zero_400_sims',
    ],
    help='The oracle to use to grant the solution and give metrics.',
    required=False,
)


def analyse_puzzle_from_board_with_LST(
    board: chess.Board,
    engine: engine_lib.Engine,
    search: bool = False
) -> pd.DataFrame:
  """Returns the evaluation of all posible moves by LST (or a neural model), ordered by CP."""
  
  # Obtenemos la jugada del motor a evaluar (LST). ESTO PARA MODELOS SIN PROFUNDIDAD
  if not search:
    buckets_log_probs = engine.analyse(board)['log_probs']
    win_probs = np.inner(np.exp(buckets_log_probs), engine._return_buckets_values)
  else:
    win_probs = engine.analyse(board)['probs']
  sorted_legal_moves = engine_lib.get_ordered_legal_moves(board)
  
  """
  print('Jugadas:', sorted_legal_moves)
  print('Probabilidades:', win_probs)
  
  for i in np.argsort(win_probs)[::-1]:
    print(i)
    cp = win_probability_to_centipawns(win_probs[i])
    print(f'  {sorted_legal_moves[i].uci()} -> {100*win_probs[i]:.1f}% cp: {cp}')
  """
  
  df_results = pd.DataFrame({
    'Jugada LST': [move.uci() for move in sorted_legal_moves],
    '%win LST': win_probs,
    'CP LST': [win_probability_to_centipawns(p) for p in win_probs]
  })
  
  df_results = df_results.sort_values(by='%win LST', ascending=False).reset_index(drop=True)
  
  return df_results

def analyse_puzzle_from_board_with_Stockfish(
    board: chess.Board,
    engine: engine_lib.Engine
) -> pd.DataFrame:
  """Returns the evaluation of all posible moves by Stockfish, ordered by CP."""
  
  # Obtenemos las valoraciones de Stockfish
  analysis_results_1 = engine.analyse(board)['scores']
  moves = [move.uci() for move, _ in analysis_results_1]
  moves_cp = [eval_.score(mate_score=3000) for _, eval_ in analysis_results_1] # Un mate se penaliza con 3000 CP
  
  results_df = pd.DataFrame({
    'Jugada Stockfish': moves,
    '%win Stockfish': [centipawns_to_win_probability(cp) for cp in moves_cp],  
    'CP Stockfish': moves_cp
  })
  
  # Ordenamos las jugadas de mejor a peor
  results_df = results_df.sort_values(by='CP Stockfish', ascending=False).reset_index(drop=True)
  
  return results_df

def compute_puzzle_metrics(
  results_lst: pd.DataFrame,
  results_sf: pd.DataFrame,
  correct_move: str
) -> pd.DataFrame:
  """Returns the metrics of the puzzle."""
  
  # Obtenemos la mejor jugada de cada modelo
  best_move_lst = results_lst.iloc[0]['Jugada LST']
  best_move_sf = results_sf.iloc[0]['Jugada Stockfish']
  print(correct_move, best_move_lst, best_move_sf)
  # Devolvemos ambas en un dataframe
  results_df = pd.DataFrame({
    # Métricas para el mejor movimiento
    '%win_LST_best_move': [results_lst[results_lst['Jugada LST'] == correct_move]['%win LST'].values[0]],
    '%win_Stockfish_best_move': [results_sf[results_sf['Jugada Stockfish'] == correct_move]['%win Stockfish'].values[0]],
    'CP_LST_best_move': [results_lst[results_lst['Jugada LST'] == correct_move]['CP LST'].values[0]],
    'CP_Stockfish_best_move': [results_sf[results_sf['Jugada Stockfish'] == correct_move]['CP Stockfish'].values[0]],
    # Métricas para el movimiento de LST 
    'move_LST': [best_move_lst], 
    'correct_LST': [best_move_lst == correct_move],
    '%win_LST_move_LST': [results_lst[results_lst['Jugada LST'] == best_move_lst]['%win LST'].values[0]],
    '%win_Stockfish_move_LST': [results_sf[results_sf['Jugada Stockfish'] == best_move_lst]['%win Stockfish'].values[0]],
    'CP_LST_move_LST': [results_lst[results_lst['Jugada LST'] == best_move_lst]['CP LST'].values[0]],
    'CP_Stockfish_move_LST': [results_sf[results_sf['Jugada Stockfish'] == best_move_lst]['CP Stockfish'].values[0]],
    # Métricas para el movimiento de Stockfish
    'move_Stockfish': [best_move_sf],
    'correct_Stockfish': [best_move_sf == correct_move],
    '%win_LST_move_Stockfish': [results_lst[results_lst['Jugada LST'] == best_move_sf]['%win LST'].values[0]],
    '%win_Stockfish_move_Stockfish': [results_sf[results_sf['Jugada Stockfish'] == best_move_sf]['%win Stockfish'].values[0]],
    'CP_LST_move_Stockfish': [results_lst[results_lst['Jugada LST'] == best_move_sf]['CP LST'].values[0]],
    'CP_Stockfish_move_Stockfish': [results_sf[results_sf['Jugada Stockfish'] == best_move_sf]['CP Stockfish'].values[0]]
  })
  print("Puzzle evaluado correctamente")
  return results_df



def main(argv: Sequence[str]) -> None:
  
  # Lectura de parametros
  n = len(sys.argv)
  puzzles_file = _INPUT_FILE.value
  if _OUTPUT_FILE.value == None:
    output_file = os.path.splitext(puzzles_file)[0] + "_solved.csv"
  else:
    output_file = _OUTPUT_FILE.value

  # Leemos los puzzles
  if(_NUM_PUZZLES.value == None):
    puzzles = pd.read_csv(puzzles_file)
  else:
    puzzles = pd.read_csv(puzzles_file, nrows=_NUM_PUZZLES.value)
    
  # Obtenemos el motor de ajedrez a analizar
  engine_lst = constants.ENGINE_BUILDERS[_AGENT.value]()
  
  # Obtenemos el motor de ajedrez a usar como oraculo
  # Por defecto, se usa stockfish con time_limit = 0.05
  #engine_oracle = constants.ENGINE_BUILDERS[_ORACLE.value]()
  
  # Opcion alternativa: limitar por profundidad
  #stockfish_depth = 10
  #limit = chess.engine.Limit(depth=stockfish_depth)
  limit=chess.engine.Limit(time=1.5)
  engine_oracle = stock_eng.AllMovesStockfishEngine(limit)
  
  # Añadimos una nueva columna "Played" y otra "Correct" al dataframe
  results_list = []
  
  # Analizamos todos los puzzles
  for _, puzzle in puzzles.iterrows():
    board = chess.Board(puzzle['FEN'])
    move = puzzle['Moves_UCI']
    # Predecimos la jugada con LST
    results_LST = analyse_puzzle_from_board_with_LST(
        board=board,
        engine=engine_lst,
        search='Depth' in _AGENT.value
    )
    # Predecimos la jugada con Stockfish
    results_stockfish = analyse_puzzle_from_board_with_Stockfish(
        board=board,
        engine=engine_oracle
    )
    metrics_puzzle = compute_puzzle_metrics(
        results_lst=results_LST,
        results_sf=results_stockfish,
        correct_move=move
    )
    
    print(metrics_puzzle)
    
    # Guardamos los resultados
    results_list.append(metrics_puzzle)

  final_results_df = pd.concat(results_list, ignore_index=True)
  evaluated_puzzles = pd.concat([puzzles, final_results_df], axis=1)
  #evaluated_puzzles.to_csv(output_file, index=False)
  evaluated_puzzles.to_csv(output_file, index=False)

if __name__ == '__main__':
  app.run(main)
