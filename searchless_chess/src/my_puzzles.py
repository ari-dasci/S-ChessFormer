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
import logging
from searchless_chess.src.engines import constants
from searchless_chess.src.engines import engine as engine_lib

import engines.stockfish_engine as stock_eng
import json
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

_DEPTH = flags.DEFINE_integer(
    name='depth',
    default=None,
    help='The depth to use for the engine (only used by Stockfish).',
)

_TIME = flags.DEFINE_integer(
    name='time',
    default=None,
    help='The time the engine uses to analyse each move (only used by Stockfish).',
)


def analyse_puzzle_from_board_with_LST(
    board: chess.Board,
    engine: engine_lib.Engine,
    search: bool = False
) -> dict:
  """Returns the evaluation of all posible moves by LST (or a neural model), ordered by CP."""
  
  # Obtenemos las valoraciones del motor a evaluar (LST)
  if search:
    win_probs = engine.analyse(board)['probs']
  else:
    buckets_log_probs = engine.analyse(board)['log_probs']
    win_probs = np.inner(np.exp(buckets_log_probs), engine._return_buckets_values)
  
  sorted_legal_moves = engine_lib.get_ordered_legal_moves(board)
  
  # Guardamos los resultados en un diccionario
  dict_results = {
    move.uci(): {
      'wp': wp, 
      'cp': win_probability_to_centipawns(wp)
    } for move, wp in zip(sorted_legal_moves, win_probs)
  }
  
  return dict_results


def analyse_puzzle_from_board_with_Stockfish(
    board: chess.Board,
    engine: engine_lib.Engine
) -> dict:
  """Returns the evaluation of all posible moves by Stockfish, ordered by CP."""
  
  # Obtenemos las valoraciones de Stockfish
  
  analysis_results_1 = engine.analyse(board)
  logging.info(f"Analysis results: {analysis_results_1}")
  analysis_results_1 = analysis_results_1['scores']
  moves = [move.uci() for move, _ in analysis_results_1]
  moves_cp = [eval_.score(mate_score=3000) for _, eval_ in analysis_results_1] # Un mate se penaliza con 3000 CP
  
  dict_results = {
    move: {
      'wp': centipawns_to_win_probability(cp), 
      'cp': cp
    } for move, cp in zip(moves, moves_cp)
  }
  
  return dict_results

def get_engine(agent: str, limit: chess.engine.Limit = None) -> engine_lib.Engine:
    """Returns the engine instance based on the agent and limit."""
    if agent in ['stockfish', 'stockfish_all_moves']:
        return constants.ENGINE_BUILDERS[agent](limit=limit)
    else:
        return constants.ENGINE_BUILDERS[agent]()

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
  if _AGENT.value in ['stockfish', 'stockfish_all_moves']:
    if _DEPTH.value is not None:
      limit = chess.engine.Limit(depth=_DEPTH.value)
    elif _TIME.value is not None:
      limit = chess.engine.Limit(time=_TIME.value)
    else:
      limit = chess.engine.Limit(time=0.05)
    engine = get_engine(_AGENT.value, limit=limit)
  else:
    engine = get_engine(_AGENT.value)
  
  # Analizamos todos los puzzles
  results_list = []
  
  for i, puzzle in puzzles.iterrows():
    board = chess.Board(puzzle['FEN'])
    logging.info(f"Analysing puzzle {i+1}")
    # Predecimos las jugadas
    if _AGENT.value in ['stockfish', 'stockfish_all_moves']:
      results = analyse_puzzle_from_board_with_Stockfish(
          board=board,
          engine=engine
      )
    else:
      results = analyse_puzzle_from_board_with_LST(
          board=board,
          engine=engine,
          search='Depth' in _AGENT.value
      )
    logging.info(f"Results puzzle {i+1}: {results}")
    # Guardamos los resultados
    results_list.append(results)
  
  column_name = _AGENT.value + '_results'
  if _AGENT.value in ['stockfish', 'stockfish_all_moves']:
    if _DEPTH.value is not None:
      column_name += '_' + str(_DEPTH.value) + 'depth'
    elif _TIME.value is not None:
      column_name += '_' + str(_TIME.value) + 'time'
      
  final_results_df = pd.DataFrame({column_name: results_list})
  evaluated_puzzles = pd.concat([puzzles, final_results_df], axis=1)
  evaluated_puzzles.to_csv(output_file, index=False)

if __name__ == '__main__':
  app.run(main)
