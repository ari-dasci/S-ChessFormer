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
import logging
import cProfile
import pstats

# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)

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
    default='AlgorithmProposed',
    enum_values=[
        'AlgorithmProposed',
       # 'AlgorithmProposedIterative'
    ],
    help='The agent to evaluate.',
    required=False,
)

_DEPTH = flags.DEFINE_integer(
    name='depth',
    default=None,
    help='The depth to use for the engine (only used by Stockfish).',
)


def analyse_puzzle_from_board_with_LST(
    board: chess.Board,
    engine: engine_lib.Engine
) -> dict:
  """Returns the evaluation of all posible moves by LST (or a neural model), ordered by CP."""
  
  # Obtenemos las valoraciones del motor a evaluar (LST)
  aux = engine.analyse(board)
  win_probs = aux['probs']
  sorted_legal_moves = aux['top_move']
  
  # Guardamos los resultados en un diccionario
  dict_results = {
    move.uci(): {
      'wp': wp, 
      'cp': win_probability_to_centipawns(wp)
    } for move, wp in zip(sorted_legal_moves, win_probs)
  }
  
  return dict_results

# Si se usa un método de búsqueda iterativa, pasamos el parámetro correspondiente y el resto como None
def get_engine(agent: str, depth: int, reeval_level: int, iter_search_method: str = None, N_best = None, percentage = None, percentage_epsilon = None) -> engine_lib.Engine:
    """Returns the engine instance based on the agent and limit."""

    if iter_search_method is not None and _AGENT.value == 'AlgorithmProposed':   
      return constants.get_algorithm_proposed(depth=depth,reeval_level=reeval_level,N_preevals=10, iter_search_method=iter_search_method, N_best=N_best, percentage=percentage, percentage_epsilon=percentage_epsilon)
    elif _AGENT.value == 'AlgorithmProposed':
      return constants.get_algorithm_proposed(depth=depth,reeval_level=reeval_level,N_preevals=10)

def main(argv: Sequence[str]) -> None:
  
  # Lectura de parametros
  n = len(sys.argv)
  puzzles_file = _INPUT_FILE.value
  if _OUTPUT_FILE.value == None:
    output_file = os.path.splitext(puzzles_file)[0] + "_solved.csv"
  else:
    output_file = _OUTPUT_FILE.value
  logging.info(f"Input file: {puzzles_file}")
  logging.info(f"Output file: {output_file}")
  logging.info(f"Agent: {_AGENT.value}")
  logging.info(f"Number of puzzles: {_NUM_PUZZLES.value}")
  # Leemos los puzzles
  if(_NUM_PUZZLES.value == None):
    puzzles = pd.read_csv(puzzles_file)
  else:
    puzzles = pd.read_csv(puzzles_file, nrows=_NUM_PUZZLES.value)
  
  if _AGENT.value == 'AlgorithmProposed':
    depths = [2, 3, 4]
    all_results = []

    evaluated_puzzles = puzzles.copy()

    for depth in depths:
        engine = get_engine(_AGENT.value, depth=depth, reeval_level=depth-1)
        results_list = []

        for i, puzzle in puzzles.iterrows():
            board = chess.Board(puzzle['FEN'])
            logging.info("------------------------------------------------------------------------------------------------------------------------------")

            results = analyse_puzzle_from_board_with_LST(
                board=board,
                engine=engine
            )

            logging.info(f"Results puzzle {i+1}: {results}")
            logging.info("------------------------------------------------------------------------------------------------------------------------------")
            results_list.append(results)

        column_name = _AGENT.value + '_results_depth_' + str(depth)
        evaluated_puzzles[column_name] = results_list
    evaluated_puzzles.to_csv(output_file, index=False)

  elif _AGENT.value == 'AlgorithmProposedIterative':
    depths = [2, 3, 4, 5]
    iterative_methods = {'N_best': [5,6,7,8,9,10], 'percentage':{0.1,0.2,0.3,0.4,0.5}, 'percentage_epsilon':{0.1,0.2,0.3}}
    all_results = []

    evaluated_puzzles = puzzles.copy()

    for depth in depths:
        for method in iterative_methods:
           for value in iterative_methods[method]:
              logging.info(f"Selected model: depth {depth} with method {method} and value {value}")
              if method == 'N_best':
                engine = get_engine(_AGENT.value, depth=depth, reeval_level=depth-1, iter_search_method=method, N_best=value, percentage=None, percentage_epsilon=None)
                column_name = _AGENT.value + '_results_depth_' + str(depth) + '_N_best_' + str(value)
              elif method == 'percentage':
                engine = get_engine(_AGENT.value, depth=depth, reeval_level=depth-1, iter_search_method=method, N_best=None, percentage=value, percentage_epsilon=None)
                column_name = _AGENT.value + '_results_depth_' + str(depth) + '_percentage_' + str(value)
              elif method == 'percentage_epsilon':
                engine = get_engine(_AGENT.value, depth=depth, reeval_level=depth-1, iter_search_method=method, N_best=None, percentage=None, percentage_epsilon=value)
                column_name = _AGENT.value + '_results_depth_' + str(depth) + '_percentage_epsilon_' + str(value)
        results_list = []

        for i, puzzle in puzzles.iterrows():
            board = chess.Board(puzzle['FEN'])
            logging.info("------------------------------------------------------------------------------------------------------------------------------")

            results = analyse_puzzle_from_board_with_LST(
                board=board,
                engine=engine
            )

            logging.info(f"Results puzzle {i+1}: {results}")
            logging.info("------------------------------------------------------------------------------------------------------------------------------")
            results_list.append(results)

        #column_name = _AGENT.value + '_results_depth_' + str(depth)
        evaluated_puzzles[column_name] = results_list
    evaluated_puzzles.to_csv(output_file, index=False)

  
  

if __name__ == '__main__':
  app.run(main)
