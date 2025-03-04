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

import metrics_TFM
import engines.stockfish_engine as stock_eng

import numpy as np


_INPUT_FILE = flags.DEFINE_string(
    name='input_file',
    default="../problemas/unsolved_puzzles/SBP_HARD.csv",
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
        'stockfish',
        'stockfish_all_moves',
        'leela_chess_zero_depth_1',
        'leela_chess_zero_policy_net',
        'leela_chess_zero_400_sims',
    ],
    help='The agent to evaluate.',
    required=False,
)


def evaluate_puzzle_from_board(
    index,
    board: chess.Board,
    move: str,
    engine: engine_lib.Engine,
    stockfish_depth
) -> bool:
  """Returns True if the `engine` makes the right move and False otherwise."""

  # Metrics accord Stockfish
  probs_St_accord_St = []
  probs_LST_accord_St = []
  centipawnss_St_accord_St = []
  centipawnss_LST_accord_St = []

  # Metrics accord LST
  probs_St_accord_LST = []
  probs_LST_accord_LST = []
  centipawnss_St_accord_LST = []
  centipawnss_LST_accord_LST = []

  # Initialize Stockfish for scores
  limit = chess.engine.Limit(depth=stockfish_depth)
  st_engine = stock_eng.AllMovesStockfishEngine(limit)

  metrics_TFM.createCSVFile()

  predicted_move = engine.play(board=board).uci()
  predicted_move_St = st_engine.play(board=board).uci()
  #print(f'Mov predicho por LST: {predicted_move}')
  #print(f'Mov predicho por Stockfish: {predicted_move_St}')
  pi_St_accord_St, pi_LST_accord_St, centipawns_St_accord_St, centipawns_LST_accord_St = metrics_TFM.computeMetricsAccordStockfish(predicted_move,predicted_move_St,board,st_engine,metrics_TFM.TAYLOR_FUNCTION)
  pi_St_accord_LST,pi_LST_accord_LST,centipawns_St_accord_LST,centipawns_LST_accord_LST = metrics_TFM.computeMetricsAccordLST(predicted_move,predicted_move_St,board,engine,metrics_TFM.TAYLOR_FUNCTION)

  probs_St_accord_St.append(pi_St_accord_St)
  probs_LST_accord_St.append(pi_LST_accord_St)
  centipawnss_St_accord_St.append(centipawns_St_accord_St)
  centipawnss_LST_accord_St.append(centipawns_LST_accord_St)

  probs_St_accord_LST.append(pi_St_accord_LST)
  probs_LST_accord_LST.append(pi_LST_accord_LST)
  centipawnss_St_accord_LST.append(centipawns_St_accord_LST)
  centipawnss_LST_accord_LST.append(centipawns_LST_accord_LST)

  metrics_TFM.writeWithinCSV(index,predicted_move_St,predicted_move,probs_St_accord_St,probs_LST_accord_St,centipawnss_St_accord_St,centipawnss_LST_accord_St,np.abs(np.array(probs_St_accord_St,dtype=float)-np.array(probs_LST_accord_St,dtype=float)),np.abs(np.array(centipawnss_St_accord_St,dtype=float)-np.array(centipawnss_LST_accord_St,dtype=float)),probs_St_accord_LST,probs_LST_accord_LST,centipawnss_St_accord_LST,centipawnss_LST_accord_LST,np.abs(np.array(probs_St_accord_LST,dtype=float)-np.array(probs_LST_accord_LST,dtype=float)),np.abs(np.array(centipawnss_St_accord_LST,dtype=float)-np.array(centipawnss_LST_accord_LST,dtype=float)))
  # Lichess puzzles consider all mate-in-1 moves as correct, so we need to
  # check if the `predicted_move` results in a checkmate if it differs from
  # the solution.
  if move != predicted_move:
    board.push(chess.Move.from_uci(predicted_move))
    return board.is_checkmate(), predicted_move

  # If we decide to solve puzzles with more than a movement, we should do it
  #probs_St_accord_St = probs_LST_accord_St = centipawnss_St_accord_St = centipawnss_LST_accord_St = []
  #probs_St_accord_LST = probs_LST_accord_LST = centipawnss_St_accord_LST = centipawnss_LST_accord_LST = []

  return True, predicted_move


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
  engine = constants.ENGINE_BUILDERS[_AGENT.value]()
  stockfish_depth=1
  # Añadimos una nueva columna "Played" y otra "Correct" al dataframe
  puzzles['Played'] = None
  puzzles['Correct'] = None

  # Analizamos todos los puzzles
  for index, puzzle in puzzles.iterrows():
    board = chess.Board(puzzle['FEN'])
    move = puzzle['Moves_UCI']
    # Predecimos la jugada
    correct, play = evaluate_puzzle_from_board(
        index,
        board=board,
        move=move,
        engine=engine,
        stockfish_depth=stockfish_depth
    )
    # Guardamos los resultados
    puzzles.at[index, 'Played'] = play
    puzzles.at[index, 'Correct'] = correct

    print({'puzzle_id': index, 'correct': correct, 'play': play, 'solution_UCI': puzzle['Moves_UCI']})

  puzzles.to_csv(output_file, index=False)


if __name__ == '__main__':
  app.run(main)