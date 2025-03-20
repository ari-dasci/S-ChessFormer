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
import numpy as np
from utils import centipawns_to_win_probability, win_probability_to_centipawns

from searchless_chess.src.engines import constants
from searchless_chess.src import constants as constants_src
from searchless_chess.src.engines import engine as engine_lib
from searchless_chess.src.engines import neural_engines
from searchless_chess.src import tokenizer
from searchless_chess.src import training_utils
from searchless_chess.src import transformer
from searchless_chess.src import utils
from searchless_chess.src import data_loader
from searchless_chess.src import config as config_lib
import engines.stockfish_engine as stock_eng


from jax import random as jrandom
import json


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
        '9M',
        '136M',
        '270M'
    ],
    help='The agent to evaluate.',
    required=False,
)

def build_predictor(
      agent: str
    ) -> constants_src.Predictor:
  """Builds a transformer predictor."""
  match agent:
    case '9M':
      policy = 'action_value'
      num_layers = 8
      embedding_dim = 256
      num_heads = 8
    case '136M':
      policy = 'action_value'
      num_layers = 8
      embedding_dim = 1024
      num_heads = 8
    case '270M':
      policy = 'action_value'
      num_layers = 16
      embedding_dim = 1024
      num_heads = 8
    case 'local':
      policy = 'action_value'
      num_layers = 4
      embedding_dim = 64
      num_heads = 4
    case _:
      raise ValueError(f'Unknown model: {agent}')

  num_return_buckets = 128

  match policy:
    case 'action_value':
      output_size = num_return_buckets
    case 'behavioral_cloning':
      output_size = utils.NUM_ACTIONS
    case 'state_value':
      output_size = num_return_buckets

  predictor_config = transformer.TransformerConfig(
      vocab_size=utils.NUM_ACTIONS,
      output_size=output_size,
      pos_encodings=transformer.PositionalEncodings.LEARNED,
      max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
      num_heads=num_heads,
      num_layers=num_layers,
      embedding_dim=embedding_dim,
      apply_post_ln=True,
      apply_qk_layernorm=False,
      use_causal_mask=False,
  )
  predictor = transformer.build_transformer_predictor(config=predictor_config)
  return predictor


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
    
  # Obtenemos el predictor
  predictor = build_predictor(_AGENT.value)
  
  # Cargamos los parametros del modelo
  file_dir = os.path.dirname(os.path.abspath(__file__))
  checkpoint_dir = os.path.join(
      file_dir,
      f'../checkpoints/{_AGENT.value}'
  )
  params = training_utils.load_parameters(
      checkpoint_dir=checkpoint_dir,
      params=predictor.initial_params(
          rng=jrandom.PRNGKey(1),
          targets=np.ones((1, 1), dtype=np.uint32),
      ),
      step=6_400_000,
  )
  
  # Definimos el tamaño del batch
  bs = 32
  num_return_buckets = 128
  
  # Creamos la funcion predictora
  _, return_buckets_values = utils.get_uniform_buckets_edges_values(
      num_return_buckets
  )
  predict_fn = neural_engines.wrap_predict_fn(
      predictor=predictor,
      params=params,
      batch_size=bs,
  )
  
  # Creamos el dataloader
  data_config = config_lib.DataConfig(
      split='test',
      policy='action_value',
      num_return_buckets=num_return_buckets,
      batch_size=bs,
      shuffle=False,
      worker_count=1,
      seed=0,
      data_path=puzzles_file,
  )
  
  data_loader_csv = data_loader.build_data_loader_csv_all_moves(
      config=data_config
  )
  data_iter = data_loader_csv.__iter__()
  
  results_list = []
  fen_list = []
  
  ## Evaluamos cada par tablero-movimiento
  
  try:
    sequences, loss_mask, fens = next(data_iter)
    
    while True:
      print('data_loader:', data_iter.get_state())
      print('sequences', sequences)
      print('sequences.shape', sequences.shape)
      buckets_log_probs_array = predict_fn(sequences)[:, -1]
      
      for buckets_log_probs, fen in zip(buckets_log_probs_array, fens):
        if fen not in fen_list:
          fen_list.append(fen)
          
        win_probs = np.inner(np.exp(buckets_log_probs), return_buckets_values)
        
        dict_results = {
          'fen': fen,
          '%win': win_probs
        }
        
        results_list.append(dict_results)
      sequences, loss_mask, fens = next(data_iter)
  except StopIteration:
    print("Fin de la evaluacion")
  # Obtenemos las jugadas correspondientes a cada fen 
  #fen_list = np.unique(fen_list)
  move_list = np.array([])
  for fen in fen_list:
    board = chess.Board(fen)
    sorted_legal_moves = engine_lib.get_ordered_legal_moves(board)
    move_list = np.append(move_list, sorted_legal_moves)
  
  """
  print('move_list:', move_list)
  print('move_list:', len(move_list))
  print('results_dict:', results_list)
  print('results_dict:', len(results_list))
  """
  column_name = _AGENT.value + '_results'
  final_results_df = pd.DataFrame(results_list)
  final_results_df = final_results_df.loc[: move_list.shape[0] - 1]
  print('final_results_df:', final_results_df)
  final_results_df.to_csv('mid_csv2.csv', index=False)
  print('final_results_df:', final_results_df.shape)
  print('move_list:', move_list.shape)
  final_results_df['move'] = move_list.flatten()
  print('final_results_df:', final_results_df)
  
  final_results_df.to_csv(output_file, index=False)
  """
  _, return_buckets_values = utils.get_uniform_buckets_edges_values(128)
  
  data_loader = training_utils.build_data_loader(config_lib.DataConfig(
      split='test',
      policy='action_value',
      num_return_buckets=128,
      batch_size=32,
      shuffle=False,
      worker_count=1,
      seed=0,
  ))
  
  data_iter = data_loader.__iter__()
  
  sequences, loss_mask = next(data_iter)
  sequences = jax.lax.with_sharding_constraint(sequences, sharding)
  loss_mask = jax.lax.with_sharding_constraint(loss_mask, sharding)

  conditionals = predictor.predict(params=params, targets=sequences, rng=None)
 
  # Obtenemos el dataloader
  data_loader = training_utils.build_data_loader(
      config=constants.DataConfig
  )
  # Guardamos los resultados en un dataframe    
  #final_results_df = pd.DataFrame({column_name: results_list})
  #evaluated_puzzles = pd.concat([puzzles, final_results_df], axis=1)
  #evaluated_puzzles.to_csv(output_file, index=False)
  """
  
  
if __name__ == '__main__':
  app.run(main)
