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

from lichess_bot.homemade import ThinkMore_9M#, ThinkMore_136M, ThinkMore_270M

from jax import random as jrandom
import json
import time

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
        'local', '9M', '136M', '270M',
        '9M_Depth', '136M_Depth', '270M_Depth',
        'stockfish', 'stockfish_all_moves',
        'leela_chess_zero_depth_1',
        'leela_chess_zero_policy_net',
        'leela_chess_zero_400_sims',
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


def analyse_puzzle_from_board_with_LST(
    board: chess.Board,
    engine: engine_lib.Engine
) -> dict:
  """Returns the evaluation of all posible moves by LST (or a neural model), ordered by CP."""
  
  # Obtenemos la jugada del motor a evaluar (LST)
  buckets_log_probs = engine.analyse(board)['log_probs']
  win_probs = np.inner(np.exp(buckets_log_probs), engine._return_buckets_values)
  sorted_legal_moves = engine_lib.get_ordered_legal_moves(board)
  
  dict_results = {
    move.uci(): {
      'wp': wp, 
      'cp': win_probability_to_centipawns(wp)
    } for move, wp in zip(sorted_legal_moves, win_probs)
  }
  
  return dict_results

def get_engine(model_name: str, bs: int, limit: chess.engine.Limit = None):
    if model_name in ['stockfish', 'stockfish_all_moves']:
        return constants.ENGINE_BUILDERS[model_name](limit=limit)
    else:
        match model_name:
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
            raise ValueError(f'Unknown model: {model_name}')

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
        file_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.normpath(os.path.join(
            file_dir,
            f'../checkpoints/{model_name}'
        ))

        params = training_utils.load_parameters(
            checkpoint_dir=checkpoint_dir,
            params=predictor.initial_params(
                rng=jrandom.PRNGKey(1),
                targets=np.ones((1, 1), dtype=np.uint32),
            ),
            step=6_400_000,
        )

        bs = bs
        
        _, return_buckets_values = utils.get_uniform_buckets_edges_values(
            num_return_buckets
        )
        return neural_engines.ENGINE_FROM_POLICY[policy](
            return_buckets_values=return_buckets_values,
            predict_fn=neural_engines.wrap_predict_fn(
                predictor=predictor,
                params=params,
                batch_size=bs,
            ),
            temperature=0.005,
        )

def main(argv: Sequence[str]) -> None:
    batchsize_0 = 32
    puzzles_file = _INPUT_FILE.value
    output_file = _OUTPUT_FILE.value or os.path.splitext(puzzles_file)[0] + "_solved.csv"
    puzzles = pd.read_csv(puzzles_file, nrows=_NUM_PUZZLES.value) if _NUM_PUZZLES.value else pd.read_csv(puzzles_file)
    predictor = build_predictor(_AGENT.value)
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../checkpoints/{_AGENT.value}')
    params = training_utils.load_parameters(
        checkpoint_dir=checkpoint_dir,
        params=predictor.initial_params(rng=jrandom.PRNGKey(1), targets=np.ones((1, 1), dtype=np.uint32)),
        step=6_400_000,
    )

    num_return_buckets = 128
    _, return_buckets_values = utils.get_uniform_buckets_edges_values(num_return_buckets)

    print(f'INPUT FILE: {puzzles_file}')
    print(f'OUTPUT FILE: {output_file}')

    print("|---------------------------------------------------------------------------------------------------------------------------------|")
    print("|          BATCHSIZE          |          Sin DataLoader          |           Con DataLoader          |          Cociente          |")
    print("|---------------------------------------------------------------------------------------------------------------------------------|")
   

    while True:
        # Resolución del predictor SIN dataloader
        start_sin = time.perf_counter()
        engine = get_engine(_AGENT.value,batchsize_0)
        results_list = []
        for i,puzzle in puzzles.iterrows():
            board = chess.Board(puzzle['FEN'])
            move = puzzle['Moves_UCI']

            results = analyse_puzzle_from_board_with_LST(
              board=board,
              engine=engine
            )
            results_list.append(results)

        column_name = _AGENT.value + '_results'
        final_results_df = pd.DataFrame({column_name: results_list})
        evaluated_puzzles = pd.concat([puzzles, final_results_df], axis=1)
        evaluated_puzzles.to_csv(output_file, index=False)
        
        end_sin = time.perf_counter()
        tiempo_sin = end_sin-start_sin
        # Resolución del predictor CON dataloader
        start_con = time.perf_counter()
         # Motor basado en modelo
        engine = neural_engines.wrap_predict_fn(
            predictor=predictor,
            params=params,
            batch_size=batchsize_0,
        )

        # Construimos el dataloader
        data_config = config_lib.DataConfig(
            split='test',
            policy='action_value',
            num_return_buckets=num_return_buckets,
            batch_size=batchsize_0,
            shuffle=False,
            worker_count=1,
            seed=0,
            data_path=puzzles_file,
        )
        data_loader_csv = data_loader.build_data_loader_csv_all_moves(config=data_config)
        data_iter = iter(data_loader_csv)

        results_list = []
        fen_list = []

        try:
            sequences, loss_mask, fens = next(data_iter)
            while True:
                buckets_log_probs_array = engine(sequences)[:, -1]
                for buckets_log_probs, fen in zip(buckets_log_probs_array, fens):
                    if fen not in fen_list:
                        fen_list.append(fen)
                    win_probs = np.inner(np.exp(buckets_log_probs), return_buckets_values)
                    results_list.append({'fen': fen, '%win': win_probs})
                sequences, loss_mask, fens = next(data_iter)
        except StopIteration:
            pass

        move_list = []
        for fen in fen_list:
            board = chess.Board(fen)
            ordered_moves = engine_lib.get_ordered_legal_moves(board)
            move_list.extend(ordered_moves)

        final_results_df = pd.DataFrame(results_list)
        final_results_df = final_results_df.loc[:len(move_list) - 1]
        final_results_df['move'] = [move.uci() for move in move_list]

        final_results_df.to_csv(output_file, index=False)
        end_con = time.perf_counter()
        tiempo_con = end_con-start_con

        print(f"|          {batchsize_0}          |          {tiempo_sin:.6f}          |           {tiempo_con:.6f}          |          {(tiempo_con/tiempo_sin):.6f}          |")

        batchsize_0 = batchsize_0*2

    print("FIN DEL EXPERIMENTO. EXPLOTÓ POR EL BATCHSIZE")

if __name__ == '__main__':
  app.run(main)