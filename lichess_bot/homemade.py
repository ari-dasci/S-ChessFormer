"""
Some example classes for people who want to create a homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""
import chess
from chess.engine import PlayResult, Limit
import random
from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import MOVE, HOMEMADE_ARGS_TYPE
import logging
import gc

# Imports for searchless_chess
from searchless_chess.src.engines import constants
from searchless_chess.src.engines import engine as engine_lib
from searchless_chess.src import tokenizer
from searchless_chess.src import training_utils
from searchless_chess.src import transformer
from searchless_chess.src import utils
from searchless_chess.src.engines import engine
from searchless_chess.src.engines import neural_engines

from collections.abc import Sequence
import io
import os
import sys
import math
import jax

from absl import app
from absl import flags
import chess
import chess.engine
import chess.pgn
import chess.svg
import pandas as pd
import numpy as np

from jax import random as jrandom




# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)


class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""


# Bot names and ideas from tom7's excellent eloWorld video

class RandomMove(ExampleEngine):
    """Get a random move."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose a random move."""
        return PlayResult(random.choice(list(board.legal_moves)), None)


class Alphabetical(ExampleEngine):
    """Get the first move when sorted by san representation."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose the first move alphabetically."""
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return PlayResult(moves[0], None)


class FirstMove(ExampleEngine):
    """Get the first move when sorted by uci representation."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose the first move alphabetically in uci representation."""
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)


class ComboEngine(ExampleEngine):
    """
    Get a move using multiple different methods.

    This engine demonstrates how one can use `time_limit`, `draw_offered`, and `root_moves`.
    """

    def search(self,
               board: chess.Board,
               time_limit: Limit,
               ponder: bool,  # noqa: ARG002
               draw_offered: bool,
               root_moves: MOVE) -> PlayResult:
        """
        Choose a move using multiple different methods.

        :param board: The current position.
        :param time_limit: Conditions for how long the engine can search (e.g. we have 10 seconds and search up to depth 10).
        :param ponder: Whether the engine can ponder after playing a move.
        :param draw_offered: Whether the bot was offered a draw.
        :param root_moves: If it is a list, the engine should only play a move that is in `root_moves`.
        :return: The move to play.
        """
        if isinstance(time_limit.time, int):
            my_time = time_limit.time
            my_inc = 0
        elif board.turn == chess.WHITE:
            my_time = time_limit.white_clock if isinstance(time_limit.white_clock, int) else 0
            my_inc = time_limit.white_inc if isinstance(time_limit.white_inc, int) else 0
        else:
            my_time = time_limit.black_clock if isinstance(time_limit.black_clock, int) else 0
            my_inc = time_limit.black_inc if isinstance(time_limit.black_inc, int) else 0

        possible_moves = root_moves if isinstance(root_moves, list) else list(board.legal_moves)

        if my_time / 60 + my_inc > 10:
            # Choose a random move.
            move = random.choice(possible_moves)
        else:
            # Choose the first move alphabetically in uci representation.
            possible_moves.sort(key=str)
            move = possible_moves[0]
        return PlayResult(move, None, draw_offered=draw_offered)


# Searchless chess models

class ThinkLess_9M(ExampleEngine):
    """
    Get a move using searchless chess 9M parameters engine.
    """
    def __init__(self, commands, options, stderr, draw_or_resign, game, cwd=None):  # Agregamos `cwd` para evitar el error
        # Guardamos el motor en un atributo de la clase
        super().__init__(commands, options, stderr, draw_or_resign, game, cwd=None)
        self.search_engine = constants.ENGINE_BUILDERS['9M']()

    def search(self,
               board: chess.Board,
               time_limit: Limit,
               ponder: bool,  # noqa: ARG002
               draw_offered: bool,
               root_moves: MOVE):
        # Usamos el motor para realizar la predicción
        predicted_move = self.search_engine.play(board=board).uci()

        # Devolvemos la jugada
        return PlayResult(predicted_move, None)

class ThinkLess_136M(ExampleEngine):
    """
    Get a move using searchless chess 136M parameters engine.
    """
    def __init__(self, commands, options, stderr, draw_or_resign, game, cwd=None):  # Agregamos `cwd` para evitar el error
        # Guardamos el motor en un atributo de la clase
        super().__init__(commands, options, stderr, draw_or_resign, game, cwd=None)
        self.search_engine = constants.ENGINE_BUILDERS['136M']()

    def search(self,
               board: chess.Board,
               time_limit: Limit,
               ponder: bool,  # noqa: ARG002
               draw_offered: bool,
               root_moves: MOVE):
        # Usamos el motor para realizar la predicción
        predicted_move = self.search_engine.play(board=board).uci()

        gc.collect()
        # Devolvemos la jugada
        return PlayResult(predicted_move, None)

class ThinkLess_270M(ExampleEngine):
    """
    Get a move using searchless chess 270M parameters engine.
    """
    def __init__(self, commands, options, stderr, draw_or_resign, game, cwd=None):  # Agregamos `cwd` para evitar el error
        # Guardamos el motor en un atributo de la clase
        super().__init__(commands, options, stderr, draw_or_resign, game, cwd=None)
        self.search_engine = constants.ENGINE_BUILDERS['270M']()

    def search(self,
               board: chess.Board,
               time_limit: Limit,
               ponder: bool,  # noqa: ARG002
               draw_offered: bool,
               root_moves: MOVE):
        # Usamos el motor para realizar la predicción
        predicted_move = self.search_engine.play(board=board).uci()

        # Devolvemos la jugada
        return PlayResult(predicted_move, None)
    
    def __del__(self):
        try:
            print("Liberando memoria de GPU...")
            # Borra atributos que puedan estar ocupando memoria
            del self.search_engine

            # Fuerza la recolección de basura
            gc.collect()
            
        except Exception as e:
            print(f"Error liberando memoria: {e}")


def win_probability_to_centipawns(win_probability: float) -> int:
  """Returns the centipawn score converted from the win probability (in [0, 1]).

  Args:
    win_probability: The win probability in the range [0, 1].
  """
  if not 0 <= win_probability <= 1:
    raise ValueError("Win probability must be in the range [0, 1].")

  centipawns = -1 / 0.00368208 * math.log((1 - win_probability) / win_probability)
  return int(centipawns)

class ThinkMore_9M_bot(ExampleEngine):
    """
    Get a move using searchless chess 9M parameters engine with tree search.
    """
    def __init__(self, commands, options, stderr, draw_or_resign, game, cwd=None):  # Agregamos `cwd` para evitar el error
        super().__init__(commands, options, stderr, draw_or_resign, game, cwd=None)

        # Guardamos el motor 9M en un atributo de la clase
        # Esta vez lo hacemos de forma manual para tener un mejor control de la arquitectura
        # y poder obtener las valoraciones de cada movimiento.
        policy = 'action_value'
        num_return_buckets = 128
        output_size = num_return_buckets

        predictor_config = transformer.TransformerConfig(
            vocab_size=utils.NUM_ACTIONS,
            output_size=output_size,
            pos_encodings=transformer.PositionalEncodings.LEARNED,
            max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
            num_heads=8,
            num_layers=8,
            embedding_dim=256,
            apply_post_ln=True,
            apply_qk_layernorm=False,
            use_causal_mask=False,
        )

        predictor = transformer.build_transformer_predictor(config=predictor_config)

        checkpoint_dir = os.path.join(

        os.getcwd(),
            f'engines/searchless_chess/checkpoints/9M',
        )
        dummy_params = predictor.initial_params(
            rng=jrandom.PRNGKey(6400000),
            targets=np.zeros((1, 1), dtype=np.uint32),
        )
        params = training_utils.load_parameters(
            checkpoint_dir=checkpoint_dir,
            params=dummy_params,
            use_ema_params=True,
            step=-1,
        )

        predict_fn = neural_engines.wrap_predict_fn(predictor, params, batch_size=1)

        _, self.return_buckets_values = utils.get_uniform_buckets_edges_values(
            num_return_buckets
        )

        self.neural_engine = neural_engines.ENGINE_FROM_POLICY[policy](
            return_buckets_values=self.return_buckets_values,
            predict_fn=predict_fn,
            temperature=0.005,
        )

    def search(self,
               board: chess.Board,
               time_limit: Limit,
               ponder: bool,  # noqa: ARG002
               draw_offered: bool,
               root_moves: MOVE):

            top_move = None
            depth = 3
            # Opposite of our minimax
            if board.turn == chess.WHITE:
                top_eval = -np.inf
            else:
                top_eval = np.inf
            #print('--------------INIT MINIMAX--------------')
            for move in engine_lib.get_ordered_legal_moves(board):
                board.push(move)

                #print("EVALUATING MOVE: ", move)

                # WHEN WE ARE BLACK, WE WANT TRUE AND TO GRAB THE SMALLEST VALUE
                #print("Turno:", board.turn)
                eval = self.minimax(board, depth - 1, -np.inf, np.inf, board.turn)

                board.pop()

                if board.turn == chess.WHITE:
                    if eval > top_eval:
                        top_move = move
                        top_eval = eval
                else:
                    if eval < top_eval:
                        top_move = move
                        top_eval = eval

            #print("CHOSEN MOVE: ", top_move, "WITH EVAL: ", top_eval)

            # Devolvemos la jugada
            return PlayResult(top_move, None)

    def evaluate_actions(self, board):
        results = self.neural_engine.analyse(board)
        buckets_log_probs = results['log_probs']

        # Compute the expected return.
        win_probs = np.inner(np.exp(buckets_log_probs), self.return_buckets_values)
        sorted_legal_moves = engine.get_ordered_legal_moves(board)
        #print('WIN PROBS: ', win_probs)
        #print('SORTED LEGAL MOVES: ', sorted_legal_moves)

        for i in np.argsort(win_probs)[:-3:-1]:
            print(i)
            cp = win_probability_to_centipawns(win_probs[i])
            print(f'  {sorted_legal_moves[i].uci()} -> {100*win_probs[i]:.1f}% cp: {cp}')

        return win_probs, sorted_legal_moves


    def minimax(self, board, depth, alpha, beta, maximizing_player):
        #print("DEPTH: ", depth)
        if depth == 0 or board.is_game_over():
            win_probs, _ = self.evaluate_actions(board)
            if maximizing_player:
                best_win_prob = min(win_probs)
            else:
                best_win_prob = max(win_probs)
            #print("BEST WIN PROB: ", win_probability_to_centipawns(best_win_prob))
            return best_win_prob

        if maximizing_player:
            max_eval = -np.inf
            for move in board.legal_moves:
                #print("EVALUATING MOVE: ", move)
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    #print(f"Poda: {beta} <= {alpha}")
                    break
            return max_eval
        else:
            min_eval = np.inf
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    #print(f"Poda: {beta} <= {alpha}")
                    break
            return min_eval


class ThinkMore_9M(ExampleEngine):
    """
    Get a move using searchless chess 9M parameters engine with tree search.
    """
    def __init__(self, depth=3):  # Agregamos `cwd` para evitar el error
        #super().__init__(commands, options, stderr, draw_or_resign, game, cwd=None)

        # Guardamos el motor 9M en un atributo de la clase
        # Esta vez lo hacemos de forma manual para tener un mejor control de la arquitectura
        # y poder obtener las valoraciones de cada movimiento.
        policy = 'action_value'
        num_return_buckets = 128
        output_size = num_return_buckets
        self.depth=depth
        predictor_config = transformer.TransformerConfig(
            vocab_size=utils.NUM_ACTIONS,
            output_size=output_size,
            pos_encodings=transformer.PositionalEncodings.LEARNED,
            max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
            num_heads=8,
            num_layers=8,
            embedding_dim=256,
            apply_post_ln=True,
            apply_qk_layernorm=False,
            use_causal_mask=False,
        )

        predictor = transformer.build_transformer_predictor(config=predictor_config)

        checkpoint_dir = os.path.join(

        os.getcwd(),
            f'searchless_chess/checkpoints/9M',
        )
        dummy_params = predictor.initial_params(
            rng=jrandom.PRNGKey(6400000),
            targets=np.zeros((1, 1), dtype=np.uint32),
        )
        params = training_utils.load_parameters(
            checkpoint_dir=checkpoint_dir,
            params=dummy_params,
            use_ema_params=True,
            step=-1,
        )

        self.predict_fn = neural_engines.wrap_predict_fn(predictor, params, batch_size=32)

        _, self.return_buckets_values = utils.get_uniform_buckets_edges_values(
            num_return_buckets
        )

        self._return_buckets_values = self.return_buckets_values

        self.neural_engine = neural_engines.ENGINE_FROM_POLICY[policy](
            return_buckets_values=self.return_buckets_values,
            predict_fn=self.predict_fn,
            temperature=0.005,
        )

    def analyse(self,
               board: chess.Board):
              # time_limit: Limit,
              # ponder: bool,  # noqa: ARG002
              # draw_offered: bool,
              # root_moves: MOVE):

            top_move = None
            depth = self.depth
            evals = []

            # Opposite of our minimax
            if board.turn == chess.WHITE:
                top_eval = -np.inf
            else:
                top_eval = np.inf
            #print('--------------INIT MINIMAX--------------')
            logging.info(f"INIT MINIMAX with depth: {depth}")
            for move in engine_lib.get_ordered_legal_moves(board):
                board.push(move)

                logging.info(f"EVALUATING MOVE: {move}")

                # WHEN WE ARE BLACK, WE WANT TRUE AND TO GRAB THE SMALLEST VALUE
                #print("Turno:", board.turn)
                eval = self.minimax(board, depth - 1, -np.inf, np.inf, board.turn)

                logging.info(f"FINAL EVALUATION OF MOVE {move}: {eval}")
                board.pop()
                
                # Si el bot juega con piezas negras, invertimos las probabilidades
                if board.turn == chess.BLACK:
                    eval = 1 - eval
                    
                evals.append(eval)
                
                # Nos quedamos con la mejor jugada para el jugador actual
                if eval > top_eval:
                    top_move = move
                    top_eval = eval
                

            #print("CHOSEN MOVE: ", top_move, "WITH EVAL: ", top_eval)

            # Devolvemos la jugada
            return {'top_move':PlayResult(top_move, None),'probs':evals}

    def evaluate_actions(self, board, maximizing_player):
        results = self.analyse_without_depth(board)
        buckets_log_probs = results['log_probs']

        # Compute the expected return
        win_probs = np.inner(np.exp(buckets_log_probs), self.return_buckets_values)
        # Si el bot juega con piezas negras, invertimos las probabilidades
        if not maximizing_player:
            win_probs = -win_probs + 1
            
        sorted_legal_moves = engine.get_ordered_legal_moves(board)
        #print('WIN PROBS: ', win_probs)
        #print('SORTED LEGAL MOVES: ', sorted_legal_moves)

        return win_probs, sorted_legal_moves


    def minimax(self, board : chess.Board, depth, alpha, beta, maximizing_player):
        #print("DEPTH: ", depth)
        if depth == 0 or board.is_game_over():
            # Si es game over puede pasar:
            # - mate: si lo doy yo, 1 si soy blancas (resp. -1 si soy negras), y -1 si me lo da el oponente (resp. ...)
            # - tablas: 0.5---> hay empate
            #  ---> Conviene usar 'outcome' de Chess, que da lo que ocurre en el tablero
            
            if len(engine.get_ordered_legal_moves(board)) == 0:  # No hay jugadas legales: puede ser jaque mate o tablas
                situation = board.outcome()
                
                if situation is not None:
                    who_wins = situation.winner
                    if who_wins is None:
                        # Hay tablas: ahogado, repetición, material insuficiente, etc.
                        return 0.5
                    elif situation.termination == chess.Termination.CHECKMATE:
                        return 0.0001  # el jugador actual ha perdido
                else:
                    logging.warning("ALGO ANDA MAL: No hay jugadas legales, pero no hay outcome.")
                    exit()

            win_probs, _ = self.evaluate_actions(board, maximizing_player)
            if maximizing_player:
                best_win_prob = max(win_probs)
            else:
                best_win_prob = min(win_probs)
            #print("BEST WIN PROB: ", win_probability_to_centipawns(best_win_prob))
            return best_win_prob
        
        logging.info(f"MINIMAX AT DEPTH: {depth}")
        if maximizing_player:
            logging.info(f"WHITE PLAYER")
            max_eval = -np.inf
            for move in engine.get_ordered_legal_moves(board):
                #print("EVALUATING MOVE: ", move)
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                logging.info(f"NEW EVAL: {eval} \t\t MAX EVAL: {max_eval}")
                alpha = max(alpha, eval)
                if beta <= alpha:
                    #print(f"Poda: {beta} <= {alpha}")
                    break
            return max_eval
        else:
            logging.info(f"BLACK PLAYER")
            min_eval = np.inf
            for move in engine.get_ordered_legal_moves(board):
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                logging.info(f"NEW EVAL: {eval} \t\t MIN EVAL: {min_eval}")
                beta = min(beta, eval)
                if beta <= alpha:
                    #print(f"Poda: {beta} <= {alpha}")
                    break
            return min_eval


    def analyse_without_depth(self, board: chess.Board) -> engine.AnalysisResult:
        """Returns buckets log-probs for each action, and FEN."""
        # Tokenize the legal actions.
        sorted_legal_moves = engine.get_ordered_legal_moves(board)
        legal_actions = [utils.MOVE_TO_ACTION[x.uci()] for x in sorted_legal_moves]
        legal_actions = np.array(legal_actions, dtype=np.int32)
        legal_actions = np.expand_dims(legal_actions, axis=-1)
        # Tokenize the return buckets.
        dummy_return_buckets = np.zeros((len(legal_actions), 1), dtype=np.int32)
        # Tokenize the board.
        tokenized_fen = tokenizer.tokenize(board.fen()).astype(np.int32)
        sequences = np.stack([tokenized_fen] * len(legal_actions))
        # Create the sequences.
        sequences = np.concatenate(
            [sequences, legal_actions, dummy_return_buckets],
            axis=1,
        )
        return {'log_probs': self.predict_fn(sequences)[:, -1], 'fen': board.fen()}
