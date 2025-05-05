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


from absl import app
from absl import flags
import chess
import chess.engine
import chess.pgn
import chess.svg
import pandas as pd
import numpy as np

from jax import random as jrandom
import time



# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose #logging is enabled.
#logger = #logging.getLogger(__name__)


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
            for move in board.legal_moves:
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

# Las siguientes clases implementan el motor de ajedrez 9M, 136M y 270M con profundidad. Se implementan las clases de tal forma que el código común (minimax y evaluate_actions) se encuentra en la clase base, y las clases hijas solo implementan el constructor y el método analyse. De esta forma, si se quiere cambiar el motor de ajedrez, solo hay que cambiar el constructor y el método analyse, sin necesidad de modificar el resto del código.

class ThinkMoreTemplate(ExampleEngine):
    """
    Get a move using searchless chess engine with tree search.
    """
    def __init__(self, depth=3):  # Agregamos `cwd` para evitar el error
        #super().__init__(commands, options, stderr, draw_or_resign, game, cwd=None)

        # Guardamos el motor en un atributo de la clase
        # Esta vez lo hacemos de forma manual para tener un mejor control de la arquitectura
        # y poder obtener las valoraciones de cada movimiento.
        self.depth=depth

    def analyse(self,
               board: chess.Board):
              # time_limit: Limit,
              # ponder: bool,  # noqa: ARG002
              # draw_offered: bool,
              # root_moves: MOVE):
            top_move = None
            depth = self.depth
            evals = []

            if board.turn == chess.WHITE:
                top_eval = -np.inf
            else:
                top_eval = np.inf
            #print('--------------INIT MINIMAX--------------')
            #logging.info(f"INIT MINIMAX with depth: {depth}")
            for move in engine.get_ordered_legal_moves(board):
                board.push(move)

                #logging.info(f"EVALUATING MOVE: {move}")

                # WHEN WE ARE BLACK, WE WANT TRUE AND TO GRAB THE SMALLEST VALUE
                #print("Turno:", board.turn)
                eval = self.minimax(board, depth - 1, -np.inf, np.inf, board.turn)

                board.pop()
                
                # Si el bot juega con piezas negras, invertimos las probabilidades
                if board.turn == chess.BLACK:
                    eval = 1 - eval
                    
                evals.append(eval)
                
                # Nos quedamos con la mejor jugada para el jugador actual
                if eval > top_eval:
                    top_move = move
                    top_eval = eval
                #logging.info(f"FINAL EVALUATION OF MOVE {move}: {eval}")
            #print("CHOSEN MOVE: ", top_move, "WITH EVAL: ", top_eval)
            # Devolvemos la jugada
            return {'top_move':PlayResult(top_move, None),'probs':evals}

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
                    #logging.warning("ALGO ANDA MAL: No hay jugadas legales, pero no hay outcome.")
                    exit()
            win_probs, _ = self.evaluate_actions(board, maximizing_player)
            if maximizing_player:
                best_win_prob = max(win_probs)
            else:
                best_win_prob = min(win_probs)
            #print("BEST WIN PROB: ", win_probability_to_centipawns(best_win_prob))
            return best_win_prob
        #print("MINIMAX AT DEPTH: ", depth)
        if maximizing_player:
            #print("WHITE PLAYER")
            max_eval = -np.inf
            for move in engine.get_ordered_legal_moves(board):
                #print("EVALUATING MOVE: ", move)
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                ##logging.info(f"NEW EVAL: {eval} \t\t MAX EVAL: {max_eval}")
                alpha = max(alpha, eval)
                if beta <= alpha:
                    #print(f"Poda: {beta} <= {alpha}")
                    break
            return max_eval
        else:
            #print("BLACK PLAYER")
            min_eval = np.inf
            for move in engine.get_ordered_legal_moves(board):
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                ##logging.info(f"NEW EVAL: {eval} \t\t MIN EVAL: {min_eval}")
                beta = min(beta, eval)
                if beta <= alpha:
                    #print(f"Poda: {beta} <= {alpha}")
                    break
            return min_eval

    def evaluate_actions(self, board, maximizing_player):
        results = self.analyse_without_depth(board)
        buckets_log_probs = results['log_probs']

        # Compute the expected return
        win_probs = np.inner(np.exp(buckets_log_probs), self.return_buckets_values)
        # Si el bot juega con piezas negras, invertimos las probabilidades
        if not maximizing_player:
            win_probs = -win_probs + 1
        #logging.info(f'###Probabilidades:{win_probs}. Maximizing player: {maximizing_player}')          
        sorted_legal_moves = engine.get_ordered_legal_moves(board)
        #print('WIN PROBS: ', win_probs)
        #print('SORTED LEGAL MOVES: ', sorted_legal_moves)

        return win_probs, sorted_legal_moves

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

class ThinkMore_9M(ThinkMoreTemplate):
    """
    Get a move using searchless chess 9M parameters engine with tree search.
    """
    def __init__(self, depth=3):  # Agregamos `cwd` para evitar el error
        super().__init__(depth)

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
            #logging.info(f"INIT MINIMAX with depth: {depth}")
            for move in engine_lib.get_ordered_legal_moves(board):
                board.push(move)

                #logging.info(f"EVALUATING MOVE: {move}")

                # WHEN WE ARE BLACK, WE WANT TRUE AND TO GRAB THE SMALLEST VALUE
                #print("Turno:", board.turn)
                eval = self.minimax(board, depth - 1, -np.inf, np.inf, board.turn)

                board.pop()
                
                # Si el bot juega con piezas negras, invertimos las probabilidades
                if board.turn == chess.BLACK:
                    eval = 1 - eval
                    
                evals.append(eval)
                
                # Nos quedamos con la mejor jugada para el jugador actual
                if eval > top_eval:
                    top_move = move
                    top_eval = eval
                #logging.info(f"FINAL EVALUATION OF MOVE {move}: {eval}")

            #print("CHOSEN MOVE: ", top_move, "WITH EVAL: ", top_eval)

            # Devolvemos la jugada
            return {'top_move':PlayResult(top_move, None),'probs':evals}



class ThinkMore_136M(ThinkMoreTemplate):
    """
    Get a move using searchless chess 136M parameters engine with tree search.
    """
    def __init__(self, depth=3):  # Agregamos `cwd` para evitar el error
        super().__init__(depth)

        # Guardamos el motor 136M en un atributo de la clase
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
            embedding_dim=1024,
            apply_post_ln=True,
            apply_qk_layernorm=False,
            use_causal_mask=False,
        )

        predictor = transformer.build_transformer_predictor(config=predictor_config)

        checkpoint_dir = os.path.join(

        os.getcwd(),
            f'searchless_chess/checkpoints/136M',
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
            #logging.info(f"INIT MINIMAX with depth: {depth}")
            for move in engine_lib.get_ordered_legal_moves(board):
                board.push(move)

                #logging.info(f"EVALUATING MOVE: {move}")

                # WHEN WE ARE BLACK, WE WANT TRUE AND TO GRAB THE SMALLEST VALUE
                #print("Turno:", board.turn)
                eval = self.minimax(board, depth - 1, -np.inf, np.inf, board.turn)

                board.pop()
                
                # Si el bot juega con piezas negras, invertimos las probabilidades
                if board.turn == chess.BLACK:
                    eval = 1 - eval
                    
                evals.append(eval)
                
                # Nos quedamos con la mejor jugada para el jugador actual
                if eval > top_eval:
                    top_move = move
                    top_eval = eval
                #logging.info(f"FINAL EVALUATION OF MOVE {move}: {eval}")

            #print("CHOSEN MOVE: ", top_move, "WITH EVAL: ", top_eval)

            # Devolvemos la jugada
            return {'top_move':PlayResult(top_move, None),'probs':evals}



class ThinkMore_270M(ThinkMoreTemplate):
    """
    Get a move using searchless chess 270M parameters engine with tree search.
    """
    def __init__(self, depth=3):  # Agregamos `cwd` para evitar el error
        super().__init__(depth)

        # Guardamos el motor 270M en un atributo de la clase
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
            num_layers=16,
            embedding_dim=1024,
            apply_post_ln=True,
            apply_qk_layernorm=False,
            use_causal_mask=False,
        )

        predictor = transformer.build_transformer_predictor(config=predictor_config)

        checkpoint_dir = os.path.join(

        os.getcwd(),
            f'searchless_chess/checkpoints/270M',
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
            #logging.info(f"INIT MINIMAX with depth: {depth}")
            for move in engine_lib.get_ordered_legal_moves(board):
                board.push(move)

                #logging.info(f"EVALUATING MOVE: {move}")

                # WHEN WE ARE BLACK, WE WANT TRUE AND TO GRAB THE SMALLEST VALUE
                #print("Turno:", board.turn)
                eval = self.minimax(board, depth - 1, -np.inf, np.inf, board.turn)

                board.pop()
                
                # Si el bot juega con piezas negras, invertimos las probabilidades
                if board.turn == chess.BLACK:
                    eval = 1 - eval
                    
                evals.append(eval)
                
                # Nos quedamos con la mejor jugada para el jugador actual
                if eval > top_eval:
                    top_move = move
                    top_eval = eval
                #logging.info(f"FINAL EVALUATION OF MOVE {move}: {eval}")

            #print("CHOSEN MOVE: ", top_move, "WITH EVAL: ", top_eval)

            # Devolvemos la jugada
            return {'top_move':PlayResult(top_move, None),'probs':evals}



class AlgorithmProposed(ThinkMoreTemplate):
    """
        Our proposed algorithm. A tree search with 9M evaluations and, in a certain level, a 270M reevaluations
    """
    def __init__(self,depth=5,reeval_level=2,N_moves_preev=10, iter_method = None, N_best = None, percentage = None, percentage_epsilon = None):
        super().__init__(depth)
        self.nodos_explorados = 0
        # Guardamos el motor 9M en un atributo de la clase para las predicciones oportunas
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
        self.predict_fn_9M = neural_engines.wrap_predict_fn(predictor, params, batch_size=32)
        _, self.return_buckets_values_9M = utils.get_uniform_buckets_edges_values(
            num_return_buckets
        )
        self._return_buckets_values_9M = self.return_buckets_values_9M
        self.neural_engine_9M = neural_engines.ENGINE_FROM_POLICY[policy](
            return_buckets_values=self.return_buckets_values_9M,
            predict_fn=self.predict_fn_9M,
            temperature=0.005,
        )
        # Guardamos el motor 270M en un atributo de la clase para las reevaluaciones
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
            num_layers=16,
            embedding_dim=1024,
            apply_post_ln=True,
            apply_qk_layernorm=False,
            use_causal_mask=False,
        )
        predictor = transformer.build_transformer_predictor(config=predictor_config)
        checkpoint_dir = os.path.join(
        os.getcwd(),
            f'searchless_chess/checkpoints/270M',
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
        self.predict_fn_270M = neural_engines.wrap_predict_fn(predictor, params, batch_size=32)
        _, self.return_buckets_values_270M = utils.get_uniform_buckets_edges_values(
            num_return_buckets
        )
        self._return_buckets_values_270M = self.return_buckets_values_270M
        self.neural_engine_270M = neural_engines.ENGINE_FROM_POLICY[policy](
            return_buckets_values=self.return_buckets_values_270M,
            predict_fn=self.predict_fn_270M,
            temperature=0.005,
        )

        # Guardamos el nivel de reevaluación en un atributo de la clase
        self.reeval_level = reeval_level

        # Guardamos el número de movimientos a preevaluar en un atributo de la clase
        self.N_moves_preev = N_moves_preev

        # Almacenamos el método iterativo para emplearlo en la búsqueda si se indicase
        self.iter_method = iter_method

        self.N_best = N_best
        self.percentage = percentage
        #self.epsilon = epsilon
        self.percentage_epsilon = percentage_epsilon


        self.n_puzzles_hechos = 0


    def analyse(self , board:chess.Board):
        # time_limit: Limit,
        # ponder: bool,  # noqa: ARG002
        # draw_offered: bool,
        # root_moves: MOVE):
            self.nodos_explorados = 0
            self.nodos_270M_evaluados = 0
            self.time_9M = 0
            self.time_270M = 0
            self.ramas_de_raiz_exploradas=0
         #   top_move = None
            depth = self.depth
            evals = []
            self.n_puzzles_hechos += 1

            # Opposite of our minimax
            start_time_preevaluations = time.perf_counter()
            # Now we get ordered legal moves decreasing according to the 270M engine
            win_probs, sorted_legal_moves = self.evaluate_actions_with_270M(board, True)

            # Ahora asociamos movimientos a sus probabilidades
            move_probs = dict(zip(sorted_legal_moves, win_probs))

            # Ordenamos según probabilidad según el turno es de negras o blancas
            
            aux = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)  
            movs_preevaled = aux[:self.N_moves_preev]


            logging.info("Probabilidades del 270M para cada movimiento: %s", aux)
            #logging.info("All moves %s", move_probs)
            logging.info("Info about preevaluations: %s", movs_preevaled)
            # Nos quedamos solo con esos N mejores movimientos según 270M
            movs_preevaled = [move for move, _ in movs_preevaled]
            end_time_preevaluations = time.perf_counter()
            logging.info("Time for preevaluations: %s seconds. Number of moves preevaluated: %s", end_time_preevaluations - start_time_preevaluations, len(movs_preevaled))
            #print('--------------INIT MINIMAX--------------')
            #logging.info(f"INIT MINIMAX with depth: {depth}")
            # Iteramos sobre los N mejores movimientos según 270M desde el mejor hasta el N-ésimo
            alpha = -np.inf
            beta = np.inf

            start_time_algorithm = time.perf_counter()
            for move in movs_preevaled:
                start_time_move_evaluation = time.perf_counter()
                board.push(move)

                #logging.info(f"EVALUATING MOVE: {move}")

                # WHEN WE ARE BLACK, WE WANT TRUE AND TO GRAB THE SMALLEST VALUE
                #print("Turno:", board.turn)
                self.ramas_de_raiz_exploradas+=1
                eval, _ = self.minimax(board, depth - 1, alpha, beta, board.turn)

                board.pop()
                
                # Si el bot juega con piezas negras, invertimos las probabilidades
                if board.turn == chess.BLACK:
                    eval = 1 - eval
                    
                evals.append(eval)
                
      
                if board.turn == chess.WHITE:
                    alpha = max(alpha, eval)
                else:
                    beta = min(beta, eval)
                end_time_move_evaluation = time.perf_counter()
                logging.info("Time for move evaluated: %s seconds. Move evaluated: %s. Evaluation: %s. Color player is white: %s", end_time_move_evaluation - start_time_move_evaluation, move, eval, board.turn)
                
                #logging.info(f"FINAL EVALUATION OF MOVE {move}: {eval}")
                if beta <= alpha:
                    logging.info("PODA NODO RAIZ")
                    break

            #print("CHOSEN MOVE: ", top_move, "WITH EVAL: ", top_eval)
            end_time_algorithm = time.perf_counter()
            logging.info("Time for algorithm: %s seconds. Number of moves evaluated: %s. Depth: %s. Reeval_level: %s. Color playes is white: %s", end_time_algorithm - start_time_algorithm, len(movs_preevaled), self.depth, self.reeval_level, board.turn)
           # logging.info("Selected move: %s", top_move)
           # #logging.info("Final evaluation of the move: %s", top_eval)
            logging.info("Se han explorado %s ramas desde el nodo raíz",self.ramas_de_raiz_exploradas)
            logging.info("Antes de salir del método de clase, la asociación mov-evaluación es\n")
            logging.info("final Puzzle número %s", self.n_puzzles_hechos)
            for move, eval in zip(movs_preevaled, evals):
                logging.info("final minimax Move: %s, Evaluation: %s", move, eval)

            logging.info("final ******************************************************")
            # Devolvemos la jugada
            return {'top_move':movs_preevaled,'probs':evals}
   
    def minimax(self, board: chess.Board, depth, alpha, beta, maximizing_player):
        self.nodos_explorados += 1

        if depth == 0 or board.is_game_over():
            if len(engine.get_ordered_legal_moves(board)) == 0:
                outcome = board.outcome()
                if outcome is not None:
                    if outcome.winner is None:
                        return 0.5, board  # tablas
                    elif outcome.termination == chess.Termination.CHECKMATE:
                        return 0.0001, board  # derrota
                else:
                    #logging.warning("No outcome found despite no legal moves.")
                    return 0.5, board

            # Medimos el tiempo de la evaluación y mostramos cuántos nodos ha evaluado hasta el momento
            st_time = time.perf_counter()
            win_probs, _ = self.evaluate_actions_with_9M(board, maximizing_player)
            et_time = time.perf_counter()
            self.time_9M+=(et_time-st_time)

            logging.info("Un nodo ha tardado en evaluarse: %s seconds. Nodos evaluados con 9M hasta el momento: %s. Tiempo de evaluación total hasta ahora de nodos con 9M: %s", et_time - st_time, self.nodos_explorados, self.time_9M)
            if maximizing_player:
                best_win_prob = max(win_probs)
            else:
                best_win_prob = min(win_probs)

            aux = "#" * (self.depth - depth)
            return best_win_prob, board

        best_board = None
        best_move = None

        if maximizing_player:
            max_eval = -np.inf
            movs = engine.get_ordered_legal_moves(board)
            if self.iter_method is not None:
                movs = self.get_best_moves(board,True)

            for move in movs:
                board.push(move)
                eval, next_board = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()

                if eval > max_eval:
                    max_eval = eval
                    best_board = next_board
                    best_move = move

                alpha = max(alpha, max_eval)    ###### max_eval ---> eval?
                if beta <= alpha:
                    ###logging.info("%s Poda en profundidad: %d", "#" * (self.depth - depth), depth)
                    break

            if depth == self.reeval_level and best_board is not None:
                st_time = time.perf_counter()
                win_probs, sorted_moves = self.evaluate_actions_with_270M(best_board, True)
                et_time = time.perf_counter()
                self.nodos_270M_evaluados+=1
                self.time_270M+=(et_time-st_time)
                logging.info("Tiempo en evaluar un nodo con 270M: %s. Nodos evaluados con 270M hasta ahora: %s. Tiempo total empleado en usar 270M: %s",et_time-st_time,self.nodos_270M_evaluados,self.time_270M)
                #move_probs = dict(zip(sorted_moves, win_probs))
                logging.info('depurando Probabilidad del mejor movimiento %s antes de 270M: %s', best_move,max_eval)
                reevaluated = max(win_probs) #move_probs.get(best_move, max_eval)
                logging.info('depurando Probabilidad del mejor movimiento (no necesariamente %s) después de 270M: %s. Nodos evaluados con 270M: %s', best_move,reevaluated, self.nodos_270M_evaluados)
                logging.info('depurando Probabilidad de los movimientos %s después de 270M: %s', sorted_moves,win_probs)
                #logging.info("%s Reevaluating -> %s", "#" * (self.depth - depth), reevaluated)
                return reevaluated, None
            else:
                return max_eval, best_board

        else:
            min_eval = np.inf
            movs = engine.get_ordered_legal_moves(board)
            if self.iter_method is not None:
                movs = self.get_best_moves(board,False)

            for move in movs:
                board.push(move)
                eval, next_board = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()

                if eval < min_eval:
                    min_eval = eval
                    best_board = next_board
                    best_move = move

                beta = min(beta, min_eval)      ###### min_eval ---> eval?
                if beta <= alpha:
                    #logging.info("%s Poda en profundidad: %d", "#" * (self.depth - depth), depth)
                    break

            if depth == self.reeval_level and best_board is not None:
                st_time = time.perf_counter()
                win_probs, sorted_moves = self.evaluate_actions_with_270M(best_board, True)
                et_time = time.perf_counter()
                self.nodos_270M_evaluados+=1
                self.time_270M+=(et_time-st_time)
                logging.info("Tiempo en evaluar un nodo con 270M: %s. Nodos evaluados con 270M hasta ahora: %s. Tiempo total empleado en usar 270M: %s",et_time-st_time,self.nodos_270M_evaluados,self.time_270M)
                #move_probs = dict(zip(sorted_moves, win_probs))
                logging.info('depurando Probabilidad del mejor movimiento %s antes de 270M: %s', best_move,min_eval)
                reevaluated = 1.0-max(win_probs) #move_probs.get(best_move, min_eval)
                logging.info('depurando Probabilidad del mejor movimiento (no necesariamente %s) después de 270M: %s. Nodos evaluados con 270M: %s', best_move,reevaluated,self.nodos_270M_evaluados)
                logging.info('depurando Probabilidad de los movimientos %s después de 270M: %s', sorted_moves,win_probs)
                #logging.info("%s Reevaluating -> %s", "#" * (self.depth - depth), reevaluated)
                return reevaluated, None
            else:
                return min_eval, best_board


    # Evalua todos los movimientos legales dado un tablero según el modelo 9M y obtiene los mejores según puntuaciones y método iterativo escogido
    def get_best_moves(self, board: chess.Board, maximize_player):
            win_probs_iter, sorteed_legal_moves_iter = self.evaluate_actions_with_9M(board, maximize_player)
            # Ahora ordenamos los movimientos según la probabilidad de victoria de mayor a menor
            move_probs = dict(zip(sorteed_legal_moves_iter, win_probs_iter))
            movs_preevaled = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)

            if self.iter_method == "N_best":
                assert self.N_best is not None
                movs_for_iter = [move for move, _ in movs_preevaled[:self.N_best]]
            if self.iter_method == "percentage":
                assert self.percentage is not None
                movs_for_iter = [move for move, _ in movs_preevaled[:int(len(movs_preevaled) * self.percentage)]]
            if self.iter_method == "epsilon":
                assert self.percentage_epsilon is not None
                epsilon = self.perc_part_of_max*max(win_probs_iter)

                # Nos quedamos con los movimientos con prob de victoria entre max-epsilon y max
                movs_for_iter = [move for move, _ in movs_preevaled if move_probs[move] >= max(win_probs_iter) - epsilon]


            return movs_for_iter


    def evaluate_actions_with_9M(self, board, maximizing_player):
        results = self.analyse_without_depth_with_9M(board)
        buckets_log_probs = results['log_probs']

        # Compute the expected return
        win_probs = np.inner(np.exp(buckets_log_probs), self.return_buckets_values_9M)
        # Si el bot juega con piezas negras, invertimos las probabilidades
        if not maximizing_player:
            win_probs = -win_probs + 1
        #logging.info(f'###Probabilidades:{win_probs}. Maximizing player: {maximizing_player}')          
        sorted_legal_moves = engine.get_ordered_legal_moves(board)
        #print('WIN PROBS: ', win_probs)
        #print('SORTED LEGAL MOVES: ', sorted_legal_moves)

        return win_probs, sorted_legal_moves

    def analyse_without_depth_with_9M(self, board: chess.Board) -> engine.AnalysisResult:
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
        return {'log_probs': self.predict_fn_9M(sequences)[:, -1], 'fen': board.fen()}
        

    def evaluate_actions_with_270M(self, board, maximizing_player):
        results = self.analyse_without_depth_with_270M(board)
        buckets_log_probs = results['log_probs']

        # Compute the expected return
        win_probs = np.inner(np.exp(buckets_log_probs), self.return_buckets_values_270M)
        # Si el bot juega con piezas negras, invertimos las probabilidades
        if not maximizing_player:
            win_probs = -win_probs + 1
        #logging.info(f'###Probabilidades:{win_probs}. Maximizing player: {maximizing_player}')          
        sorted_legal_moves = engine.get_ordered_legal_moves(board)
        #print('WIN PROBS: ', win_probs)
        #print('SORTED LEGAL MOVES: ', sorted_legal_moves)

        return win_probs, sorted_legal_moves

    def analyse_without_depth_with_270M(self, board: chess.Board) -> engine.AnalysisResult:
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
        return {'log_probs': self.predict_fn_270M(sequences)[:, -1], 'fen': board.fen()}
    

def heuristica_valor_material(board: chess.Board):
    valores = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }

    puntuacion = 0

    for pieza_tipo in valores:
        puntuacion+=len(board.pieces(pieza_tipo, chess.WHITE))*valores[pieza_tipo]
        puntuacion-=len(board.pieces(pieza_tipo, chess.BLACK))*valores[pieza_tipo]

    return puntuacion


def heuristica_valor_posicional_caballo(board: chess.Board):
    # Tabla de caballos para blancas (fila 0 = fila 1 del tablero)
    tabla_caballo = [
        [-5, -4, -3, -3, -3, -3, -4, -5],
        [-4, -2,  0,  0,  0,  0, -2, -4],
        [-3,  0,  1,  1.5, 1.5, 1,  0, -3],
        [-3, 0.5, 1.5, 2, 2, 1.5, 0.5, -3],
        [-3, 0.5, 1.5, 2, 2, 1.5, 0.5, -3],
        [-3, 0, 1, 1.5, 1.5, 1, 0, -3],
        [-4, -2, 0, 0.5, 0.5, 0, -2, -4],
        [-5, -4, -3, -3, -3, -3, -4, -5]
    ]

    puntuacion = 0
    for square in chess.SQUARES:
        pieza = board.piece_at(square)
        if pieza is not None:
            tipo = pieza.piece_type
            color = pieza.color
            valor = 0

            if tipo == chess.KNIGHT:
                fila = chess.square_rank(square)
                col = chess.square_file(square)
                if color == chess.WHITE:
                    valor += tabla_caballo[fila][col]
                else:
                    valor -= tabla_caballo[7 - fila][col]

            # Suma valor material
            if tipo == chess.PAWN:
                valor += 1
            elif tipo == chess.KNIGHT:
                valor += 3
            elif tipo == chess.BISHOP:
                valor += 3
            elif tipo == chess.ROOK:
                valor += 5
            elif tipo == chess.QUEEN:
                valor += 9
            # El rey se evalúa por posición o seguridad, lo ignoramos aquí

            if color == chess.WHITE:
                puntuacion += valor
            else:
                puntuacion -= valor

    return puntuacion

def MiguelHeuristic(move, board: chess.Board):
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000  # No se usa realmente en evaluación
    }


    score = 0

    # Si es una captura
    if board.is_capture(move):
        src_piece = board.piece_at(move.from_square)
        dest_piece = board.piece_at(move.to_square)

        if dest_piece is not None and src_piece is not None:   # El segundo condicional lo hemos añadido
            score += 10 * PIECE_VALUES[dest_piece.piece_type] - PIECE_VALUES[src_piece.piece_type]
        elif board.is_en_passant(move):
            score += PIECE_VALUES[chess.PAWN]

    # Si es una promoción
    if move.promotion:
        score += PIECE_VALUES[move.promotion]

    # Simular el tablero después de hacer la jugada
    new_board = board.copy()
    new_board.push(move)

    # Si después de mover el rey queda bajo ataque
    king_square = new_board.king(not board.turn)
    if king_square is not None and new_board.is_attacked_by(board.turn, king_square):
        score += 50

    return score
