#### Métricas interesantes para CSV
# a) Tablero / IdTablero  OK
# b) Movimiento mejor (Stockfish) OK
# c) Movimiento mejor (LST) OK
# d) % victoria mejor Stockfish (según Stockfish) OK
# e) % victoria mejor LST (según Stockfish) OK
# f) % victoria mejor Stockfish (según LST)
# g) % victoria mejor LST (según LST)
# h) Diferencia entre abs(Stockfish-LST) (según % Stockfish). abs( d)-e) )
# i) Diferencia entre abs(Stockfish-LST) (según % LST). abs( f)-g) )
# Ídem con centipeones de d) hasta i)

import utils
import os
import numpy as np
import pandas as pd
from engines.searchless_chess.src.engines import engine as aux_engine

TANG_FUNCTION = "tg"
TAYLOR_FUNCTION = "taylor"
OUTPUT_PUZZLES_FILENAME = "metrics_results/metrics.csv"
OUTPUT_LICHESS_FILENAME = "metrics_results/lichess_metrics.csv"

def centipawns_to_win_probability_tang(centipawns):
    return 111.714640912*np.tan(1.5620688421*centipawns)

def win_probability_to_centipawns_tang(prob_win):
    return np.arctan(prob_win / 111.714640912) / 1.5620688421


def createCSVFile():
    if not os.path.isfile(OUTPUT_PUZZLES_FILENAME):
        with open(OUTPUT_PUZZLES_FILENAME,'w') as file:
            file.write('id,best_movs_St,best_movs_LST,probs_St_accord_St,probs_LST_accord_St,probs_St_accord_LST,probs_LST_accord_LST,diffs_probs_accord_St,diffs_probs_accord_LST,centipawns_St_accord_St,centipawns_LST_accord_St,centipawns_St_accord_LST,centipawns_LST_accord_LST,diffs_centipawns_accord_St,diffs_centipawns_accord_LST,diffs_centipawns_accord_St,diffs_accord_centipawns_LST\n')

def writeWithinCSV(num_puzzle,best_movs_St,best_movs_LST,probs_St_accord_St,probs_LST_accord_St,probs_St_accord_LST,probs_LST_accord_LST,diffs_probs_accord_St,diffs_probs_accord_LST,centipawns_St_accord_St,centipawns_LST_accord_St,centipawns_St_accord_LST,centipawns_LST_accord_LST,diffs_centipawns_accord_St,diffs_centipawns_accord_LST):
    with open(OUTPUT_PUZZLES_FILENAME,'a') as file:
        file.write(f"{num_puzzle},{best_movs_St},{best_movs_LST},{probs_St_accord_St},{probs_LST_accord_St},{probs_St_accord_LST},{probs_LST_accord_LST},{diffs_probs_accord_St},{diffs_probs_accord_LST},{centipawns_St_accord_St},{centipawns_LST_accord_St},{centipawns_St_accord_LST},{centipawns_LST_accord_LST},{diffs_centipawns_accord_St},{diffs_centipawns_accord_LST},{diffs_centipawns_accord_St},{diffs_centipawns_accord_LST}\n")

def save_lichess_metrics(num_game,move,pi_accord_St,pi_accord_LST,centipawns_accord_St,centipawns_accord_LST):
    if not os.path.exists(OUTPUT_LICHESS_FILENAME,"w"):
        with open(OUTPUT_LICHESS_FILENAME,"w") as file:
            file.write("id_game,moves,probs_accord_St,probs_accord_LST,centipawns_accord_St,centipawns_accord_LST\n")

    df = pd.read_csv()
    with open(OUTPUT_LICHESS_FILENAME,"w") as file:
        if num_game in df["id_game"].values:
            idx = df[df["id_game"]==num_game].index[0]

            # Convertimos las columnas a listas si no lo están
            df.at[idx, "moves"] = eval(df.at[idx, "moves"]) if isinstance(df.at[idx, "moves"], str) else df.at[idx, "moves"]
            df.at[idx, "probs_accord_St"] = eval(df.at[idx, "probs_accord_St"]) if isinstance(df.at[idx, "probs_accord_St"], str) else df.at[idx, "probs_accord_St"]
            df.at[idx, "probs_accord_LST"] = eval(df.at[idx, "probs_accord_LST"]) if isinstance(df.at[idx, "probs_accord_LST"], str) else df.at[idx, "probs_accord_LST"]
            df.at[idx, "centipawns_accord_St"] = eval(df.at[idx, "centipawns_accord_St"]) if isinstance(df.at[idx, "centipawns_accord_St"], str) else df.at[idx, "centipawns_accord_St"]
            df.at[idx, "centipawns_accord_LST"] = eval(df.at[idx, "centipawns_accord_LST"]) if isinstance(df.at[idx, "centipawns_accord_LST"], str) else df.at[idx, "centipawns_accord_LST"]

            df.at[idx, "moves"] += " " + move
            df.at[idx, "probs_accord_St"] += pi_accord_St
            df.at[idx, "probs_accord_LST"] += pi_accord_LST
            df.at[idx, "centipawns_accord_St"] += centipawns_accord_St
            df.at[idx, "centipawns_accord_LST"] += centipawns_accord_LST
        else:
            df.concat([df,
                {
                    "id_game":num_game,
                    "moves": move,
                    "probs_accord_St": pi_accord_St,
                    "probs_accord_LST": pi_accord_LST,
                    "centipawns_accord_St": centipawns_accord_St,
                    "centipawns_accord_LST": centipawns_accord_LST
                }]
                ,
                ignore_index=True
            )
        df.to_csv(OUTPUT_LICHESS_FILENAME,index=False)


def computeMetricsAccordStockfish(move, move_St, board, stockfish_engine, which_function):
    aux_board_1 = board.copy()

    # Evaluamos la jugada de LST
   # aux_board_1.push(utils.chess.Move.from_uci(move))
    analysis_results_1 = stockfish_engine.analyse(aux_board_1)

    centipawns_LST = None
    centipawns_St = None


    # Extraer el valor de centipawns para el movimiento de LST
    for mv, score in analysis_results_1['scores']:

        if mv == move:
            centipawns_LST = score.score()
            break

    if centipawns_LST is None:     # Si no da Stockfish el mov jugado, se juega el del LST y se coge el mov de mejor puntuación de Stockfish
        aux_board_1.push(utils.chess.Move.from_uci(move))
        analysis_results_1 = stockfish_engine.analyse(aux_board_1)
        if board.turn:   # Juegan las blancas
            centipawns_LST = max(analysis_results_1['scores'],key=lambda x:x[1])[1]
        else:
            centipawns_LST = min(analysis_results_1['scores'],key=lambda x:x[1])[1]

    # Si move_St no es None, evaluamos también el movimiento de Stockfish
    if move_St is not None:
        aux_board_2 = board.copy()
        #aux_board_2.push(utils.chess.Move.from_uci(move_St))
        analysis_results_2 = stockfish_engine.analyse(aux_board_2)


        for mv, score in analysis_results_2['scores']:
            if mv == utils.chess.Move.from_uci(move_St):
                centipawns_St = score.score()
                break


    # Si move_St es None, centipawns_St se mantiene en None
    if which_function == TAYLOR_FUNCTION:
        return (
            'N' if centipawns_St is None else utils.centipawns_to_win_probability(centipawns_St),
            'N' if centipawns_LST is None else utils.centipawns_to_win_probability(centipawns_LST),
            centipawns_St,
            centipawns_LST
        )

    if which_function == TANG_FUNCTION:
        return (
            'N' if centipawns_St is None else centipawns_to_win_probability_tang(centipawns_St.score()),
            'N' if centipawns_LST is None else centipawns_to_win_probability_tang(centipawns_LST.score()),
            centipawns_St,
            centipawns_LST
        )

def computeMetricsAccordLST(move, move_St, board, LST_engine, which_function):
    aux_board_1 = board.copy()

    # Evaluamos la jugada de LST
#    aux_board_1.push(utils.chess.Move.from_uci(move))
    sorted_log_probs = LST_engine.analyse(aux_board_1)['log_probs']
    sorted_legal_moves = aux_engine.get_ordered_legal_moves(board)
    centipawns_LST = None
    centipawns_St = None

    # Extraer el valor de centipawns para el movimiento de LST
    if utils.chess.Move.from_uci(move) in sorted_legal_moves:
        idx = sorted_legal_moves.index(chess.Move.from_uci(move))
        log_prob = sorted_log_probs[idx]
        max_log_prob = np.max(log_prob)
        max_prob_LST = np.exp(max_log_prob)

    # Si move_St no es None, evaluamos también el movimiento de Stockfish
    if move_St is not None:
        aux_board_2 = board.copy()
        #aux_board_2.push(utils.chess.Move.from_uci(move_St))
        sorted_log_probs = LST_engine.analyse(aux_board_2)['log_probs']
        sorted_legal_moves = aux_engine.get_ordered_legal_moves(board)

        if utils.chess.Move.from_uci(move_St) in sorted_legal_moves:
            idx = sorted_legal_moves.index(chess.Move.from_uci(move_St))
            log_prob = sorted_log_probs[idx]
            max_log_prob = np.max(log_prob)
            max_prob_St = np.exp(max_log_prob)


    # Si move_St es None, centipawns_St se mantiene en None
    if which_function == TAYLOR_FUNCTION:
        return (
            max_prob_St,
            max_prob_LST,
            'N' if max_prob_St is None else utils.win_probability_to_centipawns(max_prob_St),
            'N' if max_prob_LST is None else utils.win_probability_to_centipawns(max_prob_LST)
        )

    if which_function == TANG_FUNCTION:
        return (
            max_prob_St,
            max_prob_LST,
            'N' if max_prob_St is None else win_probability_to_centipawns_tang(max_prob_St),
            'N' if max_prob_LST is None else win_probability_to_centipawns_tang(max_prob_LST)
        )