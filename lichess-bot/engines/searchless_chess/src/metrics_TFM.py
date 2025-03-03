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

TANG_FUNCTION = "tg"
TAYLOR_FUNCTION = "taylor"
OUTPUT_FILENAME = "metrics.csv"

def centipawns_to_win_probability_tang(centipawns):
    return 111.714640912*np.tan(1.5620688421*centipawns)

def createCSVFile():
    if not os.path.isfile(OUTPUT_FILENAME):
        with open(OUTPUT_FILENAME,'w') as file:
            file.write('id,best_movs_St,best_movs_LST,probs_St_accord_St,probs_LST_accord_St,probs_St_accord_LST,probs_LST_accord_LST,diffs_probs_accord_St,diffs_probs_accord_LST,centipawns_St_accord_St,centipawns_LST_accord_St,centipawns_St_accord_LST,centipawns_LST_accord_LST,diffs_centipawns_accord_St,diffs_centipawns_accord_LST,diffs_centipawns_accord_St,diffs_accord_centipawns_LST\n')

def writeWithinCSV(num_puzzle,best_movs_St,best_movs_LST,probs_St_accord_St,probs_LST_accord_St,probs_St_accord_LST,probs_LST_accord_LST,diffs_probs_accord_St,diffs_probs_accord_LST,centipawns_St_accord_St,centipawns_LST_accord_St,centipawns_St_accord_LST,centipawns_LST_accord_LST,diffs_centipawns_accord_St,diffs_centipawns_accord_LST):
    with open(OUTPUT_FILENAME,'a') as file:
        file.write(f"{num_puzzle},{best_movs_St},{best_movs_LST},{probs_St_accord_St},{probs_LST_accord_St},{probs_St_accord_LST},{probs_LST_accord_LST},{diffs_probs_accord_St},{diffs_probs_accord_LST},{centipawns_St_accord_St},{centipawns_LST_accord_St},{centipawns_St_accord_LST},{centipawns_LST_accord_LST},{diffs_centipawns_accord_St},{diffs_centipawns_accord_LST},{diffs_centipawns_accord_St},{diffs_centipawns_accord_LST}\n")

def computeMetricsAccordStockfish(move,move_St,board,stockfish_engine,which_function):
    aux_board_1 = board.copy()
    aux_board_2 = board.copy()

    # LST move accord Stockfish
    aux_board_1.push(utils.chess.Move.from_uci(move))
    analysis_results_1 = stockfish_engine.analyse(aux_board_1)
    # Stockfish move accord Stockfish
    aux_board_2.push(utils.chess.Move.from_uci(move_St))
    analysis_results_2 = utils.chess.Move.from_uci(aux_board_2)

    centipawns_LST = None
    centipawns_St = None
    for mv,score in analysis_results_1['scores']:
        # If do we have a mate? Probability 1 or -1 depending on colour
        # We should implement it
        if mv==move:
            centipawns_LST = score.score()
            break

    for mv,score in analysis_results_2['scores']:
        # If do we have a mate? Probability 1 or -1 depending on colour
        # We should implement it
        if mv==move_St:
            centipawns_St = score.score()
            break        

    if which_function==TAYLOR_FUNCTION:
        return utils.centipawns_to_win_probability(centipawns_St),utils.centipawns_to_win_probability(centipawns_LST),centipawns_St,centipawns_LST

    if which_function==TANG_FUNCTION:
        return centipawns_to_win_probability_tang(centipawns_St),centipawns_to_win_probability_tang(centipawns_LST),centipawns_St,centipawns_LST
    


def computeMetricsAccordLST(move,move_St,board,LST_engine,which_function):
    aux_board_1 = board.copy()
    aux_board_2 = board.copy()

    # LST move accord Stockfish
    aux_board_1.push(utils.chess.Move.from_uci(move))
    analysis_results_1 = LST_engine.analyse(aux_board_1)
    # Stockfish move accord Stockfish
    aux_board_2.push(utils.chess.Move.from_uci(move_St))
    analysis_results_2 = utils.chess.Move.from_uci(aux_board_2)

    centipawns_LST = None
    centipawns_St = None
    for mv,score in analysis_results_1['scores']:
        # If do we have a mate? Probability 1 or -1 depending on colour
        # We should implement it
        if mv==move:
            centipawns_LST = score.score()
            break

    for mv,score in analysis_results_2['scores']:
        # If do we have a mate? Probability 1 or -1 depending on colour
        # We should implement it
        if mv==move_St:
            centipawns_St = score.score()
            break        

    if which_function==TAYLOR_FUNCTION:
        return utils.centipawns_to_win_probability(centipawns_St),utils.centipawns_to_win_probability(centipawns_LST),centipawns_St,centipawns_LST

    if which_function==TANG_FUNCTION:
        return centipawns_to_win_probability_tang(centipawns_St),centipawns_to_win_probability_tang(centipawns_LST),centipawns_St,centipawns_LST