import utils
import os

TANG_FUNCTION = "tg"
TAYLOR_FUNCTION = "taylor"
OUTPUT_FILENAME = "probabilities.csv"

def centipawns_to_win_probability_tang(centipawns):
    return 111.714640912*utils.np.tan(1.5620688421*centipawns)

def createProbFile():
    if not os.path.isfile(OUTPUT_FILENAME):
        with open(OUTPUT_FILENAME,'w') as file:
            file.write('id,probabilities\n')

def writeProbabilities(num_puzzle,probs):
    with open(OUTPUT_FILENAME,'a') as file:
        file.write(f"{num_puzzle},{probs}\n")

def computeProbabilities(move,board,stockfish_engine,which_function):
    aux_board = board.copy()
    aux_board.push(utils.chess.Move.from_uci(move))
    analysis_results = stockfish_engine.analyse(aux_board)

    centipawns = None
    for mv,score in analysis_results['scores']:
        # If do we have a mate? Probability 1 or -1 depending on colour
        # We should implement it
        if mv==move:
            centipawns = score.score()
        break

    if which_function==TAYLOR_FUNCTION:
        return utils.centipawns_to_win_probability(centipawns)

    if which_function==TANG_FUNCTION:
        return centipawns_to_win_probability_tang(centipawns)