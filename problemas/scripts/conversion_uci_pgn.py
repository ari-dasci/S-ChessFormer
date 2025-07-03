import chess

fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Posición inicial
board = chess.Board(fen)

# Jugada en PNG (ejemplo: "e4")
move_png = "e4"

# Convertir a objeto `Move`
move = board.parse_san(move_png)

# Convertir a UCI
move_uci = move.uci()
print("PNG:", move_png, "-> UCI:", move_uci)

# Convertir a PNG
# Jugada en UCI (ejemplo: "e2e4")
move_uci = "e2e4"

# Crear objeto `Move`
move = chess.Move.from_uci(move_uci)

# Convertir a PNG
move_png = board.san(move)
print("UCI:", move_uci, "-> PNG:", move_png)

