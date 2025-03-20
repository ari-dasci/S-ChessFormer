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

"""Implements a PyGrain DataLoader for chess data."""

import abc
import os

import grain.python as pygrain
import jax
import numpy as np
import pandas as pd

from searchless_chess.src import bagz
from searchless_chess.src import config as config_lib
from searchless_chess.src import constants
from searchless_chess.src import tokenizer
from searchless_chess.src import utils
from searchless_chess.src.engines import engine as engine_lib

import chess


def _process_fen(fen: str) -> np.ndarray:
  return tokenizer.tokenize(fen).astype(np.int32)


def _process_move(move: str) -> np.ndarray:
  return np.asarray([utils.MOVE_TO_ACTION[move]], dtype=np.int32)


def _process_win_prob(
    win_prob: float,
    return_buckets_edges: np.ndarray,
) -> np.ndarray:
  return utils.compute_return_buckets_from_returns(
      returns=np.asarray([win_prob]),
      bins_edges=return_buckets_edges,
  )


class ConvertToSequence(pygrain.MapTransform, abc.ABC):
  """Base class for converting chess data to a sequence of integers."""

  def __init__(self, num_return_buckets: int) -> None:
    super().__init__()
    self._return_buckets_edges, _ = utils.get_uniform_buckets_edges_values(
        num_return_buckets,
    )
    # The loss mask ensures that we only train on the return bucket.
    self._loss_mask = np.full(
        shape=(self._sequence_length,),
        fill_value=True,
        dtype=bool,
    )
    self._loss_mask[-1] = False

  @property
  @abc.abstractmethod
  def _sequence_length(self) -> int:
    raise NotImplementedError()


class ConvertBehavioralCloningDataToSequence(ConvertToSequence):
  """Converts the fen, move, and win probability into a sequence of integers."""

  @property
  def _sequence_length(self) -> int:
    return tokenizer.SEQUENCE_LENGTH + 1  # (s) + (a)

  def map(
      self, element: bytes
  ) -> tuple[constants.Sequences, constants.LossMask]:
    fen, move = constants.CODERS['behavioral_cloning'].decode(element)
    state = _process_fen(fen)
    action = _process_move(move)
    sequence = np.concatenate([state, action])
    return sequence, self._loss_mask


class ConvertStateValueDataToSequence(ConvertToSequence):
  """Converts the fen, move, and win probability into a sequence of integers."""

  @property
  def _sequence_length(self) -> int:
    return tokenizer.SEQUENCE_LENGTH + 1  # (s) +  (r)

  def map(
      self, element: bytes
  ) -> tuple[constants.Sequences, constants.LossMask]:
    fen, win_prob = constants.CODERS['state_value'].decode(element)
    state = _process_fen(fen)
    return_bucket = _process_win_prob(win_prob, self._return_buckets_edges)
    sequence = np.concatenate([state, return_bucket])
    return sequence, self._loss_mask


class ConvertActionValueDataToSequence(ConvertToSequence):
  """Converts the fen, move, and win probability into a sequence of integers."""

  @property
  def _sequence_length(self) -> int:
    return tokenizer.SEQUENCE_LENGTH + 2  # (s) + (a) + (r)

  def map(
      self, element: bytes
  ) -> tuple[constants.Sequences, constants.LossMask]:
    fen, move, win_prob = constants.CODERS['action_value'].decode(element)
    state = _process_fen(fen)
    action = _process_move(move)
    return_bucket = _process_win_prob(win_prob, self._return_buckets_edges)
    sequence = np.concatenate([state, action, return_bucket])
    return sequence, self._loss_mask


class ConvertActionValueDataCSVToSequence(ConvertToSequence):
    """Convierte los datos del CSV en secuencias de enteros para el modelo."""
    @property
    def _sequence_length(self) -> int:
        return tokenizer.SEQUENCE_LENGTH + 2  # (s) + (a) + (r)

    def map(self, row):
        """Convierte una fila del CSV en una secuencia."""
        fen = row['FEN']
        move = row['Move']
        win_prob = 0 #row['%win_Stockfish_best_move']
        state = _process_fen(fen)
        action = _process_move(move)
        return_bucket = _process_win_prob(win_prob, self._return_buckets_edges)

        sequence = np.concatenate([state, action, return_bucket])
        return sequence, self._loss_mask, fen


_TRANSFORMATION_BY_POLICY = {
    'behavioral_cloning': ConvertBehavioralCloningDataToSequence,
    'action_value': ConvertActionValueDataToSequence,
    'state_value': ConvertStateValueDataToSequence,
    'action_value_csv': ConvertActionValueDataCSVToSequence,
}


# Follows the base_constants.DataLoaderBuilder protocol.
def build_data_loader(config: config_lib.DataConfig) -> pygrain.DataLoader:
  """Returns a data loader for chess from the config."""
  data_source = bagz.BagDataSource(
      os.path.join(
          os.getcwd(),
          f'../data/{config.split}/{config.policy}_data.bag',
      ),
  )

  if config.num_records is not None:
    num_records = config.num_records
    if len(data_source) < num_records:
      raise ValueError(
          f'[Process {jax.process_index()}]: The number of records requested'
          f' ({num_records}) is larger than the dataset ({len(data_source)}).'
      )
  else:
    num_records = len(data_source)

  sampler = pygrain.IndexSampler(
      num_records=num_records,
      shard_options=pygrain.NoSharding(),
      shuffle=config.shuffle,
      num_epochs=1,
      seed=config.seed,
  )
  transformations = (
      _TRANSFORMATION_BY_POLICY[config.policy](
          num_return_buckets=config.num_return_buckets
      ),
      pygrain.Batch(config.batch_size, drop_remainder=True),
  )
  return pygrain.DataLoader(
      data_source=data_source,
      sampler=sampler,
      operations=transformations,
      worker_count=config.worker_count,
      read_options=None,
  )
class CSVDataset(pygrain.RandomAccessDataSource):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, index):
        return self.dataframe.loc[index]

    def __len__(self):
        return len(self.dataframe)
      
def build_data_loader_csv(config: config_lib.DataConfig) -> pygrain.DataLoader:
    """Crea un DataLoader que itera sobre los pares Tablero-Mejor Movimiento desde un archivo CSV."""
    csv_path = config.data_path
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Archivo CSV no encontrado: {csv_path}")

    df = pd.read_csv(csv_path)

    if 'FEN' not in df.columns or 'Moves_UCI' not in df.columns:
        raise ValueError("El CSV debe contener las columnas 'FEN', 'Moves_UCI'.")
    df = df[["FEN", "Moves_UCI"]]#, "%win_Stockfish_best_move"]]
    
    df_dataset = CSVDataset(df)
    
    data_source = pygrain.MapDataset.source(df_dataset)

    sampler = pygrain.IndexSampler(
        num_records=len(df),
        shard_options=pygrain.NoSharding(),
        shuffle=config.shuffle,
        num_epochs=1,
        seed=config.seed,
    )

    transformations = (
        ConvertActionValueDataCSVToSequence(config.num_return_buckets),
        pygrain.Batch(config.batch_size, drop_remainder=True),
    )

    return pygrain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=transformations,
        worker_count=config.worker_count,
        read_options=None,
    )
    
    
def build_data_loader_csv_all_moves(config: config_lib.DataConfig) -> pygrain.DataLoader:
    """Crea un DataLoader que itera sobre los Tableros y todas las jugadas posibles aplicables desde un archivo CSV."""
    csv_path = config.data_path
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Archivo CSV no encontrado: {csv_path}")

    df = pd.read_csv(csv_path)

    if 'FEN' not in df.columns or 'Moves_UCI' not in df.columns:
        raise ValueError("El CSV debe contener las columnas 'FEN', 'Moves_UCI'.")
      
    # Creamos el nuevo df con todas las jugadas posibles
    data = []
    
    for idx, row in df.iterrows():
      fen = row["FEN"]
      board = chess.Board(fen)  # Cargar el tablero a partir del FEN
      
      # Obtener los movimientos legales ordenados
      legal_moves = engine_lib.get_ordered_legal_moves(board)

      # Agregar cada movimiento posible a la lista
      for move in legal_moves:
        data.append({"ID": idx, "FEN": fen, "Move": move.uci()})  # move.uci() convierte el movimiento a notación estándar
    
    # Obtenemos el dataframe final sobre el que iterar
    df_full = pd.DataFrame(data)
    #print(df_full)
    df_dataset = CSVDataset(df_full)
    
    data_source = pygrain.MapDataset.source(df_dataset)

    sampler = pygrain.IndexSampler(
        num_records=len(df_full)+config.batch_size -1,
        shard_options=pygrain.NoSharding(),
        shuffle=config.shuffle,
        num_epochs=1,
        seed=config.seed,
    )

    transformations = (
        ConvertActionValueDataCSVToSequence(config.num_return_buckets),
        pygrain.Batch(config.batch_size, drop_remainder=True),
    )

    return pygrain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=transformations,
        worker_count=config.worker_count,
        read_options=None,
        #repeat=False,
    )

if __name__ == '__main__':
  """
  # Pruebas con archivos .bag
  file_name = '../data/test/action_value_data.bag'
  data_source = bagz.BagDataSource(file_name)
  #print(data_source)
  #print(type(data_source))
  #print(len(data_source))
  #print(data_source[0])
  
  
  fen, move, win_prob = constants.CODERS['action_value'].decode(data_source[0])
  #print('FEN', fen)
  #print('move', move)
  #print('win_prob', win_prob)
  state = _process_fen(fen)
  action = _process_move(move)
  buckets_fn, _ = utils.get_uniform_buckets_edges_values(128)
  return_bucket = _process_win_prob(
    win_prob, 
    buckets_fn
  )
  
  #print(state)
  #print(action)
  #print(return_bucket)
  #print(buckets_fn)
  
  map_data = ConvertActionValueDataToSequence(128).map(data_source[0])
  #print(map_data)
  """
  
  data_loader = build_data_loader(config_lib.DataConfig(
      split='test',
      policy='action_value',
      num_return_buckets=128,
      batch_size=32,
      shuffle=False,
      worker_count=1,
      seed=0,
  ))
  
  data_iter = data_loader.__iter__()
  
  """
  # Pruebas con archivos .csv
  file_name = '../../problemas/solved_puzzles/CBP_HARD_PREVIEW_solved_9M_prof20.csv'
  df = pd.read_csv(file_name)
  #print(df.head())
  
  map_data = ConvertActionValueDataCSVToSequence(128).map(df.iloc[0])
  #print(map_data)
  data_config = config_lib.DataConfig(
          batch_size=256,
          shuffle=False,
          worker_count=0,  # 0 disables multiprocessing.
          num_return_buckets=128,
          policy='action_value',
          split='test',
      )
  build_data_loader_csv(data_config)
  
  """
  