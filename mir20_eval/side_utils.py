import scipy.stats
import numpy as np
import pandas as pd

N_PITCH_CLS = 12 # {C, C#, ..., Bb, B}

def get_event_seq(piece_csv, seq_col_name='ENCODING'):
  '''
  Extracts the event sequence from a piece of music (stored in .csv file).
  NOTE: You should modify this function if you use different formats.

  Parameters:
    piece_csv (str): path to the piece's .csv file.
    seq_col_name (str): name of the column containing event encodings.

  Returns:
    list: the event sequence of the piece.
  '''
  df = pd.read_csv(piece_csv, encoding='utf-8')
  return df[seq_col_name].astype('int32').tolist()

def compute_histogram_entropy(hist):
  ''' 
  Computes the entropy (log base 2) of a normalised histogram.

  Parameters:
    hist (ndarray): input pitch (or duration) histogram, should be normalised.

  Returns:
    float: entropy (log base 2) of the histogram.
  '''
  return scipy.stats.entropy(hist) / np.log(2)

def get_pitch_histogram(ev_seq, pitch_evs=range(128), verbose=False):
  '''
  Computes the pitch-class histogram from an event sequence.

  Parameters:
    ev_seq (list): a piece of music in event sequence representation.
    pitch_evs (list): encoding IDs of ``Note-On`` events.
    verbose (bool): whether to print msg. when ev_seq has no notes.

  Returns:
    ndarray: the resulting pitch-class histogram.
  '''
  ev_seq = [x for x in ev_seq if x in pitch_evs]

  if not len(ev_seq):
    if verbose:
      print ('[Info] The sequence contains no notes.')
    return None

  # compress sequence to pitch classes & get normalised counts
  ev_seq = pd.Series(ev_seq) % N_PITCH_CLS
  ev_hist = ev_seq.value_counts(normalize=True)

  # make the final histogram
  hist = np.zeros( (N_PITCH_CLS,) )
  for i in range(N_PITCH_CLS):
    if i in ev_hist.index:
      hist[i] = ev_hist.loc[i]

  return hist

def get_bars_crop(ev_seq, start_bar, end_bar, bar_ev_id, verbose=False):
  '''
  Returns the designated crop (bars) of the input piece.

  Parameter:
    ev_seq (list): a piece of music in event sequence representation.
    start_bar (int): the starting bar of the crop.
    end_bar (int): the ending bar (inclusive) of the crop.
    bar_ev_id (int): encoding ID of the ``Bar`` event, vocabulary-dependent.
    verbose (bool): whether to print messages when unexpected operations happen.

  Returns:
    list: a cropped segment of music consisting of (end_bar - start_bar + 1) bars.
  '''
  if start_bar < 0 or end_bar < 0:
    raise ValueError('Invalid start_bar: {}, or end_bar: {}.'.format(start_bar, end_bar))

  # get the indices of ``Bar`` events
  ev_seq = np.array(ev_seq)
  bar_markers = np.where(ev_seq == bar_ev_id)[0]

  if start_bar > len(bar_markers) - 1:
    raise ValueError('start_bar: {} beyond end of piece.'.format(start_bar))

  if end_bar < len(bar_markers) - 1:
    cropped_seq = ev_seq[ bar_markers[start_bar] : bar_markers[end_bar + 1] ]
  else:
    if verbose:
      print (
        '[Info] end_bar: {} beyond or equal the end of the input piece; only the last {} bars are returned.'.format(
          end_bar, len(bar_markers) - start_bar
        ))
    cropped_seq = ev_seq[ bar_markers[start_bar] : ]

  return cropped_seq