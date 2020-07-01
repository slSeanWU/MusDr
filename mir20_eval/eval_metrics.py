import numpy as np
from glob import glob

from side_utils import get_event_seq, get_bars_crop, get_pitch_histogram, compute_histogram_entropy

# ``Bar`` event for Jazz Transformer, should be changed according to vocab
JAZZ_TRSFMR_BAR_EV = 192  

def compute_piece_pitch_entropy(piece_ev_seq, window_size, bar_ev_id=JAZZ_TRSFMR_BAR_EV, pitch_evs=range(128), verbose=False):
  '''
  Computes the averaged pitch-class histogram entropy of a piece.

  Parameters:
    piece_ev_seq (list): a piece of music in event sequence representation.
    window_size (int): length of segment (in bars) involved in the calc. of entropy at once.
    bar_ev_id (int): encoding ID of the ``Bar`` event, vocabulary-dependent.
    pitch_evs (list): encoding IDs of ``Note-On`` events.
    verbose (bool): whether to print msg. when a crop contains no notes.

  Returns:
    float: the average n-bar pitch-class histogram entropy of the input piece.
  '''
  # remove redundant ``Bar`` marker
  if piece_ev_seq[-1] == bar_ev_id:
    piece_ev_seq = piece_ev_seq[:-1]

  n_bars = piece_ev_seq.count(bar_ev_id)
  if window_size > n_bars:
    print ('[Warning] window_size: {} too large for the piece, falling back to #(bars) of the piece.'.format(window_size))
    window_size = n_bars

  # compute entropy of all possible segments
  pitch_ents = []
  for st_bar in range(0, n_bars - window_size + 1):
    seg_ev_seq = get_bars_crop(piece_ev_seq, st_bar, st_bar + window_size - 1, bar_ev_id)

    pitch_hist = get_pitch_histogram(seg_ev_seq, pitch_evs=pitch_evs)
    if pitch_hist is None:
      if verbose:
        print ('[Info] No notes in this crop: {}~{} bars.'.format(st_bar, st_bar + window_size - 1))
      continue

    pitch_ents.append( compute_histogram_entropy(pitch_hist) )

  return np.mean(pitch_ents)

if __name__ == "__main__":
  # codes below are for testing
  test_pieces = sorted( glob('./testdata/*.csv') )
  print (test_pieces)

  for p in test_pieces:
    print ('>> now processing: {}'.format(p))
    seq = get_event_seq(p)
    print ('  1-bar: {:.3f}'.format(compute_piece_pitch_entropy(seq, 1)))
    print ('  4-bar: {:.3f}'.format(compute_piece_pitch_entropy(seq, 4)))