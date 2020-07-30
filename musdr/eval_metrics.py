import numpy as np
from glob import glob
import random, itertools

from side_utils import (
  get_event_seq, 
  get_bars_crop, 
  get_pitch_histogram, 
  compute_histogram_entropy, 
  get_onset_xor_distance,
  get_chord_sequence,
  read_fitness_mat
)


'''
Default event encodings (ones used by the Jazz Transformer).
You may override the defaults in function arguments to suit your own vocabulary.
'''
BAR_EV = 192               # the ID of ``Bar`` event
POS_EVS = range(193, 257)  # the IDs of ``Position`` events
CHORD_EVS = {              # the IDs of Chord-related events
  'Chord-Tone': range(322, 334),
  'Chord-Type': range(346, 393),
  'Chord-Slash': range(334, 346)
}

def compute_piece_pitch_entropy(piece_ev_seq, window_size, bar_ev_id=BAR_EV, pitch_evs=range(128), verbose=False):
  '''
  Computes the average pitch-class histogram entropy of a piece.
  (Metric ``H``)

  Parameters:
    piece_ev_seq (list): a piece of music in event sequence representation.
    window_size (int): length of segment (in bars) involved in the calc. of entropy at once.
    bar_ev_id (int): encoding ID of the ``Bar`` event, vocabulary-dependent.
    pitch_evs (list): encoding IDs of ``Note-On`` events, should be sorted in increasing order by pitches.
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

def compute_piece_groove_similarity(piece_ev_seq, bar_ev_id=BAR_EV, pos_evs=POS_EVS, pitch_evs=range(128), max_pairs=1000):
  '''
  Computes the average grooving pattern similarity between all pairs of bars of a piece.
  (Metric ``GS``)

  Parameters:
    piece_ev_seq (list): a piece of music in event sequence representation.
    bar_ev_id (int): encoding ID of the ``Bar`` event, vocabulary-dependent.
    pos_evs (list): encoding IDs of ``Note-Position`` events, vocabulary-dependent.
    pitch_evs (list): encoding IDs of ``Note-On`` events, should be sorted in increasing order by pitches.
    max_pairs (int): maximum #(pairs) considered, to save computation overhead.

  Returns:
    float: 0~1, the average grooving pattern similarity of the input piece.
  '''
  # remove redundant ``Bar`` marker
  if piece_ev_seq[-1] == bar_ev_id:
    piece_ev_seq = piece_ev_seq[:-1]

  # get every single bar & compute indices of bar pairs
  n_bars = piece_ev_seq.count(bar_ev_id)
  bar_seqs = []
  for b in range(n_bars):
    bar_seqs.append( get_bars_crop(piece_ev_seq, b, b, bar_ev_id) )
  pairs = list( itertools.combinations(range(n_bars), 2) )
  if len(pairs) > max_pairs:
    pairs = random.sample(pairs, max_pairs)

  # compute pairwise grooving similarities
  grv_sims = []
  for p in pairs:
    grv_sims.append(
      1. - get_onset_xor_distance(bar_seqs[p[0]], bar_seqs[p[1]], bar_ev_id, pos_evs, pitch_evs=pitch_evs)
    )

  return np.mean(grv_sims)


def compute_piece_chord_progression_irregularity(piece_ev_seq, chord_evs=CHORD_EVS, ngram=3):
  '''
  Computes the chord progression irregularity of a piece.
  (Metric ``CPI``)

  Parameters:
    piece_ev_seq (list): a piece of music in event sequence representation.
    chord_evs (dict of lists): [key] type of chord-related event --> [value] encodings belonging to the type.
    ngram (int): the n-gram in chord progression considered (e.g., bigram, trigram, 4-gram ...), defaults to trigram.

  Returns:
    float: 0~1, the chord progression irregularity of the input piece, measured on the n-gram specified.
  '''
  chord_seq = get_chord_sequence(piece_ev_seq, chord_evs)
  if len(chord_seq) <= ngram:
    return 1.

  num_ngrams = len(chord_seq) - ngram
  unique_set = set()
  for i in range(num_ngrams):
    str_repr = '_'.join([ '-'.join(str(x)) for x in chord_seq[i : i + ngram]])
    if str_repr not in unique_set:
      unique_set.add( str_repr )

  return len(unique_set) / num_ngrams


def compute_structure_indicator(mat_file, low_bound_sec=0, upp_bound_sec=128, sample_rate=2):
  '''
  Computes the structureness indicator SI(low_bound_sec, upp_bound_sec) from fitness scape plot (stored in a MATLAB .mat file).
  (Metric ``SI``)

  Parameters:
    mat_file (str): path to the .mat file containing fitness scape plot of a piece. (computed by ``run_matlab_scapeplot.py``).
    low_bound_sec (int, >0): the smallest timescale (in seconds) you are interested to examine.
    upp_bound_sec (int, >0): the largest timescale (in seconds) you are interested to examine.
    sample_rate (int): sample rate (in Hz) of the input fitness scape plot.

  Returns:
    float: 0~1, the structureness indicator (i.e., max fitness value) of the piece within the given range of timescales.
  '''
  assert low_bound_sec > 0 and upp_bound_sec > 0, '`low_bound_sec` and `upp_bound_sec` should be positive, got: low_bound_sec={}, upp_bound_sec={}.'.format(low_bound_sec, upp_bound_sec)
  low_bound_ts = int(low_bound_sec * sample_rate) - 1
  upp_bound_ts = int(upp_bound_sec * sample_rate)
  f_mat = read_fitness_mat(mat_file)

  if low_bound_ts >= f_mat.shape[0]:
    score = 0
  else:
    score = np.max(f_mat[ low_bound_ts : upp_bound_ts ])

  return score

if __name__ == "__main__":
  # codes below are for testing
  test_pieces = sorted( glob('./testdata/symbolic/*.csv') )
  test_pieces_scplot = sorted( glob('./testdata/scplot/*.fit.mat') )
  print (test_pieces, test_pieces_scplot)

  for p, p_sc in zip(test_pieces, test_pieces_scplot):
    print ('>> now processing: {}'.format(p))
    seq = get_event_seq(p)
    print ('  1-bar H: {:.3f}'.format(compute_piece_pitch_entropy(seq, 1)))
    print ('  4-bar H: {:.3f}'.format(compute_piece_pitch_entropy(seq, 4)))
    print ('  GS: {:.4f}'.format(compute_piece_groove_similarity(seq, BAR_EV)))
    print ('  CPI: {:.4f}'.format(compute_piece_chord_progression_irregularity(seq)))
    print ('  SI(3, 8): {:.4f}'.format(compute_structure_indicator(p_sc, 3, 8)))
    print ('  SI(8, 15): {:.4f}'.format(compute_structure_indicator(p_sc, 8, 15)))
    print ('  SI(15, ): {:.4f}'.format(compute_structure_indicator(p_sc, 15)))
    print ('==========================')