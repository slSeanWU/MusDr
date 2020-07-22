import os, time
from multiprocessing import Pool
from argparse import ArgumentParser

import numpy as np

from mueller_audio_tools.scapeplot import (
  compute_fitness_scape_plot,
  normalization_properties_SSM
)
from mueller_audio_tools.ssm_features import (
  compute_SM_from_filename,
  compute_tempo_rel_set
)

'''
This script computes:
  1) self-similarity matrices; and
  2) fitness scape plots
from audio music.

Acknowledgement:
  The utilities used for SSM & scape plot computation (``muller_audio_tools``) were written by [Mueller et. al];
  and, are freely accessible via: audiolabs-erlangen.de/resources/MIR/FMP/C0/C0.html
'''

# parameters for SSM computation
tempo_rel_set = compute_tempo_rel_set(0.5, 2, 7) # for tempo invariance
shift_set = np.array([x for x in range(12)])     # for tranposition invariance
rel_threshold = 0.25                             # the proportion of (highest) values to retain
penalty = -2                                     # all values below ``rel_threshold`` are set to this


def compute_piece_ssm_scplot(idx, audio_dir, audio_file, ssm_out_dir, fitness_out_dir):
    print ('>> now processing file no. {} ...'.format(idx))
    time_st = time.time()
    full_af_path = os.path.join(audio_dir, audio_file)
    af_ext = os.path.splitext(audio_file)[-1]

    # compute & save self-similarity matrix (default resulting sample rate: 2Hz)
    _, _, _, _, S, _ = compute_SM_from_filename(
      full_af_path, 
      tempo_rel_set=tempo_rel_set, 
      shift_set=shift_set, 
      thresh=rel_threshold,
      penalty=penalty
    )
    S = normalization_properties_SSM(S)
    np.save(os.path.join(ssm_out_dir, audio_file.replace(af_ext, '_ssm.npy')), S)

    # compute & save fitness scape plot
    SP = compute_fitness_scape_plot(S)[0]
    np.save(os.path.join(fitness_out_dir, audio_file.replace(af_ext, '_fitness.npy')), SP)

    print ('[completed] file no. {}, shape of scape plot: {}, processing time: {:.2f} sec.'.format(
      idx, SP.shape, time.time() - time_st
    ))
    return

if __name__ == "__main__":
  # command-line arguments
  parser = ArgumentParser(
    description='''
      Computes and stores the 1) self-similarity matrix (SSM); and, 2) fitness scape plots of all audio files under a given directory.
    '''
  )
  parser.add_argument(
    '-a', '--audio_dir',
    required=True, type=str, help='''
      path to the directory containing audio files.
    '''
  )
  parser.add_argument(
    '-s', '--ssm_out_dir',
    required=True, type=str, help='output directory for computed SSMs'
  )
  parser.add_argument(
    '-p', '--scplot_out_dir',
    required=True, type=str, help='output directory for computed fitness scape plots'
  )
  parser.add_argument(
    '-j', '--num_workers',
    default=1, type=int, help='# of processes that will be spawned for SSM & scape plot computation'
  )
  args = parser.parse_args()
  fitness_dir, ssm_dir, audio_dir = args.scplot_out_dir, args.ssm_out_dir, args.audio_dir
  num_workers = args.num_workers

  # create output directories
  if not os.path.exists(fitness_dir):
    os.makedirs(fitness_dir)
  if not os.path.exists(ssm_dir):
    os.makedirs(ssm_dir)

  # get audio files
  audio_files = sorted( os.listdir(audio_dir) )
  print ('number of audios:', len(audio_files))

  # multi-processing
  p = Pool(processes=num_workers)
  run_arglist = []

  for i, af in enumerate(audio_files):
    run_arglist.append([i+1, audio_dir, af, ssm_dir, fitness_dir])

  p.starmap(compute_piece_ssm_scplot, run_arglist)