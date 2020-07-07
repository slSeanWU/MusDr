import os, time
import subprocess
from multiprocessing import Pool

from argparse import ArgumentParser

def run_matlab(idx, audio_file, adir, sdir, fdir, skip_existing=True):
  start = time.time()
  ext = os.path.splitext(audio_file)[-1]
  if skip_existing and os.path.exists( os.path.join(fdir, audio_file.replace(ext, '.fit.mat')) ):
    print ('>> file no. {:02d} exists, skipping ...'.format(idx))
  else:
    print ('>> now processing no. {:02d}: {}'.format(idx, audio_file))
    subprocess.run('''matlab -wait -nosplash -nodesktop -r "sm_compute_scapeplot('{}', '{}', '{}', '{}'); quit()"'''.format(adir, audio_file, fdir, sdir))
    print ('[finished] no. {:02d} in {:.3f} secs'.format(idx, time.time()-start))

  return

if __name__ == "__main__":
  # command-line arguments
  parser = ArgumentParser(
    description='''
      Computes and stores the 1) self-similarity matrix (SSM); and, 2) fitness scape plots of all audio files under the given directory.
      NOTE: This script invokes a MATLAB script & SM Toolbox functions, you should have them installed beforehand.
    '''
  )
  parser.add_argument(
    '-a', '--audio_dir',
    required=True, type=str, help='''
      path to the directory containing audio files, note that all the filenames within the directory should be ASCII-compatible'. 
      Accepted format: mp3, wav.
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
    default=1, type=int, help='# of MATLAB processes that will be spawned'
  )
  parser.add_argument(
    '--skip_existing',
    default=True, type=bool, help='whether to skip files that already has their scape plots computed'
  )
  args = parser.parse_args()

  p = Pool(processes=args.num_workers)
  run_arglist = []
  fitness_dir, ssm_dir, audio_dir = args.scplot_out_dir, args.ssm_out_dir, args.audio_dir
  
  if fitness_dir[-1] != '/':
    fitness_dir += '/'
  if ssm_dir[-1] != '/':
    ssm_dir += '/'
  if audio_dir[-1] != '/':
    audio_dir += '/'

  if not os.path.exists(fitness_dir):
    os.makedirs(fitness_dir)

  if not os.path.exists(ssm_dir):
    os.makedirs(ssm_dir)

  audio_files = sorted( os.listdir(audio_dir) )
  print ('number of audios:', len(audio_files))

  for i, af in enumerate(audio_files):
    run_arglist.append([i, af, audio_dir, ssm_dir, fitness_dir, args.skip_existing])

  p.starmap(run_matlab, run_arglist)
    