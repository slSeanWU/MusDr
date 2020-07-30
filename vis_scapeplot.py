from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from musdr.side_utils import read_fitness_mat

def visualize_scapeplot(mat_file, fig_out_dir):
  plt.clf()
  f_mat = read_fitness_mat(mat_file)
  ext = os.path.splitext(mat_file)[-1].lower()
  
  ax = plt.gca()
  ax.set_aspect(1)
  im = ax.imshow(f_mat, vmin=0.0, vmax=0.5, cmap='hot_r')
  ax.set_ylim(ax.get_ylim()[::-1])
  plt.title('Fitness Scapeplot')
  plt.xlabel('Segment Center (in 2 Hz Frames)')
  plt.ylabel('Segment Length (in 2 Hz Frames)')

  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="3%", pad=0.15)
  plt.colorbar(im, cax=cax)
  
  out_figfile = os.path.join(
    fig_out_dir,
    mat_file.replace('\\', '/').split('/')[-1].replace(ext, '.png')
  )
  plt.savefig(out_figfile)


def visualize_scapeplots_dir(scplot_dir, fig_out_dir):
  if not os.path.exists(fig_out_dir):
    os.makedirs(fig_out_dir)
  mat_files = [
    x for x in os.listdir(scplot_dir) if os.path.splitext(x)[-1] in ['.npy', '.mat']
  ]
  print (mat_files)

  for mf in mat_files:
    visualize_scapeplot(os.path.join(scplot_dir, mf), fig_out_dir)


if __name__ == "__main__":
  # command-line arguments
  parser = ArgumentParser(
    description='''
      Visualizes and saves the scape plot images.
    '''
  )
  parser.add_argument(
    '-p', '--scplot_dir',
    required=True, type=str, help='directory containing the computed scape plots (.npy or .mat files)'
  )
  parser.add_argument(
    '-f', '--fig_out_dir',
    required=True, type=str, help='output directory for scape plot images'
  )
  args = parser.parse_args()

  visualize_scapeplots_dir(args.scplot_dir, args.fig_out_dir)