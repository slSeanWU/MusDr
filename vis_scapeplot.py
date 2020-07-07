from scipy.io import loadmat
from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize_scapeplot(mat_file, fig_out_dir):
  plt.clf()

  mat_dict = loadmat(mat_file)
  f_mat = mat_dict['fitness_info'][0, 0][0]
  f_mat[ np.isnan(f_mat) ] = 0.0

  for slen in range(f_mat.shape[0]):
    f_mat[slen] = np.roll(f_mat[slen], slen // 2)

  ax = plt.gca()
  ax.set_aspect(1)
  im = ax.imshow(f_mat, vmin=0.0, vmax=0.4, cmap='Greys')
  ax.set_ylim(ax.get_ylim()[::-1])
  plt.title('Fitness Scapeplot')
  plt.xlabel('Segment Center (in 2 Hz Frames)')
  plt.ylabel('Segment Length (in 2 Hz Frames)')

  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="3%", pad=0.15)
  plt.colorbar(im, cax=cax)
  
  out_figfile = os.path.join(
    fig_out_dir,
    mat_file.replace('\\', '/').split('/')[-1].replace('.mat', '.png')
  )
  plt.savefig(out_figfile)


def visualize_scapeplots_dir(scplot_dir, fig_out_dir):
  if not os.path.exists(fig_out_dir):
    os.makedirs(fig_out_dir)
  mat_files = [
    x for x in os.listdir(scplot_dir) if os.path.splitext(x)[-1] == '.mat'
  ]
  print (mat_files)

  for mf in mat_files:
    visualize_scapeplot(os.path.join(scplot_dir, mf), fig_out_dir)


if __name__ == "__main__":
  # command-line arguments
  parser = ArgumentParser(
    description='''
      Visualizes and saves the scapeplots computed by ``run_matlab_scapeplot.py``.
    '''
  )
  parser.add_argument(
    '-p', '--scplot_dir',
    required=True, type=str, help='directory containing the computed scape plots (.mat files)'
  )
  parser.add_argument(
    '-f', '--fig_out_dir',
    required=True, type=str, help='output directory for scape plot images'
  )
  args = parser.parse_args()

  visualize_scapeplots_dir(args.scplot_dir, args.fig_out_dir)