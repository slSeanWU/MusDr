function sm_compute_scapeplot(audio_dir, audio_file, output_fitness_dir, output_ssm_dir)

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Name: sm_compute_scapeplot.m
  % Date of Revision: 2020-05
  % Programmer: Nanzhu Jiang, Peter Grosche, Meinard M�ller, Shih-Lun Wu
  % http://www.audiolabs-erlangen.de/resources/MIR/SMtoolbox/
  %
  % Description: 
  %   This file computes a scape plot
  %   represention of a give music recording as described in: 
  %  
  %   Meinard M�ller, Nanzhu Jiang, Peter Grosche: 
  %   A Robust Fitness Measure for Capturing Repetitions in Music Recordings With Applications to Audio Thumbnailing. 
  %   IEEE Transactions on Audio, Speech & Language Processing 21(3): 531-543 (2013)
  %
  %   1. Loads a wav file and converts it into 22050 Hz, mono.
  %   2. Computes chroma features (CENS variant with a feature resolution 
  %      of 2 Hertz). The used functions are part of the Chroma Toolbox
  %      http://www.mpi-inf.mpg.de/resources/MIR/chromatoolbox/ 
  %   3. Computes and visualizes an enhanced and thresholded similarity 
  %      matrix. 
  %   4. Computes and saves a fitness scape plot.
  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % Reference: 
  %   If you use the 'SM toobox' please refer to:
  %   [MJG13] Meinard M�ller, Nanzhu Jiang, Harald Grohganz
  %   SM Toolbox: MATLAB Implementations for Computing and Enhancing Similarity Matrices
  %   Proceedings of the 53rd Audio Engineering Society Conference on Semantic Audio, London, 2014.
  %
  % License:
  %     This file is part of 'SM Toolbox'.
  % 
  %     'SM Toolbox' is free software: you can redistribute it and/or modify
  %     it under the terms of the GNU General Public License as published by
  %     the Free Software Foundation, either version 2 of the License, or
  %     (at your option) any later version.
  % 
  %     'SM Toolbox' is distributed in the hope that it will be useful,
  %     but WITHOUT ANY WARRANTY; without even the implied warranty of
  %     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  %     GNU General Public License for more details.
  % 
  %     You should have received a copy of the GNU General Public License
  %     along with 'SM Toolbox'. If not, see
  %     <http://www.gnu.org/licenses/>.
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  
  initPaths;
  
  %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %  1. Loads a wav file and converts it into 22050 Hz, mono
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % clear; close all;
  [pathstr,pathname,ext] = fileparts(audio_file);
  [f_audio,sideinfo] = wav_to_audio('', audio_dir, audio_file);
  
  %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %  2. Computes chroma features (CENS variant with a feature resolution 
  %      of 2 Hertz). The used functions are part of the Chroma Toolbox
  %      http://www.mpi-inf.mpg.de/resources/MIR/chromatoolbox/ 
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  paramPitch.winLenSTMSP = 4410;
  [f_pitch] = audio_to_pitch_via_FB(f_audio,paramPitch);
  paramCENS.winLenSmooth = 11;
  paramCENS.downsampSmooth = 5;
  [f_CENS] = pitch_to_CENS(f_pitch,paramCENS);
  
  
  %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %   3. Computes and visualizes an enhanced and thresholded similarity 
  %      matrix. 
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  paramSM.smoothLenSM = 20;
  paramSM.tempoRelMin = 0.5;
  paramSM.tempoRelMax = 2;
  paramSM.tempoNum = 7;
  paramSM.forwardBackward = 1;
  paramSM.circShift = [0:11];
  [S,I] = features_to_SM(f_CENS,f_CENS,paramSM);
  
  paramVis.colormapPreset = 2;
  visualizeSM(S,paramVis);
  title('S');
  
  visualizeTransIndex(I,paramVis);
  title('Transposition index');
  
  
  paramThres.threshTechnique = 2;
  paramThres.threshValue = 0.25;
  paramThres.applyBinarize = 0;
  paramThres.applyScale = 1;
  paramThres.penalty = -2;
  [S_final] = threshSM(S,paramThres);  
  
  paramVis.imagerange = [-2,1];
  paramVis.colormapPreset = 3;
  handleFigure = visualizeSM(S_final,paramVis);
  title('Final S with thresholding for computing the scapeplot matrix');
  
  ssm_filename = strrep(audio_file, ext, '.ssm.mat');
  save(strcat(output_ssm_dir, ssm_filename), 'S', 'S_final')
  
  %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %   4. Computes and saves a fitness scape plot.
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % compute fitness scape plot and save
  parameter.dirFitness = output_fitness_dir;
  parameter.saveFitness = 1;
  parameter.title = 'Scapeplot';
  parameter.fitFileName = strrep(audio_file, ext, '.fit');
  
  %-----------!!IMPORTANT!!--------------------------------------------------%
  % For fast computing of fitness scape plot, please enable parallel computing.
  % To enable that, use command 'matlabpool open'.
  % To disable that, use command 'matlabpool close'
  %--------------------------------------------------------------------------%
  [fitness_info,parameter] = SSM_to_scapePlotFitness(S_final, parameter);
  fitness_matrix = fitness_info.fitness;
  
  % % instead of computing fitness, you can load a previously computed scape plot:
  % fitnessSaveFileName = ['data_fitness/',filename(1:end-4),'_fit','.mat'];
  % fitnessFile = load(fitnessSaveFileName);
  % fitness_matrix = fitnessFile.fitness_info.fitness;
  
  paramVisScp = [];
  % paramVisScp.timeLineUnit = 'sample';
  % paramVisScp.timeLineUnit = 'second'; paramVisScp.featureRate = ... 
  [h_fig_scapeplot,x_axis,y_axis] = visualizeScapePlot(fitness_matrix,paramVisScp);
  title('Fitness scape plot','Interpreter','none');
  
  return