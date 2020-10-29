# MusDr.

Evaluation metrics for machine-composed symbolic music. 

Python implementation of the evaluation metrics proposed in Section 5 of our paper: 
 * **The Jazz Transformer on the Front Line: Exploring the Shortcomings of AI-composed Music through Quantitative Measures**  
   Shih-Lun Wu and Yi-Hsuan Yang  
   _The 21st International Society for Music Information Retrieval Conference_ (ISMIR), 2020.

## Metrics

### Computed from Symbolic Music
  The supported input format is _event token sequences_ that can be mapped to MIDIs, rather than general MIDIs. See [this paper](https://arxiv.org/abs/2002.00212) (Huang and Yang, 2020) for a thorough introduction.
  * Pitch-Class Histogram Entropy (**H**)  
    -- measures erraticity of **pitch usage** in shorter timescales (e.g., 1 or 4 bars) 
  * Grooving Pattern Similarity (**GS**)  
    -- measures consistency of **rhythm** across the entire piece
  * Chord Progression Irregularity (**CPI**)  
    -- measures consistency of **harmony** across the entire piece
    
### Computed from Audio (and the Resulting "Fitness Scape Plot")
  * Structureness Indicator (**SI**)  
    -- detects presence of **repeated structures** within a specified range of timescale
  
## Directory Structure
  * ``initPaths.m``: for MATLAB ([SM Toolbox](https://www.audiolabs-erlangen.de/resources/MIR/SMtoolbox/)) imports
  * ``sm_compute_scapeplot.m``: MATLAB script for computing SSMs and scape plots
  * ``run_matlab_scapeplot.py``: Python program that invokes the MATLAB script above
  * ``run_python_scapeplot.py``: Python-native equivalent of ``run_matlab_scapeplot.py``
  * ``vis_scapeplot.py``: visualizes the computed scape plots
  * ``run_all_metrics.py``: runs all evaluation metrics and outputs results
  * ``musdr/``
    * ``testdata/``: contains example testdata
    * ``eval_metrics.py``: contains the implementation of the 4 metrics mentioned above
    * ``side_utils.py``: some I/O and computation utilities
  * ``mueller_audio_tools/``: contains the Python equivalents of the required utilities in SM Toolbox, retrieved from [FMP Notebooks](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C0/C0.html)
    
## Usage Notes

### Prerequisites
  * Clone this repository  
  
  * Get **Python 3.6** (with which we tested our functions and scripts)
    * URL: https://www.python.org/downloads/release/python-368/  
    
  * Install the requirements with ``pip3`` (or ``pip``, depending on your system)
    ```shell
      pip3 install -r requirements.txt
    ```

NOTE: all of the following commands run on the example testdata under ``musdr/testdata/`` 

### Computation of Fitness Scape Plots
  -- _required for metric **SI**_
  
#### Run with MATLAB
  * **More tedious setup, but runs faster**
  * Get MATLAB (if you don't have one)
    * URL: https://www.mathworks.com/products/get-matlab.html?s_tid=gn_getml
    * Be sure to include the "MATLAB Signal Processing Toolbox"  
      
  * Download the "SM Toolbox" (Müller _et. al_, 2013) for scape plot computation
    * URL: https://www.audiolabs-erlangen.de/resources/MIR/SMtoolbox/
    * Unzip it (directly, not into a new directory) under the repo root directory
    * Revise the file ``MATLAB_SM-Toolbox_1.0/MATLAB-Chroma-Toolbox_2.0/wav_to_audio.m`` (_line 107, 108_)  
      -- due to compatibility issues with newer MATLAB versions
      * Remove
      ```Matlab
         if strcmp(ext,'.wav')
            [f_audio,fs,nbits] = wavread(strcat(dirAbs,dirRel,wavfilename));
      ```
      * Add
      ```Matlab
         if ~( strcmp(ext,'.mp3') && strcmp(ext, '.wav') )
            [f_audio,fs] = audioread(strcat(dirAbs,dirRel,wavfilename));
            nbits = 24;
      ```
      
  * Run the Python script that invokes the MATLAB function to compute scape plots 
   ```shell
   python3 run_matlab_scapeplot.py \
      -a musdr/testdata/audio  \
      -s musdr/testdata/ssm   \
      -p musdr/testdata/scplot  \
      -j 1
   ```
   * ``-a``: input audio directory
   * ``-s``: output directory for SSMs
   * ``-p``: output directory for scape plots
   * ``-j``: number of processes spawned for scape plot computation (optional, defaults to ``1``; ``2~4`` is recommended)
    
#### Run with Python
 * **No additional setup required, but runs slowly on longer songs**
  ```shell
  python3 run_python_scapeplot.py \
     -a musdr/testdata/audio  \
     -s musdr/testdata/ssm   \
     -p musdr/testdata/scplot  \
     -j 1
  ```
   * ``-a``: input audio directory
   * ``-s``: output directory for SSMs
   * ``-p``: output directory for scape plots
   * ``-j``: number of processes spawned for scape plot computation (optional, defaults to ``1``; ``2~4`` is recommended)
 
#### Visualize the Scape Plots
 * Works on scape plots in both ``.mat`` and ``.npy`` formats
  ```shell
  python3 vis_scapeplot.py \
     -p musdr/testdata/scplot  \ # input scape plots directory
     -f musdr/testdata/scfig     # scape plot figures output directory
  ```
    
### Run All Evaluation Metrics and Get the Report  
 * Run ``run_all_metrics.py`` when you have computed the fitness scape plots
  ```shell
  python3 run_all_metrics.py \
     -s musdr/testdata/symbolic  \
     -p musdr/testdata/scplot  \
     -o testout.csv \
     --timescale_bounds 3 8 15
  ```
  * ``-s``: input symbolic music directory
  * ``-p``: input scape plots directory (having the same pieces as the symbolic music directory)
  * ``-o``: path to output file for results
  * ``--timescale_bounds``: timescale bounds for _short-_, _mid-_, and _long-term_ **SI** metric (optional, defaults to ``3 8 15``)
    
## Release Notes
 * July 8th, 2020
   * The metrics **H**, **GS**, and **CPI** now only natively support event-based representations defined in the **Jazz Transformer**  
     -- we welcome contributions to extend the compatibility to other representations, or even general MIDIs 
   * Metric **SI** now relies on MATLAB and audio  
     -- planning to re-implement it with Python-native package ``librosa``  
     -- considering adding an option to take pianorolls (obtained from MIDIs) as inputs
 * July 22nd, 2020
   * Add Python support for SSM and fitness scape plot computation

## Acknowledgements
This repository makes use of the following open-source utilities:
 * **MATLAB SM Toolbox**
   * **SM Toolbox: MATLAB Implementations for Computing and Enhancing Similarity Matrices**  
     Meinard Müller, Nanzhu Jiang, and Harald G. Grohganz  
     In Proceedings of _the 53rd Audio Engineering Society_ (AES), 2014. 
   * URL: https://www.audiolabs-erlangen.de/resources/MIR/SMtoolbox/
 * **Python FMP Notebooks**
   * **FMP Notebooks: Educational Material for Teaching and Learning Fundamentals of Music Processing**  
      Meinard Müller and Frank Zalkow  
      In Proceedings of _the 20th International Society for Music Information Retrieval Conference_ (ISMIR), 2019.
   * URL: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C0/C0.html
   * Special thanks to **Wen-Yi Hsiao** (_@ Taiwan AILabs_, [personal GitHub](https://github.com/wayne391)) for retrieving the required functions for this repository

## See Also
 * The repository of the **Jazz Transformer**, our AI Jazz composition model:  
   https://github.com/slSeanWU/jazz_transformer
