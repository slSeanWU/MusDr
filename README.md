# mir20_eval

Evaluation metrics for machine-composed symbolic music.

## Metrics

### Computed from Symbolic Music
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
  * ``initPaths.m``: for MATLAB (SM Toolbox) imports
  * ``sm_compute_scapeplot.m``: MATLAB script for computing SSMs and scape plots
  * ``run_matlab_scapeplot.py``: Python program that invokes the MATLAB script above
  * ``vis_scapeplot.py``: visualizes the computed scape plots
  * ``run_all_metrics.py``: runs all evaluation metrics and outputs results
  * ``mir20_eval/``
    * ``testdata/``: contains example testdata
    * ``eval_metrics.py``: contains the implementation of the 4 metrics mentioned above
    * ``side_utils.py``: some I/O and computation utilities 
    
## Usage Notes

### Prerequisites
  * Clone this repository  
  
  * Get **Python 3.6** (with which we tested our functions and scripts)
    * URL: https://www.python.org/downloads/release/python-368/  
    
  * Install the requirements with ``pip3`` (or ``pip``, depending on your system)
    ```shell
      pip3 install -r requirements.txt
    ```

NOTE: all of the following commands run on the example testdata under ``mir20_eval/testdata/`` 

### Computation of Fitness Scape Plots
  -- _required for metric **SI**_
  * Get MATLAB (if you don't have one)
    * URL: https://www.mathworks.com/products/get-matlab.html?s_tid=gn_getml
    * Be sure to include the "MATLAB Signal Processing Toolbox"  
      
  * Download the "SM Toolbox" (Mueller _et. al_, 2013) for scape plot computation
    * URL: https://rb.gy/20sr04
    * Unzip it (directly, not into a new directory) under the repo root directory
      
  * Run the Python script that invokes the MATLAB function to compute scape plots 
    ```shell
    python3 run_matlab_scapeplot.py \
       -a mir20_eval/testdata/audio  \  # input audio directory
       -s mir20_eval/testdata/ssm   \   # SSMs output directory
       -p mir20_eval/testdata/scplot  \ # scape plots output directory
       -j [num of MATLAB processes]     # 2~4 recommended
    ```
    
  * Visualize the scape plots
    ```shell
    python3 vis_scapeplot.py \
       -p mir20_eval/testdata/scplot  \ # input scape plots directory
       -f mir20_eval/testdata/scfig     # scape plot figures output directory
    ```
    
### Run All Evaluation Metrics and Get the Report  
 * Run ``run_all_metrics.py`` when you have computed the fitness scape plots
    ```shell
    python3 run_all_metrics.py \
       -s mir20_eval/testdata/symbolic  \ # input symbolic music directory
       -p mir20_eval/testdata/scplot  \   # input scape plots directory (having the same pieces as the directory above)
       -o testout.csv \                   # output file for results
       --timescale_bounds 3 8 15          # (optional) timescale bounds for short-, mid-, and long-term SI metric, respectively; defaults to 3 8 15
    ```
 
## Release Notes
 * July 8th, 2020
   * The metrics **H**, **GS**, and **CPI** now only natively support event-based representations defined in **Jazz Transformer**  
     -- we welcome contributions to extend the compatibility to other representations, or even general MIDIs 
   * Metric **SI** now relies on MATLAB and audio  
     -- planning to re-implement it with Python-native package ``librosa``  
     -- considering adding an option to take pianorolls (obtained from MIDIs) as inputs
