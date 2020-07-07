# mir20_eval

Evaluation metrics for machine-composed symbolic music.

## Metrics
  * Pitch-Class Histogram Entropy (**H**)
  * Grooving Pattern Similarity (**GS**)
  * Chord Progression Irregularity (**CPI**)
  * Structureness Indicator (**SI**)
  
## Usage Notes

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
       -a [input audio directory]  \
       -s [SSMs output directory]   \
       -p [scape plots output directory]  \
       -j [num of MATLAB processes] # 2~4 recommended
    ```
    
  * Visualize the scape plots
    ```shell
    python3 vis_scapeplot.py \
       -p [input scape plots directory]  \ # the "-p" argument of last step
       -f [scape plot figures output directory]
    ```
