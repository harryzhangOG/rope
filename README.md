# Robots of the Lost Arc: Learning to Dynamically Manipulate Fixed-Endpoint Ropes and Cables
## Requirements:
  * Python 2 / 3
  * PyTorch
  * Numpy
  * Matplotlib
  * Blender 2.8.2
  * Open CV
## Simulated Environment Usage:
  * To evaluate apex points on obstacle settings for a certain task, create a text file named `eval_settings.txt`, and list obstacle settings and apex points where each line is `[OBSTACLE X SIZE IN METERS] [OBSTACLE Y SIZE IN METERS] [OBSTACLE Z SIZE IN METERS] [OBSTACLE X IN METERS] [OBSTACLE Y IN METERS] [BASE ANGLE IN RADIANS] [SHOULDER ANGLE IN RADIANS] [ELBOW ANGLE IN RADIANS]`, where each line is a separate obstacle setting/apex point, and each value is delimited by a space bar.
    * Then, run `blender -P rope_ur5.py -- -mode EVAL -task [TASK NAME]`
    * The 3 tasks are `VAULTING`, `KNOCKING`, and `WEAVING` (Currently, only `VAULTING` is supported)
* To generate training data, run `blender -P rope_ur5.py -- -mode DATAGEN -num [NUMBER OF TRAINING SAMPLES] -image 1 -task [TASK NAME]`. (Currently, only `VAULTING` is supported)
* To fully accelerate training data generation on a local machine, I recommend using GNU Screen and run multiple datagen script in parallel, with different exp number. In this way, you can then load the generated npy files and then concatenate them together using `format_plot.ipynb`.
* To start training, run `python train_ur5_sim_resnet.py`.
* To visualize training process, load the val loss and training loss npy files to `format_plot.ipynb` and plot from there.
* To evaluate the trained mode performance, run `blender -P rope_ur5.py -- -mode MODEL_EVAL -num [NUMBER OF EVAL SAMPLES] -task [TASK NAME]`. (Currently, only `VAULTING` is supported)

## Real Environment Usage:
  * To experiment in real world environment, go to directory `physical`.
    * The Python files in that directory correspond to the three tasks and each file will call an external C++ file to run the UR5 robot with a fast trajectory that minimizes the jerk.
    * In `physical`, `physical_data_collection.py` is for vaulting and knocking tasks datagen and `physical_snake_datagen.py` is for weaving task datagen.
    * In `physical`, `train_ur5_TASK_physical.py` trains the policy for `TASK` using the collected data.
    * The scripts for the three tasks follow use the same command line arguments `python FILE.py --seq CURRENT_TRIAL_INDEX --dirc SAVE_DIRECTORY`
    * The file `repeat_plot.py` contains the method that overlays different trials of the excecution to assess the repeatability of the motion.
  * The real-world environment setup requires a clear, well-lit space (2.5m x 5m) space in front of the UR5 station.

## Pytorch and Blender
  * Making Blender run any third-party packages is hard, and making it run PyTorch is even harder. To achieve this, you need to do the following steps:
    * First, `cd` into your Blender installation path. For me, it is in `/Applications/Blender.app`. Afterwards, dive deeper into this directory by `cd` into `/Applications/Blender.app/Contents/Resources/2.82/python/lib/python3.7/site-packages`.
    * Now, you are going to install PyTorch in Blender Python. To do this, run `pip install --target=/Applications/Blender.app/Contents/Resources/2.82/python/lib/python3.7/site-packages torch`. This will make pip reinstall pytorch to blender's python, even though you might have PyTorch installed on your machine already.
    * Then, you might see `Library not loaded: @rpath/libc++.1.dylib` error when you are running the script. When this happens, do not panic, run the following command: `install_name_tool -add_rpath /Applications/Blender.app/Contents/Resources/2.82/python/lib/python3.7/site-packages/torch/_C.cpython-37m-darwin.so`, and this should fix the issue. 
