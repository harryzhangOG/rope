# Learning Rope Dynamics via Inverse Model
## Requirements:
  * Python 2 / 3
  * PyTorch
  * Numpy
  * Matplotlib
  * Blender 2.8.2
  * Open CV
## Usage:
  * To generate training data, run `blender -b -P datagen.py -- -exp YOUR_EXP_NUM`. 
  * To fully accelerate training data generation on a local machine, I recommend using GNU Screen and run multiple datagen script in parallel, with different exp number. In this way, you can then load the generated npy files and then concatenate them together using `format_plot.ipynb`.
  * To start training, run `python train_inv_model.py`.
  * To visualize training process, load the val loss and training loss npy files to `format_plot.ipynb` and plot from there.
  * To evaluate the trained mode performance, there are two different ways:
    * First, if you are only doing one-step action prediction, run `gen_test.py` to generate testing data and then load the data and predict by running `eval_inv_model_one_step.py`. Finally, load the prediction and ground truth to `blender -P eval_rope_one_step.py` to compare and visualize in Blender. This is an easy task and you should expect very high performance.
    * Second, if you are doing multi-step actions prediction, run `gen_test.py` to generate testing data and then load the data and predict by running `blender -P eval_rope_multi_step.py`, which will also be visualzied in Blender. This step involves getting PyTorch work in Blender 2.8.2, which is not trivial. The specific steps are covered below.
## Pytorch and Blender
  * Making Blender run any third-party packages is hard, and making it run PyTorch is even harder. To achieve this, you need to do the following steps:
    * First, `cd` into your Blender installation path. For me, it is in `/Applications/Blender.app`. Afterwards, dive deeper into this directory by `cd` into `/Applications/Blender.app/Contents/Resources/2.82/python/lib/python3.7/site-packages`.
    * Now, you are going to install PyTorch in Blender Python. To do this, run `pip install --target=/Applications/Blender.app/Contents/Resources/2.82/python/lib/python3.7/site-packages torch`. This will make pip reinstall pytorch to blender's python, even though you might have PyTorch installed on your machine already.
    * Then, you might see `Library not loaded: @rpath/libc++.1.dylib` error when you are running the script. When this happens, do not panic, run the following command: `install_name_tool -add_rpath /Applications/Blender.app/Contents/Resources/2.82/python/lib/python3.7/site-packages/torch/_C.cpython-37m-darwin.so`, and this should fix the issue. 
