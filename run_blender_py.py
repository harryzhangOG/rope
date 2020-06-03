import os
import sys
sys.path.append(os.getcwd())
os.system('blender -b -P rope_test.py -- -exp 0')