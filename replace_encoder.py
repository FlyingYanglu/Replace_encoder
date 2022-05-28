import sys
from replace_utils import *

model_a_path = sys.argv[1]  # model_a pt file
model_b_path = sys.argv[2]  # model_b pt file
save_dir = sys.argv[2]  # target save dir

replace_and_save(model_a_path, model_b_path, save_dir)