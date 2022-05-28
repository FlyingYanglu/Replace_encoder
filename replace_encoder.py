
from replace_utils import *
import argparse


class CommandLine:
    def __init__(self):
        parser = argparse.ArgumentParser(description = "Arguments")
        parser.add_argument("model_a")
        parser.add_argument("model_b")
        parser.add_argument("save_dir")
        parser.add_argument("-p", "--part", help = "The part you want to use for your first model", required = False, default = "encoder")
        parser.add_argument("-k", "--keep_params", help = "if set to 0, keep no other params inside the model; if set to 1, keep model_a's other parameters, if set to 2, keep model_b's other parameters", required = False, default = '0')
        
        argument = parser.parse_args()
        status = False
        self.model_a = argument.model_a
        self.model_b = argument.model_b
        self.save_dir = argument.save_dir
        self.part = "encoder"
        self.keep_params = 0
        if argument.part:
            self.part = argument.part
            status = True
        if argument.keep_params:
            self.keep_params = int(argument.keep_params)
            status = True
        


if __name__ == '__main__':
    app = CommandLine()
    #print(vars(app))
    replace_and_save(app.model_a, app.model_b, app.save_dir, app.part, app.keep_params)