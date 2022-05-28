from typing import ValuesView
from fairseq import checkpoint_utils
import torch
import copy
import os

import argparse

from collections import OrderedDict


def extract_model_part(model, part = "encoder", with_other_params = False):
    """
    extract part from the model
    :param part: the part you want to extract from the model can either be 'encoder' or 'decoder'
    :param with_other_params: default to False. If True, the model returned will include every parameter in the original model with only 'model' modified
    """
    assert part in ["encoder", "decoder"]

    if part == "encoder":
        res_model = {}
        # extract encoder and store in a orderedDict()
        encoder_orddict = OrderedDict()
        for key, values in model["model"].items():
            if key.startswith("encoder"):
                encoder_orddict[key] = values
        res_model["model"] = encoder_orddict
    else:
        res_model = {}
        # extract decoder and store in a orderedDict()
        decoder_orddict = OrderedDict()
        for key, values in model["model"].items():
            if key.startswith("decoder"):
                decoder_orddict[key] = values
        res_model["model"] = decoder_orddict
    # add other parameters if needed
    if with_other_params:
        for key, values in model.items():
            if key != "model":
                res_model[key] = values

    return res_model

def replace_part(model_a, model_b, part = 'encoder', keep_params = 0):
    """
    return a new model dict that concat model_a's part with model_b's counterpart.
    :param model_a: fairseq transformer model
    :param model_b: fairseq transformer model
    :param keep_params: if set to 0, keep no other params inside the model; if set to 1, keep a's other parameters, if set to 2, keep b's other parameters
    """
    
    if part == 'encoder':
        print("extracting encoder")
        encoder_model = extract_model_part(model_a, 'encoder')
        print("extracting decoder")
        decoder_model = extract_model_part(model_b, 'decoder')
    elif part == "decoder":
        print("extracting encoder")
        encoder_model = extract_model_part(model_b, 'encoder')
        print("extracting decoder")
        decoder_model = extract_model_part(model_a, 'decoder')
        
    new_model = {}
    new_model_orderdict = OrderedDict(list(encoder_model.items()) + list(decoder_model.items()))
    new_model["model"] = new_model_orderdict
    
    # whether to keep other parameters
    if keep_params == 0:
        return new_model
    elif keep_params == 1:
        for key, values in model_a.items():
            if key != "model":
                new_model[key] = values
    elif keep_params == 2:
        for key, values in model_b.items():
            if key != "model":
                new_model[key] = values
                
    return new_model

def replace_and_save(model_a_path, model_b_path, store_dir, part = "encoder", keep_params = 0):
    """
    :param model_a_path: path for model_a.pt
    :param model_b_path: path for model_b.pt
    :param store_dir: store directory
    :param replace_part: use replace_part of model_a and counter_part of model_b to create a new model
    :param keep_params: if set to 0, keep no other params inside the model; if set to 1, keep a's other parameters, if set to 2, keep b's other parameters
    """
    model_a = checkpoint_utils.load_checkpoint_to_cpu(model_a_path)
    model_b = checkpoint_utils.load_checkpoint_to_cpu(model_b_path)
    replaced_model = replace_part(model_a, model_b, part, keep_params)

    model_a_name = os.path.splitext(os.path.basename(model_a_path))[0]
    model_b_name = os.path.splitext(os.path.basename(model_b_path))[0]
    new_file = model_a_name + "_enc_" + model_b_name + "_dec.pt"
    torch.save(replaced_model, f=open(os.path.join(store_dir, new_file), "wb"))


