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

    encoder_model = {}
    # extract encoder and store in a orderedDict()
    encoder_orddict = OrderedDict()
    for key, values in model["model"].items():
        if key.startswith("encoder"):
            encoder_orddict[key] = values
    encoder_model["model"] = encoder_orddict

    # add other parameters if needed
    if with_other_params:
        for key, values in model.items():
            if key != "model":
                encoder_model[key] = values

    return encoder_model

def replace_encoder(model_a, model_b):
    """
    return a new model dict that uses parameter from model_b and encoder from model_a
    :param model_a: fairseq transformer model
    :param model_b: fairseq transformer model
    """
    
    #assert model_a != model_b  # do not allow overwrite input

    #print("&&&&&&&&&&&&&&&&&&&&")

    # scratchModel["model"]["encoder.version"][0] = 2 
    # print(model_a["model"]["encoder.version"])
    # print(model_a["model"]["encoder.version"].is_cuda)
    # print(model_a["model"]["encoder.version"].dtype)

    #print(type(model_a["model"]))

    # create new dict
    new_model = copy.deepcopy(model_b)
    encoder_model = extract_model_part(model_a, 'encoder')


    for (layerName_a,value_a), (layerName_b,value_b) in zip(encoder_model["model"].items(), new_model["model"].items()):
        if (layerName_a == layerName_b and layerName_a.startswith("encoder")):
            model_b["model"][layerName_b] = value_a
        else:
            print("Not Encoder anymore, STOP")

    return new_model

def replace_and_save(model_a_path, model_b_path, store_dir):
    """
    :param model_a_path: path for model_a.pt
    :param model_b_path: path for model_b.pt
    :param store_dir: store directory
    """
    model_a = checkpoint_utils.load_checkpoint_to_cpu(model_a_path)
    model_b = checkpoint_utils.load_checkpoint_to_cpu(model_b_path)
    replaced_model = replace_encoder(model_a, model_b)

    model_a_name = os.path.splitext(os.path.basename(model_a_path))[0]
    model_b_name = os.path.splitext(os.path.basename(model_b_path))[0]
    new_file = model_a_name + "_enc_" + model_b_name + "_dec.pt"
    torch.save(replaced_model, f=open(os.path.join(store_dir, new_file), "wb"))


