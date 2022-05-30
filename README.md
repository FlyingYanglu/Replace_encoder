# Replace_encoder
Tool for replacing encoder in seq2seq models

Run `python replace_encoder.py <path to model you want to save encoder> <path to model you want to replace encoder> <save directory>`

Options 

-p <string 'encoder' or 'decoder' specify the part you want to keep for your first model> 

-k <int in [0,1,2], if 0 keep no other params other than the model; if 1, keep other params from first model; if 2 keep other params from second model>` 

Or import function from replace_utils and use directly
