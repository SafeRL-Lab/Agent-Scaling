import pandas as pd
import random
import torch
from datasets import Dataset, concatenate_datasets
import re

def load_data(args, split):
    if args.data == 'hellaswag' :
        from data.hellaswag import load_data as load_hellaswag
        return load_hellaswag(args, split=split)
    elif args.data == 'pro_medicine' :
        from data.mmlu_pro_medicine import load_data 
        return load_data(args, split=split)
    elif args.data == 'formal_logic' :
        from data.mmlu_formal_logic import load_data 
        return load_data(args, split=split)
    elif args.data == 'gsm8k' :
        from data.gsm8k import load_data as load_gsm8k
        return load_gsm8k(args, split=split)
    elif args.data == 'arc':
        from data.arc import load_data as load_arc
        return load_arc(args, split=split)
    elif args.data == 'piqa':
        from data.piqa import load_data as load_piqa
        return load_piqa(args, split=split)
    elif args.data == 'truthfulqa':
        from data.truthfulqa import load_data as load_truthfulqa
        return load_truthfulqa(args, split=split)
    elif args.data == 'winogrande':
        from data.winogrande import load_data as load_winogrande
        return load_winogrande(args, split=split)
