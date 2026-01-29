

import argparse, sys, os, copy, time, random, json, pickle, re, collections
from itertools import combinations
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datetime import datetime




def get_instruction_suffix(args):
    if args.data in ['gsm8k']:
        if args.bae :
            return ' Make sure to state your answer at the end of the response.'
        elif args.cot :
            return " Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: 123}'. Let's think step by step."
        else :
            return ' Make sure to state your final answer in curly brackets at the very end of your response, just like: "{final answer: 123}".'
    elif args.data in ['hellaswag','pro_medicine','formal_logic','arc','truthfulqa','winogrande']:
        if args.bae :
            return ' Put your final answer in the form (X) at the end of your response.'
        elif args.cot :
            return " Make sure to state your final answer choice in curly brackets at the very end of your response, just like: '{final answer: (A)}'. Let's think step by step."
        else :
            return ' Make sure to state your final answer choice in curly brackets at the very end of your response, just like: "{final answer: (A)}".'




def evaluate_gsm8k(responses, answer):
    # Returns True if correct, False if incorrect
    final_answers = []
    for _, response in responses.items():
        try:
            pred = re.findall(r"\{(.*?)\}", response)[-1]
            pred = float(pred.replace("final answer:", "").strip())
            final_answers.append(np.round(pred, 1))
        except :
            final_answers.append("")

    if len(set(final_answers)) == 1 and list(set(final_answers))[0] == "":
        final_answers = [""] * len(final_answers)
        debate_answer = ""
    else :
        counter = collections.Counter([x for x in final_answers if x != ""])
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common)

    return final_answers, debate_answer, debate_answer == np.round(answer, 1)


def base_evaluate_gsm8k(responses, answer):
    final_answers = []
    for _, sentence in responses.items():
        parts = sentence.split(" ")
        for part in parts[::-1]:
            try:
                ans = float(part)
                final_answers.append(ans)
                break
            except:
                continue

    counter = collections.Counter([x for x in final_answers if x != ""])
    try:
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common)
    except :
        debate_answer = ""

    return final_answers, debate_answer, debate_answer == np.round(answer, 1)


def evaluate_mcq(responses, answer):
    # Returns True if corret, False if incorrect
    final_answers = []
    for _, response in responses.items():

        try:
            pred = re.findall(r"\{(.*?)\}", response)[-1]
            pred = pred.replace("final answer:", "").strip()
            if len(pred) == 0 :
                final_answers.append("")
            elif len(pred) < 3 :
                pred = pred[0]
                final_answers.append(f"({pred})")
            else :
                pred = pred[1]
                final_answers.append(f"({pred})")
        except :
            final_answers.append("")
    
    if len(set(final_answers)) == 1 and list(set(final_answers))[0] == "":
        final_answers = [""] * len(final_answers)
        debate_answer = ""
    else :
        counter = collections.Counter([x for x in final_answers if x != ""])
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common) # if there is a tie, will choose randomly
    return final_answers, debate_answer, debate_answer == answer








def base_evaluate_mcq(responses, answer):

    final_answers = []
    for _, input_str in responses.items():

        pattern = r'\((\w)\)'
        matches = re.findall(pattern, input_str)

        solution = None
        for match_str in matches[::-1]:
            solution = match_str.upper()
            if solution:
                final_answers.append(f"({solution})")
                break

    counter = collections.Counter([x for x in final_answers if x != ""])
    try :
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common) # if there is a tie, will choose randomly
    except :
        debate_answer = ""
    return final_answers, debate_answer, debate_answer == answer




