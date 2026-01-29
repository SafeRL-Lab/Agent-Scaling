from data.base_ds import format_ds
from datasets import load_dataset
import pandas as pd

def load_data(args, split='validation'):
    # TruthfulQA only has 'validation' split
    split = 'validation'

    dataset = load_dataset('truthful_qa', 'multiple_choice', cache_dir=args.data_dir)[split]
    dataset = pd.DataFrame(dataset)

    # Shuffle and limit dataset size
    dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True)
    if args.data_size > 0:
        dataset = dataset.head(args.data_size)
    else:
        dataset = dataset.head(min(len(dataset), 300))

    questions, labels = [], []
    choices_list = "ABCDEFGHIJ"  # Support up to 10 choices

    for question_text, mc1_targets in zip(dataset['question'], dataset['mc1_targets']):
        # mc1_targets is a dict with 'choices' and 'labels' (0/1 for each choice)
        # In mc1, exactly one choice has label=1 (correct answer)
        choice_texts = mc1_targets['choices']
        choice_labels = mc1_targets['labels']  # List of 0s and 1s

        num_choices = len(choice_texts)
        if num_choices < 2 or num_choices > 10:
            # Skip questions with too few or too many choices
            continue

        # Find the correct answer
        try:
            correct_idx = choice_labels.index(1)
            answer_letter = choices_list[correct_idx]
        except ValueError:
            # No correct answer found, skip
            continue

        # Build question template dynamically
        question_lines = [question_text]
        for i, choice_text in enumerate(choice_texts):
            question_lines.append(f"({choices_list[i]}) {choice_text}")
        question_lines.append("")  # Empty line at end

        question = "\n".join(question_lines)
        label = f"({answer_letter})"

        questions.append(question)
        labels.append(label)

    return questions, labels
