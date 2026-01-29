from data.base_ds import format_ds
from datasets import load_dataset
import pandas as pd

def load_data(args, split='validation'):
    # ARC doesn't have a validation split, use 'test' for evaluation
    split = 'test' if split in ['validation', 'test'] else split

    # Default to ARC-Challenge, but allow ARC-Easy via sub_data argument
    subset = 'ARC-Easy' if args.sub_data == 'easy' else 'ARC-Challenge'

    # Most HuggingFace datasets are now auto-converted to Parquet format
    dataset = load_dataset('allenai/ai2_arc', subset, cache_dir=args.data_dir)[split]
    dataset = pd.DataFrame(dataset)
    dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True)

    if split == 'train':
        if args.data_size > 0:
            dataset = dataset.head(args.data_size)
    else:
        # Test/validation: use data_size if specified, otherwise min(total, 300)
        if args.data_size > 0:
            dataset = dataset.head(args.data_size)
        else:
            dataset = dataset.head(min(len(dataset), 300))

    questions, labels = [], []

    for question_text, choices, answer_key in zip(dataset['question'], dataset['choices'], dataset['answerKey']):
        # Extract choice texts and labels
        choice_labels = choices['label']  # e.g., ['A', 'B', 'C', 'D']
        choice_texts = choices['text']    # e.g., ['option 1', 'option 2', ...]

        # Build the question with options
        if len(choice_labels) == 3:
            # Some ARC questions have 3 choices
            template = '{}\n(A) {}\n(B) {}\n(C) {}\n\n'
            question = template.format(question_text, choice_texts[0], choice_texts[1], choice_texts[2])
        elif len(choice_labels) == 4:
            # Most have 4 choices
            template = '{}\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n\n'
            question = template.format(question_text, choice_texts[0], choice_texts[1], choice_texts[2], choice_texts[3])
        elif len(choice_labels) == 5:
            # Some have 5 choices
            template = '{}\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n(E) {}\n\n'
            question = template.format(question_text, choice_texts[0], choice_texts[1], choice_texts[2], choice_texts[3], choice_texts[4])
        else:
            # Skip questions with unusual number of choices
            continue

        # Convert answer key to (X) format
        # Note: ARC answer keys use '1', '2', '3', '4' OR 'A', 'B', 'C', 'D'
        if answer_key.isdigit():
            # Convert numeric to letter
            answer_idx = int(answer_key) - 1
            if answer_idx < len(choice_labels):
                label = f"({choice_labels[answer_idx]})"
            else:
                continue
        else:
            # Already a letter
            label = f"({answer_key})"

        questions.append(question)
        labels.append(label)

    return questions, labels
