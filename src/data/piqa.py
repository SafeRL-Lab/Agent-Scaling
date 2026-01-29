from data.base_ds import format_ds
from datasets import load_dataset
import pandas as pd

def load_data(args, split='validation'):
    # PIQA uses 'validation' for evaluation
    split = 'validation' if split == 'test' else split

    # Use regisss/piqa instead of ybisk/piqa to avoid dataset script issues with datasets>=4.0.0
    dataset = load_dataset('regisss/piqa', cache_dir=args.data_dir)[split]
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
    template = 'Goal: {}\n\nWhich solution is more appropriate?\n(A) {}\n(B) {}\n\n'

    for goal, sol1, sol2, answer in zip(dataset['goal'], dataset['sol1'], dataset['sol2'], dataset['label']):
        question = template.format(goal, sol1, sol2)
        # label is 0 or 1 (0 for sol1, 1 for sol2)
        label = "(A)" if answer == 0 else "(B)"

        questions.append(question)
        labels.append(label)

    return questions, labels
