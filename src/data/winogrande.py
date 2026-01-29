from data.base_ds import format_ds
from datasets import load_dataset
import pandas as pd

def load_data(args, split='validation'):
    # WinoGrande uses 'validation' for test, 'train' for training
    split = 'validation' if split == 'test' else split
    
    dataset = load_dataset('winogrande', 'winogrande_xl', cache_dir=args.data_dir)[split]
    dataset = pd.DataFrame(dataset)
    
    if split == 'train':
        dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True)
        if args.data_size > 0:
            dataset = dataset.head(args.data_size)
    else:
        # Test/validation: use data_size if specified, otherwise min(total, 300)
        dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True)
        if args.data_size > 0:
            dataset = dataset.head(args.data_size)
        else:
            dataset = dataset.head(min(len(dataset), 300))

    questions, labels = [], []
    template = '{}\n(A) {}\n(B) {}\n\n'
    
    for sentence, option1, option2, answer in zip(
        dataset['sentence'], 
        dataset['option1'], 
        dataset['option2'], 
        dataset['answer']
    ):
        # answer is '1' or '2' (string or int)
        # Format the question by replacing the blank with options
        # The sentence contains an underscore '_' that needs to be replaced
        if '_' in sentence:
            # Simple approach: just show the sentence with options
            # The sentence has an underscore that represents the blank
            question_text = sentence
        else:
            question_text = sentence
        
        # Build the question with options
        question = template.format(question_text, option1, option2)
        
        # Convert answer '1' or '2' (string or int) to (A) or (B)
        answer_str = str(answer).strip()
        label = "(A)" if answer_str == "1" else "(B)"
        
        questions.append(question)
        labels.append(label)
    
    return questions, labels
