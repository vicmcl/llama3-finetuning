from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

dataset = load_dataset(
    "amaye15/Stack-Overflow-Zero-Shot-Classification", split="train"
)

tags_column = pd.DataFrame(dataset.to_dict())['Tags'].str.split(', ')
tags_list = tags_column.explode().reset_index()['Tags']
main_tags = tags_list.value_counts()[
    (tags_list.value_counts() > 1).values
].index.to_numpy()

new_tags_column = (
    tags_column
    .apply(lambda lst: [x if x in main_tags else 'N/A' for x in lst])
    .apply(lambda x: ', '.join(x))
)

n_row = 100_000
dataset = (
    dataset
    .rename_column("Title", "input")
    .add_column("output", new_tags_column.values)
    .remove_columns(["Tags", "Predicted_Tag_Scores", "Predicted_Tags"])
    .select(range(n_row))
)

X_train, X_test, y_train, y_test = train_test_split(
    dataset['input'], dataset['output'], test_size=0.05, random_state=42
)

train_set = Dataset.from_dict({'input': X_train, 'output': y_train})
test_set = Dataset.from_dict({'input': X_test, 'output': y_test})

train_set.to_csv('../../data/train.csv')
test_set.to_csv('../../data/test.csv')