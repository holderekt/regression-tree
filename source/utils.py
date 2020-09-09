import pandas as pd

# Dataset loader splits the dataframe into target (n th column) and feature
def load_dataset(dataset_name, delete_columns):
    data = pd.read_csv(dataset_name)
    data = data.drop(data.columns[delete_columns], axis=1)
    column_names = data.columns
    features = data[data.columns.values[:-1]]
    targets = data[data.columns.values[-1]]
    return features.to_numpy(), targets.to_numpy(), column_names.to_numpy()

# Mean of element in y indexed by examples
def calculate_mean(y,examples):
    total = 0
    for exp in examples:
        total = total + y[exp]
    return total / len(examples)

# Print a progress bar with percentage
def print_progress_bar(percentage, progressbar_width):
    print("\r", end='')
    elements = int((progressbar_width / 100)*percentage)
    print("[", end='')
    for _ in range(elements):
        print("#", end='')
    for _ in range(progressbar_width - elements):
        print(" ", end='')
    print("]\t", end='')
    print(str(round(percentage,2)) + "%", end='')
    if(percentage == 100):
        print("")
