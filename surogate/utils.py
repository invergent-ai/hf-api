import json


def write_dataset_info(dataset, path="/tmp"):
    splits = dataset.shape  # Shape of each split (number of rows, number of columns)
    column_names = dataset.column_names

    splits_info = {}
    for split_name, shape in splits.items():
        splits_info[split_name] = {
            "num_rows": shape[0],
            "column_names": column_names.get(split_name, []),
            "first_rows": dataset[split_name].select(range(min(3, shape[0]))).to_list()
        }
        
    

    with open(f"{path}/surogate_info.json", "w") as f:
        json.dump(splits_info, f, indent=2)