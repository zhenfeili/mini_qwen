from datasets import load_dataset

dataset = load_dataset("parquet", data_files="mini_data/pt/accommodation_catering_hotel/chinese/high/rank_00000.parquet", split="train")
print(dataset[0])

dataset = load_dataset("parquet", data_files="mini_data/sft/7M/train-00000-of-00075.parquet", split="train")
print(dataset[9])

dataset = load_dataset("parquet", data_files="mini_data/dpo/train-00000-of-00001.parquet", split="train")
print(dataset[5])
