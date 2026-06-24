from dataloader_pt import get_dataset

dataset = get_dataset("data/noisy_4s", "data/clean_4s", batch_size=2)

for noisy, clean in dataset.take(1):
    print("Batch loaded!")
    print("Noisy shape:", noisy.shape)
    print("Clean shape:", clean.shape)
