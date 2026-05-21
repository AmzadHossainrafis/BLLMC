"""
src.BLLMC.data.loader

license : MIT
author :  Amzad hossain rafi
email : [EMAIL_ADDRESS]

log history :
    2026-05-16 12:13 PM : create the data loader
    2026-05-16 12:14 PM : add the config argument



to do :
    1. Implement the dataset for classification
    2. Implement the dataset for instruction


"""

from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch


class BanglaDataset(Dataset):
    """
    A PyTorch Dataset for processing a single string of text.
    arg :
        text (str) : path to the text file
        tokenizer (str) : tokenizer to be used
        config (Config) : configuration object

    Methods :
    __init__(self, text, tokenizer, config) : initialize the dataset
    __len__(self) : return the number of samples in the dataset
    __getitem__(self, idx) : return the idx-th sample from the dataset



    """

    def __init__(self, text, tokenizer, config):
        self.config = config
        self.tokenizer = tokenizer

        token_ids = tokenizer.encode(
            text, allowed_special={"<|EOS|>", "<p>", "</p>", "<number>", "</strong>"}
        )
        self.token_ids = torch.tensor(token_ids)

    def __len__(self):

        return max(
            0,
            (len(self.token_ids) - self.config.max_length - 1) // self.config.stride
            + 1,
        )

    def __getitem__(self, idx):
        start_idx = idx * self.config.stride
        end_idx = start_idx + self.config.max_length

        input_ids = self.token_ids[start_idx:end_idx]
        target_ids = self.token_ids[start_idx + 1 : end_idx + 1]

        return input_ids, target_ids


class BanglaDatasetClassification(Dataset):
    def __init__(self, config):
        super().__init__()
        pass

    def __len__(self):
        pass

    def __getitem__(self, x):
        pass


class BanglaDatasetInstruction(Dataset):

    def __init__(self, config) -> None:
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, x):
        pass


def create_dataloader(data, encoder, config):
    tokenizer = tiktoken.get_encoding(encoder)
    dataset = BanglaDataset(data, tokenizer, config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=config.drop_last,
    )
