import sys
from pathlib import Path


class TeedLog:
    def __init__(self, dir_path):
        self.terminal = sys.stdout

        self.filename = Path(dir_path) / "output.log"
        self.log = open(self.filename, "a", encoding="utf-8")
        print(f"--- Save ouputs to file: {self.filename} ---")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss):
        score = -val_loss  
        output_message = ""
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                output_message = (
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return output_message
