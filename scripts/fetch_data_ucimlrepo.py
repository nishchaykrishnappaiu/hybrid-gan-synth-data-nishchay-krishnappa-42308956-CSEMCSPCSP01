from ucimlrepo import fetch_ucirepo
import argparse
from pathlib import Path
import pandas as pd



class fetch_dataset():
    def __init__(self,dataset_id,datasetname,output_csv_file_path):
        self.dataset_id = dataset_id
        self.datasetname = datasetname
        self.output_csv_file_path = output_csv_file_path
    def fetch_dataset(self):

        if not self.dataset_id and not self.datasetname:
            raise ValueError("Provide either --id or --name for the UCI dataset.")
        
        if self.dataset_id:
            dataset = fetch_ucirepo(id=int(self.dataset_id))
        else:
            dataset = fetch_ucirepo(name=self.datasetname)

        x = dataset.data.features
        y = dataset.data.targets

        if isinstance(y,pd.DataFrame):
            self.df = pd.concat([x,y],axis=1)
        else:
            self.df=x.copy()
            self.df["target"] = y
    def saving_dataset(self):
        Path(self.output_csv_file_path).parent.mkdir(parents=True,exist_ok=True)
        self.df.to_csv(self.output_csv_file_path,index=False)
        print(f"\u2705 Saved dataset to {self.output_csv_file_path}")
        print("Shape:", self.df.shape)
        print("Columns:", list(self.df.columns))


if __name__ == "__main__":
    fd = fetch_dataset(dataset_id=20,datasetname="Census income",output_csv_file_path="data/real/input.csv")
    fd.fetch_dataset()
    fd.saving_dataset()
