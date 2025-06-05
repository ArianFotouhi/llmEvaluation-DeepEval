import pandas as pd

class DataETL:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path

    def load_data(self):
        data = pd.read_csv(self.csv_file_path)
        return data.to_dict(orient='records')

    def get_ground_truth(self, prompt):
        data = pd.read_csv(self.csv_file_path)
        match = data[data['prompt'] == prompt]
        if not match.empty:
            return match['completion'].iloc[0]
        return None
