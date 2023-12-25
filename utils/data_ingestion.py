import pandas as pd

def data_ingest(file):
    try:
        df = pd.read_csv(file, sep="\t", header=None)
        df.columns = ["Labels", "Messages"]
        return df
    except Exception as e:
        return str(e)
