import pandas as pd
from ifcr import build_ifcr_counter

if __name__ == "__main__":
    df = pd.read_csv("data_example.csv")
    df_ifcr = build_ifcr_counter(df, timestamp_col="timestamp", event_col="event")
    print(df_ifcr)
