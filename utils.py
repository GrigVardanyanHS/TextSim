import os
import pandas as pd

video_paths = {"recognized.xlsx",
                "recognizedSDLC.xlsx",
                "recognizedSFTTesting.xlsx"
}

def load_excels():
    data_frames = [(path, pd.read_excel(f"data//{path}")[["Duration","Recognized"]]) for path in video_paths]
    return data_frames