import os
import pandas as pd

video_paths = {#"recognized.xlsx",
                #"recognizedSDLC.xlsx",
                #"recognizedSFTTesting.xlsx",
                "MantesaDEMO1.xlsx",
                #"MantesaDEMO2.xlsx"
                #"DLP.xlsx",
                #"Lucy _ Logo Detector _ POC.xlsx"
                }

def load_and_restructure(path):
    df = pd.read_excel(path)[["Duration","Recognized"]]
    texts = []
    period = []
    
    for i in range(len(df["Recognized"].values) -1):
        texts.append(df["Recognized"].values[i] + " " + df["Recognized"].values[i+1])
        period.append(f"{i*10}-{i*10+20}")

    df = pd.DataFrame({"Recognized":texts,"Duration": period})
    return df

def load_excels():
    #data_frames = [(path, load_and_restructure(f"data//{path}")[["Duration","Recognized"]]) for path in video_paths]
    data_frames = [(path, pd.read_excel(f"data//{path}")[["Duration","Recognized"]]) for path in video_paths]
    return data_frames

def save_file(file_name, content):
    file_name_txt = file_name.split(".")[0]

    with open(f"files//{file_name_txt}.txt","w+") as f:
        f.write(content)
    
