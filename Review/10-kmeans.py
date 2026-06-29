from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine(as_frame=True)
df = wine.frame

X = df[["alcohol", "color_intensity"]]

