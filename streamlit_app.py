import streamlit as st
import pandas as pd

st.title('🎈 Charlotte\'s Dope App')

st.write('Hello world!')

df = pd.read_csv("https://github.com/dataprofessor/data/blob/master/penguins_cleaned.csv")

df