import streamlit as st
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")

def make_slider(title: str, column_name : str, double_ended : bool):
    min = df[column_name].min()
    max = df[column_name].max()
    
    if double_ended:
        return st.slider(title, min_value = min, max_value = max, value = (min, max))
    else:
        return st.slider(title, min_value = min, max_value = max)

st.title('Penguin Explorer')

with st.sidebar:
        explore, predict = st.tabs([":material/filter_alt: Explore", ":material/wand_stars: Predict"])
    
        with explore:
            st.header("Explore Data")
            # MULTI-SELECT BOXES ------------------------------------------------------------
            species = st.multiselect("Species", options = df.species.unique())
            island = st.multiselect("Island", options = df.island.unique())
            sex = st.multiselect("Sex", options = df.sex.unique(), format_func = lambda x: str(x).capitalize())
            
            # DOUBLE-ENDED SLIDERS -----------------------------------------------------------------
            
            bill_length_mm = make_slider("Bill Length (mm)", "bill_length_mm", True)
            bill_depth_mm = make_slider("Bill Depth (mm)", "bill_depth_mm", True)
            flipper_length_mm = make_slider("Flipper Length (mm)", "flipper_length_mm", True)
            body_mass_g = make_slider("Body Mass (g)", "body_mass_g", True)
                    
        with predict:
            st.header("Predict Species")
            # "island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"
            
            # SELECT BOXES ------------------------------------------------------------        
            p_island = st.selectbox("Island", options = df.island.unique())
            p_sex = st.selectbox("Sex", options = df.sex.unique(), format_func = lambda x: str(x).capitalize())
                    
            # SINGLE SLIDERS -----------------------------------------------------------------
            p_bill_length_mm = make_slider("Bill Length (mm)", "bill_length_mm", False)
            p_bill_depth_mm = make_slider("Bill Depth (mm)", "bill_depth_mm", False)
            p_flipper_length_mm = make_slider("Flipper Length (mm)", "flipper_length_mm", False)
            p_body_mass_g = make_slider("Body Mass (g)", "body_mass_g", False)
            
            # Create a dataframe for the input features
            prediction_data = {
                "island": p_island,
                "sex": p_sex,
                "bill_length_mm": p_bill_length_mm,
                "bill_depth_mm": p_bill_depth_mm,
                "flipper_length_mm": p_flipper_length_mm,
                "body_mass_g": p_body_mass_g
            }
            
            input_row = pd.DataFrame(prediction_data, index = [0])
            
            # you gotta be really careful with how you concatentate thing b/c it is funky (and if you column names don't match perfectly weird stuff will happen)
            concat_df = pd.concat([input_row, df], axis = 0)
            
            # pd.get_dummies() converts categorical variables into dummy/indicator variables
            # through a process called one-hot encoding. It breaks a single column containing
            # multiple categories into several binary cols filled with 1's/Trues and 0's/Falses
            
            # Encode X
            encoded_df = pd.get_dummies(concat_df, columns = ["island", "sex"], prefix = ["Island", "Sex"])
            
            # Encode Y
            target_mapper = {
                "Adelie": 0,
                "Chinstrap": 1,
                "Gentoo": 2
            }
            
            # Encoding function (w/ type hints!)
            def target_encode(species : str) -> int:
                return target_mapper[species]
            
            y_encoded = df.species.apply(target_encode) # applies this function to this column
            
with st.expander("Raw Data"):
    # Display data frame
    df
    
    st.write('**X**')
    X = df.drop('species', axis = 1)
    X # Display the X variables 
    
    st.write('**Y**')
    Y = df.species
    Y # Display the Y variable
    
with st.expander("Data Visualization"):
    st.write("**Bill Length vs. Bill Depth**")
    st.scatter_chart(data = df,
                     x = 'bill_length_mm',
                     y = 'bill_depth_mm',
                     x_label = "Bill Length (mm)",
                     y_label = "Bill Depth (mm)",
                     color = 'species')
    
    st.write("**Bill Length vs. Body Mass**")
    st.scatter_chart(data = df,
                     x = 'bill_length_mm',
                     y = 'body_mass_g',
                     x_label = "Bill Length (mm)",
                     y_label = "Body Mass (g)",
                     color = 'species')
with st.expander("Predictions"):
    st.write("**Input Features**")
    st.write("Input row")
    input_row
    
    st.write("Encoded input row")
    encoded_df[:1]