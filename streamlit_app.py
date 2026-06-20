from sklearn.ensemble import RandomForestClassifier
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

st.set_page_config(
    page_title="Penguin Explorer",
    page_icon=":material/owl:",
    layout="wide",
    initial_sidebar_state="expanded",
    
)

st.logo("logo.png")
st.image("penguin_banner.png")
st.title('Penguin Explorer')

with st.sidebar:
        explore, predict = st.tabs([":material/filter_alt: Explore", ":material/wand_stars: Predict"])
    
        with explore:
            st.header("Visualization Filters")
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
            st.header("Prediction Input Features")
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
            X = df.drop('species', axis = 1) # remove species so we don't have None for species in the input row when we
            concat_df = pd.concat([input_row, X], axis = 0)
            
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
            
            reversed_mapper = {v: k for k, v in target_mapper.items()}
        
            
            Y_encoded = df.species.apply(lambda species : target_mapper[species]) # applies this function to this column
            
            # Model Building (random forest classifier)
            model = RandomForestClassifier()
            model.fit(encoded_df[1:], Y_encoded) # fit on all rows except the last one (which is the input row)
            prediction = model.predict(encoded_input_row := encoded_df[:1]) # predict on the first row of the encoded df (which is the input row)
            prediction_probability = model.predict_proba(encoded_input_row)

data, visualize, results = st.tabs([":material/database: Raw Data", ":material/bubble_chart: Visualizations", ":material/owl: Prediction Results"])
with data:
    # Display data frame
    with st.expander('**Full Dataframe**'):
        df
    
    with st.expander('**Predictor Variables (X\'s)**'):
        X # Display the X variables 
    
    with st.expander('**Dependent Variable (Y)**'):
        Y = df.species
        Y # Display the Y variable
    
with visualize:
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
with results:
    with st.expander("**Input Features**"):
        st.write("Input Row")
        input_row
        
        st.write("Encoded Input Row")
        encoded_df[:1]
    
    prediction_display = reversed_mapper[prediction.item()]
    st.write(f"**Prediction:** :primary-badge[{prediction_display}]")
    st.write("**Prediction Probability**")
    prediction_probability_labeled = pd.DataFrame(
        prediction_probability, #NumPy array
        columns=[reversed_mapper[i] for i in range(3)] #what to name the columns ig
    )
    #df.species.apply(lambda species : target_mapper[species]) # applies this function to this column

    prediction_probability_labeled