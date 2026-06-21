from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd

# State initialization
if "scatterplots" not in st.session_state:
    st.session_state.scatterplots = []

# Chart clearing modal
@st.dialog(":material/warning: Clear All Charts?")
def clearing_modal():
    st.write("Are you sure you want to clear all charts?")
    if st.button("Yes, clear all charts"):
        st.session_state.scatterplots = []
        st.rerun()
        
    
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")

def make_slider(title: str, column_name : str, double_ended : bool):
    min = df[column_name].min()
    max = df[column_name].max()
    
    if double_ended:
        return st.slider(title, min_value = min, max_value = max, value = (min, max))
    else:
        return st.slider(title, min_value = min, max_value = max)

label_mapping = {
        "species" : "Species",
        "island" : "Island",
        "bill_length_mm" : "Bill Length (mm)",
        "bill_depth_mm" : "Bill Depth (mm)",
        "flipper_length_mm" : "Flipper Length (mm)",
        "body_mass_g" : "Body Mass (g)",
        "sex" : "Sex"
    }

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
        add_chart, filters, predict = st.tabs([":material/bubble_chart: Add Chart", ":material/filter_alt: Filter Charts", ":material/wand_stars: Predict"])

        with add_chart:            
            #populate dictionaries
            x =  st.selectbox("X-Axis", options = df.columns.values.tolist(), format_func = lambda col : label_mapping.get(col))
            y =  st.selectbox("Y-Axis", options = df.columns.values.tolist(), format_func = lambda col : label_mapping.get(col))
            
            if st.button(":material/add_circle: Add Chart"):
                chart = {"x": x, "y": y}
                
                if chart not in st.session_state.scatterplots:
                    st.session_state.scatterplots.append(chart)
                    st.toast(":material/done_outline: Chart added successfully!", duration = 2)
                else:
                    st.toast(":material/error: Selected chart has already been added.", duration = 2)
            
        with filters:
            st.header("Visualization Filters")
            # MULTI-SELECT BOXES ------------------------------------------------------------
            species = st.multiselect("Species", options = df.species.unique(), default = df.species.unique())
            island = st.multiselect("Island", options = df.island.unique(), default = df.island.unique())
            sex = st.multiselect("Sex", options = df.sex.unique(), default = df.sex.unique(), format_func = lambda x: str(x).capitalize())
            
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

data, visualize = st.tabs([":material/database: Raw Data", ":material/bubble_chart: Visualize & Predict"])
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
    charts, predict = st.columns(2, gap = "large")
    
    with charts:
        df_filtered = df[
            df["species"].isin(species) & 
            df["island"].isin(island) & 
            df["sex"].isin(sex) & 
            df["bill_length_mm"].between(*bill_length_mm) & 
            df["bill_depth_mm"].between(*bill_depth_mm) & 
            df["flipper_length_mm"].between(*flipper_length_mm) & 
            df["body_mass_g"].between(*body_mass_g)
        ]
        
        df_filtered_w_input = pd.concat([input_row, df_filtered], axis = 0)
        df_filtered_w_input.at[0, 'species'] = 'Input Point'
        
        
        # color_mapping = {
        #     "Adelie": "#FF6F61",
        #     "Chinstrap": "#6B5B95",
        #     "Gentoo": "#88B04B",
        #     "Input Point": "#09604B"
        # }
        # df_filtered_w_input["species_color"] = df_filtered_w_input["species"].map(color_mapping)
        
        for scatterplot in st.session_state.scatterplots:
            st.write(f"**{label_mapping[scatterplot["y"]]} vs. {label_mapping[scatterplot["x"]]}**")
            st.scatter_chart(data = df_filtered_w_input,
                             x = scatterplot["x"],
                             y = scatterplot["y"],
                             x_label = label_mapping[scatterplot["x"]],
                             y_label = label_mapping[scatterplot["y"]],
                             color = "species")
        
        if st.session_state.scatterplots: #if the list ISN'T empty, show the clear button
            if st.button(":material/delete_sweep: Clear All Charts"):
                clearing_modal()
        else:
            st.write(":primary-badge[Go to the 'Add Chart' tab in the sidepanel to add a chart.]")
            #st.image("chart_placeholder.png")
            
        # st.scatter_chart(data = df_filtered_w_input,
        #                 x = 'bill_length_mm',
        #                 y = 'bill_depth_mm',
        #                 x_label = "Bill Length (mm)",
        #                 y_label = "Bill Depth (mm)",
        #                 color = "species")
        
        # st.write("**Bill Length vs. Body Mass**")
        # st.scatter_chart(data = df_filtered_w_input,
        #                 x = 'bill_length_mm',
        #                 y = 'body_mass_g',
        #                 x_label = "Bill Length (mm)",
        #                 y_label = "Body Mass (g)",
        #                 color = 'species')
        
    with predict:
        with st.expander("**Input Features**"):
            input_col, encoded_input_col = st.columns(2)
            
            with input_col:
                st.write("Input Row")
                input_row.T
            
            with encoded_input_col:
                st.write("Encoded Input Row")
                encoded_df[:1].T
        
        prediction_display = reversed_mapper[prediction.item()]
        st.write(f"**Prediction:** :primary-badge[{prediction_display}]")
        st.markdown(
            f"""
            <div style="
                padding: 1rem;
                border-radius: 10px;
                background-color: #e8f5e9;
                text-align: center;
            ">
                <h3>Predicted Species</h3>
                <h1>🐧 {prediction_display}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
        # PROBABILITY CARDS
        st.write("**Prediction Probability**")
        prediction_probability_labeled = pd.DataFrame(
            prediction_probability, #NumPy array
            columns=[reversed_mapper[i] for i in range(3)] #what to name the columns ig
        )
        #df.species.apply(lambda species : target_mapper[species]) # applies this function to this column

        predicted_species = reversed_mapper[prediction.item()]

        metric_cols = st.columns(3) # = [col1, col2, col3]

        for col, species in zip(metric_cols, prediction_probability_labeled.columns):
            probability = prediction_probability_labeled[species].iloc[0]

            label = f":material/star: {species}" if species == predicted_species else species

            with col:
                st.metric(
                    label=label,
                    value=f"{probability:.1%}",
                    border=True
                )