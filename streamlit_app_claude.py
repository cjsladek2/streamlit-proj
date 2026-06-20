from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd

# ----------------------------------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit command)
# ----------------------------------------------------------------------------------
st.set_page_config(
    page_title="Penguin Explorer",
    page_icon=":material/owl:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------------
# GLOBAL STYLES
# ----------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Tighten default top padding now that we have a full-width hero */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 2rem;
    }

    /* Full-bleed hero banner that breaks out of the centered column */
    .st-key-hero {
        width: 100vw;
        margin-left: calc(-50vw + 50%);
        margin-right: calc(-50vw + 50%);
        margin-top: 0;
        position: relative;
    }
    .st-key-hero img {
        width: 100%;
        height: 300px;
        object-fit: cover;
        display: block;
        border-radius: 0;
        filter: brightness(0.75);
    }
    .hero-title {
        position: absolute;
        bottom: 28px;
        left: 5%;
        color: white;
        text-shadow: 0 2px 12px rgba(0,0,0,0.6);
    }
    .hero-title h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: 800;
    }
    .hero-title p {
        font-size: 1.1rem;
        margin: 4px 0 0 0;
        opacity: 0.9;
    }

    /* Card-style metric containers */
    .st-key-metric_card {
        background: var(--background-color, #f8f9fa);
        border: 1px solid rgba(49, 51, 63, 0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    /* Section header spacing */
    .section-header {
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------------
# DATA LOADING (cached so we don't re-fetch on every interaction)
# ----------------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        "https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv"
    )

df = load_data()


def make_slider(title: str, column_name: str, double_ended: bool, key: str):
    col_min = df[column_name].min()
    col_max = df[column_name].max()

    if double_ended:
        return st.slider(title, min_value=col_min, max_value=col_max, value=(col_min, col_max), key=key)
    else:
        return st.slider(title, min_value=col_min, max_value=col_max, key=key)


# ----------------------------------------------------------------------------------
# HERO BANNER
# ----------------------------------------------------------------------------------
with st.container(key="hero"):
    st.image("penguin_banner.png", use_container_width=True)
    st.markdown(
        """
        <div class="hero-title">
            <h1>🐧 Penguin Explorer</h1>
            <p>Explore, visualize, and predict penguin species from the Palmer Archipelago dataset</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")  # small breathing room below the hero

# ----------------------------------------------------------------------------------
# SIDEBAR — FILTERS & PREDICTION INPUTS
# ----------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### :material/owl: Penguin Explorer")
    st.caption("Use the tabs below to filter the dataset or predict a species.")

    explore, predict = st.tabs([":material/filter_alt: Explore", ":material/wand_stars: Predict"])

    with explore:
        st.markdown("**Visualization Filters**")

        species = st.multiselect("Species", options=df.species.unique(), key="f_species")
        island = st.multiselect("Island", options=df.island.unique(), key="f_island")
        sex = st.multiselect(
            "Sex", options=df.sex.unique(), format_func=lambda x: str(x).capitalize(), key="f_sex"
        )

        with st.expander("Measurement ranges", expanded=False):
            bill_length_mm = make_slider("Bill Length (mm)", "bill_length_mm", True, "f_bill_len")
            bill_depth_mm = make_slider("Bill Depth (mm)", "bill_depth_mm", True, "f_bill_depth")
            flipper_length_mm = make_slider("Flipper Length (mm)", "flipper_length_mm", True, "f_flipper")
            body_mass_g = make_slider("Body Mass (g)", "body_mass_g", True, "f_mass")

        if st.button("Reset filters", use_container_width=True):
            st.rerun()

    with predict:
        st.markdown("**Prediction Input Features**")

        p_island = st.selectbox("Island", options=df.island.unique(), key="p_island")
        p_sex = st.selectbox(
            "Sex", options=df.sex.unique(), format_func=lambda x: str(x).capitalize(), key="p_sex"
        )

        p_bill_length_mm = make_slider("Bill Length (mm)", "bill_length_mm", False, "p_bill_len")
        p_bill_depth_mm = make_slider("Bill Depth (mm)", "bill_depth_mm", False, "p_bill_depth")
        p_flipper_length_mm = make_slider("Flipper Length (mm)", "flipper_length_mm", False, "p_flipper")
        p_body_mass_g = make_slider("Body Mass (g)", "body_mass_g", False, "p_mass")

        # Build the single-row input frame for prediction
        prediction_data = {
            "island": p_island,
            "sex": p_sex,
            "bill_length_mm": p_bill_length_mm,
            "bill_depth_mm": p_bill_depth_mm,
            "flipper_length_mm": p_flipper_length_mm,
            "body_mass_g": p_body_mass_g,
        }
        input_row = pd.DataFrame(prediction_data, index=[0])

        # Careful with concatenation — column names must match exactly
        X = df.drop("species", axis=1)
        concat_df = pd.concat([input_row, X], axis=0)

        # One-hot encode categorical columns
        encoded_df = pd.get_dummies(concat_df, columns=["island", "sex"], prefix=["Island", "Sex"])

        target_mapper = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}
        reversed_mapper = {v: k for k, v in target_mapper.items()}

        Y_encoded = df.species.apply(lambda s: target_mapper[s])

        model = RandomForestClassifier(random_state=42)
        model.fit(encoded_df[1:], Y_encoded)
        encoded_input_row = encoded_df[:1]
        prediction = model.predict(encoded_input_row)
        prediction_probability = model.predict_proba(encoded_input_row)

# ----------------------------------------------------------------------------------
# MAIN CONTENT
# ----------------------------------------------------------------------------------

# Apply explore-tab filters to the dataframe for the Data + Visualize tabs
filtered_df = df.copy()
if species:
    filtered_df = filtered_df[filtered_df.species.isin(species)]
if island:
    filtered_df = filtered_df[filtered_df.island.isin(island)]
if sex:
    filtered_df = filtered_df[filtered_df.sex.isin(sex)]
filtered_df = filtered_df[
    filtered_df.bill_length_mm.between(*bill_length_mm)
    & filtered_df.bill_depth_mm.between(*bill_depth_mm)
    & filtered_df.flipper_length_mm.between(*flipper_length_mm)
    & filtered_df.body_mass_g.between(*body_mass_g)
]

# Quick stat row
m1, m2, m3, m4 = st.columns(4)
with m1:
    with st.container(key="metric_card", border=False):
        st.metric("Penguins shown", len(filtered_df), border=False)
with m2:
    st.metric("Total in dataset", len(df))
with m3:
    st.metric("Species", filtered_df.species.nunique())
with m4:
    st.metric("Islands", filtered_df.island.nunique())

st.divider()

data_tab, visualize_tab, results_tab = st.tabs(
    [":material/database: Raw Data", ":material/bubble_chart: Visualizations", ":material/owl: Prediction Results"]
)

with data_tab:
    st.markdown('<p class="section-header"><strong>Filtered Dataset</strong></p>', unsafe_allow_html=True)
    st.caption("Reflects the filters set in the sidebar's Explore tab.")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        with st.expander("**Predictor Variables (X)**"):
            st.dataframe(X, use_container_width=True, hide_index=True)
    with col_b:
        with st.expander("**Dependent Variable (Y)**"):
            Y = df.species
            st.dataframe(Y, use_container_width=True, hide_index=True)

with visualize_tab:
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Bill Length vs. Bill Depth**")
        st.scatter_chart(
            data=filtered_df,
            x="bill_length_mm",
            y="bill_depth_mm",
            x_label="Bill Length (mm)",
            y_label="Bill Depth (mm)",
            color="species",
            use_container_width=True,
        )

    with chart_col2:
        st.markdown("**Bill Length vs. Body Mass**")
        st.scatter_chart(
            data=filtered_df,
            x="bill_length_mm",
            y="body_mass_g",
            x_label="Bill Length (mm)",
            y_label="Body Mass (g)",
            color="species",
            use_container_width=True,
        )

    st.markdown("**Flipper Length vs. Body Mass**")
    st.scatter_chart(
        data=filtered_df,
        x="flipper_length_mm",
        y="body_mass_g",
        x_label="Flipper Length (mm)",
        y_label="Body Mass (g)",
        color="species",
        use_container_width=True,
    )

with results_tab:
    prediction_display = reversed_mapper[prediction.item()]

    pred_col1, pred_col2 = st.columns([1, 1.4])

    with pred_col1:
        st.markdown("**Predicted Species**")
        st.write(f"### :primary-badge[{prediction_display}]")

        st.markdown("**Prediction Probability**")
        prediction_probability_labeled = pd.DataFrame(
            prediction_probability,
            columns=[reversed_mapper[i] for i in range(3)],
        )
        st.bar_chart(prediction_probability_labeled.T, use_container_width=True)

    with pred_col2:
        with st.expander("**Input Features**", expanded=True):
            st.markdown("Input Row")
            st.dataframe(input_row, use_container_width=True, hide_index=True)

            st.markdown("Encoded Input Row")
            st.dataframe(encoded_input_row, use_container_width=True, hide_index=True)