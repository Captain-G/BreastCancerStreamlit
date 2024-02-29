import streamlit as st
import pickle as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def get_clean_data():
    data = pd.read_csv("/home/gachuki/PycharmProjects/BreastCancerStreamlit/data/data.csv")
    data = data.drop(["Unnamed: 32", "id"], axis=1)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

    return data


def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict


def get_scaled_values(input_dict):
    data = get_clean_data()

    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict


def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area',
                  'Smoothness', 'Compactness',
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        height=600,
        width=800
    )

    return fig


def add_top_navbar():
    st.markdown("""
            <style>
                body {
                    font-family: 'Arial', sans-serif;
                    margin: 0;
                    padding: 0;
                }

                .navbar {
                    overflow: hidden;
                    width: 100%;
                }

                .navbar a {
                    float: left;
                    display: block;
                    color: white;
                    text-align: center;
                    padding: 14px 16px;
                    text-decoration: none;
                }

                .navbar a:hover {
                    background-color: #ddd;
                    color: black;
                }

                .navbar a.active {
                    background-color: #0096FF;
                    color: white;
                }
            </style>

        """, unsafe_allow_html=True)

    # Navigation bar
    st.markdown("""
            <div class="navbar">
                <a href="#">Breast Cancer</a>
                <a href="#">Lung Cancer</a>
                <a href="#">Leukemia</a>
                <a href="#">Prostate Cancer</a>
                <a href="#">Pancreatic Cancer</a>
            </div>
        """, unsafe_allow_html=True)


def add_predictions(input_data):
    model = pickle.load(open("/model/breast_cancer_model.pkl", "rb"))
    scaler = pickle.load(open("/model/breast_cancer_scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)
    st.subheader("Cell cluster prediction is:")
    if prediction[0] == 0:
        st.markdown("""
                    <style>
                       .benign {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        margin: auto;
                        padding: 10px;
                        background-color: #0096FF;
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                        margin: 10px,
                        # text-align: center;  /* Optional: Align text within the container */
}

                    </style>
                """, unsafe_allow_html=True)

        st.markdown('<div class="benign">Benign</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
                    <style>
                        .malignant {
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                            justify-content: center;
                            margin: auto;
                            padding: 10px;
                            background-color: #880808;
                            border: 1px solid #ddd;
                            border-radius: 5px;
                            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                            text-align: center;  /* Optional: Align text within the container */
                        }

                    </style>
                """, unsafe_allow_html=True)

        st.markdown('<div class="malignant">Malignant</div>', unsafe_allow_html=True)
    st.write("Probability of being benign: ")
    st.markdown("""
                        <style>
                            .probabilities {
                                display: flex;
                                flex-direction: column;
                                align-items: center;
                                justify-content: center;
                                margin: auto;
                                padding: 10px;
                                border: 1px solid #ddd;
                                border-radius: 5px;
                                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                                text-align: center;  /* Optional: Align text within the container */
                            }

                        </style>
                    """, unsafe_allow_html=True)

    st.markdown(f'<div class="probabilities">{round(model.predict_proba(input_array_scaled)[0][0], 4) * 100}%</div>',
                unsafe_allow_html=True)
    st.write("Probability of being malignant: ")
    st.markdown(f'<div class="probabilities">{round(model.predict_proba(input_array_scaled)[0][1], 4) * 100}%</div>',
                unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":microscope:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    add_top_navbar()

    st.markdown("""
        <style>
            @import url('/home/gachuki/PycharmProjects/BreastCancerStreamlit/assets/style.css');
        </style>
    """, unsafe_allow_html=True)

    input_data = add_sidebar()

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write(
            "Please connect this app to your cytology lab to help diagnose breast cancer from cell tissue, You can also update the values using the sliders in the sidebar")

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)


if __name__ == "__main__":
    main()
