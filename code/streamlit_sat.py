import ast

import streamlit as st
import pandas as pd
import json


# Load data function with cache
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data


# Function to display details of each problem entry
def display_problem_details(problem_data):
    try:
        problem_dict = ast.literal_eval(problem_data)
    except (ValueError, SyntaxError):
        st.error("Invalid format in problem data. Unable to parse.")
        return

    question = problem_dict.get('question', 'N/A')
    choices = problem_dict.get('choices', [])
    answer = problem_dict.get('answer', 'N/A')
    st.write(f"**Question:** {question}")
    st.write(f"**Choices:** {choices}")
    st.write(f"**Answer:** {answer}")


# Helper function to calculate and plot length distributions
def plot_length_distribution(data, column_name, description, is_json=False, json_key=None):
    if is_json and json_key:
        lengths = data[column_name].apply(lambda x: len(json.loads(x).get(json_key, '')))
    else:
        lengths = data[column_name].apply(len)
    st.write(f"#### {description} Length Distribution")
    st.bar_chart(lengths.value_counts().sort_index())


# Main Streamlit app
def main():
    st.title("Korean CSAT Language Section EDA")

    # File uploader for user-uploaded data
    file = st.file_uploader("Upload your dataset CSV file", type=["csv"])

    if file is not None:
        # Load the uploaded data
        data = load_data(file)

        st.write("### Dataset Preview")
        st.dataframe(data.head())

        # Basic dataset information
        st.write("### Dataset Information")
        st.write(f"Number of entries: {len(data)}")
        st.write(f"Columns: {data.columns.tolist()}")

        # Display samples of each field
        st.write("### Field Analysis")

        # Display a sample paragraph
        st.subheader("Sample Paragraph")
        st.write(data['paragraph'].iloc[0])

        # Display a sample problem with details
        st.subheader("Sample Problem")
        display_problem_details(data['problems'].iloc[0])

        # Display a sample 'question_plus' entry
        st.subheader("Sample Question Plus")
        st.write(data['question_plus'].iloc[0])

        # Additional statistics
        st.write("### Additional Statistics")

        # Plot length distributions for each field
        plot_length_distribution(data, 'paragraph', 'Paragraph')
        plot_length_distribution(data, 'problems', 'Question', is_json=True, json_key='question')
        plot_length_distribution(data, 'problems', 'Choices', is_json=True, json_key='choices')
        plot_length_distribution(data, 'question_plus', 'Question Plus')

        # Answer count analysis
        st.write("#### Answer Distribution")
        answer_counts = data['problems'].apply(lambda x: json.loads(x)['answer']).value_counts()
        st.bar_chart(answer_counts.sort_index())


if __name__ == "__main__":
    main()
