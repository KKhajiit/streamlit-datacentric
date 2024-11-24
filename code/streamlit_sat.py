import streamlit as st
import pandas as pd
from ast import literal_eval

class CSVData:
    def __init__(self, file, flatten_condition=False):
        """
        Initialize the CSVData object.
        Args:
            file: Uploaded file object (e.g., from Streamlit file uploader).
            flatten_condition: Whether to apply flatten_json to csv file
        """
        self.filename = (file.name).replace(".csv", "")  # File name
        self.data = pd.read_csv(file)  # Data loaded as DataFrame
        if flatten_condition:
            self.data = self._flatten_json(self.data)

    def _flatten_json(self, df):
        records = []
        for _, row in df.iterrows():
            problems = literal_eval(row['problems'])
            record = {
                'id': row['id'],
                'paragraph': row['paragraph'],
                'question': problems['question'],
                'choices': problems['choices'],
                'answer': problems.get('answer', None),
                "question_plus": problems.get('question_plus', None),
            }
            records.append(record)
        return pd.DataFrame(records)


def display_answer_distribution(title, df):
    st.write(f"#### {title}")
    answer_counts = df['answer'].value_counts()
    answer_counts = answer_counts.reindex(range(1, 6), fill_value=0)
    st.bar_chart(answer_counts.sort_index())


def get_comparison_data(train_data, output_data_list):
    results = []
    train_df = train_data.data
    output_dfs = [output_data.data for output_data in output_data_list]
    output_file_names = [output_data.filename for output_data in output_data_list]

    for problem_id in train_df["id"]:
        record = train_df[train_df["id"] == problem_id]

        # 각 파일의 예측 값 비교
        predictions = [df[df["id"] == problem_id]["answer"] for df in output_dfs]
        if predictions[0].empty:
            continue
        statuses = [int(pred.values[0] == record["answer"].values[0]) for pred in predictions]

        # 카테고리 생성
        correct_files = [output_file_names[i] for i, status in enumerate(statuses) if status == 1]
        incorrect_files = [output_file_names[i] for i, status in enumerate(statuses) if status == 0]

        category = ""
        if len(correct_files) == len(output_file_names):  # 모든 파일이 정답
            category = "모든 output_file이 맞춤"
        elif len(incorrect_files) == len(output_file_names):  # 모든 파일이 오답
            category = "모든 output_file이 틀림"
        else:  # 특정 파일만 정답
            category = ", ".join(f"{file}만 맞춤" for file in correct_files)

        results.append({
            "id": problem_id,
            "category": category,
            "paragraph": record["paragraph"].values[0],
            "question": record["question"].values[0],
            "choices": record["choices"].values[0],
            "correct_answer": record["answer"].values[0],
            "predictions": [pred.values[0] for pred in predictions],
        })
    return pd.DataFrame(results)

tabs = st.tabs(["Upload Files", "Answer Distribution", "Output Comparison"])

### File Upload ###
with tabs[0]:
    train_file = st.file_uploader("Upload your dataset CSV file", type=["csv"])
    train_data = None
    if train_file:
        train_data = CSVData(train_file, flatten_condition=True)

    output_files = st.file_uploader(
        "Upload your output CSV files (multiple allowed)", type=["csv"], accept_multiple_files=True
    )
    output_data_list = []
    if output_files:
        for output_file in output_files:
            output_data = CSVData(output_file, flatten_condition=False)
            output_data_list.append(output_data)

### Tabs ###
if train_data or len(output_data_list) > 0:
    # Tab 1: Answer Distribution
    with tabs[1]:
        if train_data:
            st.write("### Train Data Answer Distribution")
            display_answer_distribution(train_data.filename, train_data.data)

        st.write("---")

        if len(output_data_list) > 0:
            st.write(f"### Output Data Answer Distribution")
            for output_data in output_data_list:
                display_answer_distribution(output_data.filename, output_data.data)

    # Tab 2: Output Comparison
    with tabs[2]:
        if train_data and len(output_data_list) > 0:
            comparison_df = get_comparison_data(train_data, output_data_list)

            # Sidebar: Filter options
            category_filter_options = comparison_df["category"].unique()
            category_filter = st.sidebar.selectbox("Filter by category", category_filter_options)

            filtered_df = comparison_df[comparison_df["category"] == category_filter]

            problem_id = st.sidebar.selectbox("Select a problem ID", filtered_df["id"])

            if problem_id:
                problem_data = filtered_df[filtered_df["id"] == problem_id].iloc[0]
                st.write(f"**Paragraph:** {problem_data['paragraph']}")
                st.write(f"**Question:** {problem_data['question']}")
                st.write(f"**Choices:** {problem_data['choices']}")
                st.write(f"**Correct Answer:** {problem_data['correct_answer']}")

                st.write("### Predictions")
                for i, pred in enumerate(problem_data["predictions"]):
                    st.write(f"**{output_data_list[i].filename}: {pred}**")