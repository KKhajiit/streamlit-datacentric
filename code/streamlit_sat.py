from ast import literal_eval
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd


class CSVData:
    def __init__(self, file, flatten_condition=False):
        """
        Initialize the CSVData object.
        Args:
            file: Uploaded file object (e.g., from Streamlit file uploader).
            flatten_condition: Whether to apply flatten_json to csv file
        """
        self.filename = file.name  # File name
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
    st.write(f"#### {title} Answer Distribution")
    answer_counts = df['answer'].value_counts()
    answer_counts = answer_counts.reindex(range(1, 6), fill_value=0)
    st.bar_chart(answer_counts.sort_index())

def get_comparison_data(train_data, output_data_list):
    results = []
    train_df = train_data.data
    output_dfs = [output_data.data for output_data in output_data_list]
    for problem_id in train_df["id"]:
        record = train_df[train_df["id"] == problem_id]

        # 각 파일의 예측 값 비교
        predictions = [df[df["id"].astype(str) == problem_id]["answer"] for df in output_dfs]
        statuses = [int(pred.astype(str) == record["answer"].astype(str)) for pred in predictions]

        if sum(statuses) == len(output_dfs):  # 모두 정답
            category = "Both Correct"
        elif sum(statuses) == 0:  # 모두 오답
            category = "Both Incorrect"
        else:  # 일부 정답
            category = "One Correct"

        results.append({
            "id": problem_id,
            "category": category,
            "paragraph": record["paragraph"],
            "question": record["question"],
            "choices": record["choices"],
            "correct_answer": record["answer"],
            "predictions": predictions,
        })
    return pd.DataFrame(results)

### Load Train and Output data ###

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

##################################


if train_data:
    st.write("### Train Data EDA")

    display_answer_distribution(train_data.filename, train_data.data)

if len(output_data_list) > 0:
    st.write("### Output Data EDA")

    for output_data in output_data_list:
        display_answer_distribution(output_data.filename, output_data.data)


if train_data and len(output_data_list) > 0:

    # 문제별 비교 데이터 준비
    comparison_df = get_comparison_data(train_data, output_data_list)

    # Sidebar: 문제 번호 선택
    category_filter = st.sidebar.selectbox("Filter by category", ["Both Correct", "Both Incorrect", "One Correct"])
    filtered_df = comparison_df[comparison_df["category"] == category_filter]
    problem_id = st.sidebar.selectbox("Select a problem ID", filtered_df["id"])

    # Main: 문제 정보 및 예측 값 표시
    if problem_id:
        problem_data = filtered_df[filtered_df["id"] == problem_id].iloc[0]
        st.write(f"**Paragraph:** {problem_data['paragraph']}")
        st.write(f"**Question:** {problem_data['question']}")
        st.write(f"**Choices:** {problem_data['choices']}")
        st.write(f"**Correct Answer:** {problem_data['correct_answer']}")

        st.write("### Predictions")
        for i, pred in enumerate(problem_data["predictions"]):
            st.write(f"{output_data_list[i].filename}: {pred}")
