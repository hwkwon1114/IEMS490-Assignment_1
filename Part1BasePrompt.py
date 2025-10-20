import utils
from langchain.prompts import ChatPromptTemplate
import time


def main():
    print("--- Running Setup ---")
    utils.setup_api_key()
    df_sampled = utils.data_sample()
    chat = utils.initialize_model()
    print("\n--- Setup Complete ---")
    print(f"Data loaded with shape: {df_sampled.shape}")
    template_string = """Read the following math question.
    Your task is to provide *only* the final numerical answer.
    Do not show your work. Do not use any text, units, or LaTeX.
    Format your answer on a single line, starting with the "####" prefix.

    Question: {question}
    Final Answer: """
    prompt_template = ChatPromptTemplate.from_template(template_string)
    results = []
    llm_answers = []
    num_q = 150
    REQUEST_DELAY_SECONDS = 4
    for i in range(num_q):
        prompt = prompt_template.format_messages(question = df_sampled.iloc[i,0])
        response = chat(prompt)
        answer = response.content
        llm_answers.append(answer)
        results.append(utils.parser(df_sampled.iloc[i,1])== utils.parser(answer))
        ## Added Delay because my gemini limits requests per minute
        time.sleep(REQUEST_DELAY_SECONDS)
    accuracy = sum(results) / len(results)
    print(f"\n--- Evaluation Complete ---")
    print(f"Total Correct: {sum(results)} / {len(results)}")
    print(f"Final Accuracy: {accuracy * 100:.2f}%")

    results_df = df_sampled.iloc[:num_q].copy()
    results_df['llm_answer'] = llm_answers
    results_df['is_correct'] = results
    results_df.to_csv("Vanilla_2.5.csv", index=False)

if __name__ == '__main__':
    main()