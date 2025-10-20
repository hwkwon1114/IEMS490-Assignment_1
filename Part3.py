import utils
from langchain.prompts import ChatPromptTemplate
import time
import pandas as pd
# --- CHANGE: Added new imports for structured output parsing ---
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

def run_evaluation(prompt_template, chat_model, dataframe, description="evaluation"):
    num_questions = len(dataframe)
    if num_questions == 0:
        return 0.0, pd.DataFrame()

    results_df = dataframe.copy()
    results_df['llm_answer'] = ""
    results_df['model_parsed'] = None
    results_df['correct_parsed'] = None
    results_df['is_correct'] = False
    
    REQUEST_DELAY_SECONDS = 4

    print(f"--- Running {description} for a prompt on {num_questions} questions ---")

    for i in range(num_questions):
        try:
            # Using .loc with index is safer after sampling/slicing
            current_index = results_df.index[i]
            
            question = results_df.loc[current_index, 'question']
            ground_truth_full = results_df.loc[current_index, 'answer']
            
            correct_answer = utils.parser(ground_truth_full)
            results_df.loc[current_index, 'correct_parsed'] = correct_answer

            prompt = prompt_template.format_messages(question=question)
            response = chat_model.invoke(prompt) # Use .invoke for clarity
            answer_content = response.content
            results_df.loc[current_index, 'llm_answer'] = answer_content
            
            model_answer = utils.parser(answer_content)
            results_df.loc[current_index, 'model_parsed'] = model_answer

            is_correct = (correct_answer == model_answer) and (correct_answer is not None)
            results_df.loc[current_index, 'is_correct'] = is_correct
            
            time.sleep(REQUEST_DELAY_SECONDS)

        except Exception as e:
            print(f"\n!! ERROR on question {i+1}: {e}")
            results_df.loc[current_index, 'is_correct'] = False
            continue
    
    accuracy = results_df['is_correct'].sum() / num_questions if num_questions > 0 else 0.0
    failures_df = results_df[results_df['is_correct'] == False]

    return accuracy, failures_df

def main():
    """
    Main function to run the automated "Targeted Failure Correction" experiment.
    """
    # --- 1. SETUP ---
    print("--- Running Setup ---")
    utils.setup_api_key()
    # This ensures our optimization and test sets are completely separate.
    optimization_set, test_set = utils.load_data_for_part3(50, 150)
    chat_model = utils.initialize_model()
    print("\n--- Setup Complete ---")

    # --- 2. INITIAL STATE (GENERATION 0) ---
    current_best_prompt_str = """You are a helpful math assistant.
1. Solve the following question by showing your reasoning step-by-step.
2. After all your reasoning, provide the final numerical answer on a *new line*, prefixed with "####".
Question: {question}
Answer: Let's think step by step.
"""
    
    prompt_template = ChatPromptTemplate.from_template(current_best_prompt_str)
    # The optimization loop runs on the 'optimization_set'
    current_best_score, failures_df = run_evaluation(prompt_template, chat_model, optimization_set, "Full Initial Evaluation")
    
    print(f"Generation 0 | Initial Score: {current_best_score * 100:.2f}% | Failures: {len(failures_df)}")

    # --- 3. THE OPTIMIZATION LOOP ---
    num_generations = 5
    for gen in range(num_generations):
        print(f"\n--- Starting Generation {gen + 1}/{num_generations} ---")

        if failures_df.empty:
            print("No failures found. Ending optimization early.")
            break

        failed_examples_str = failures_df[['question', 'answer', 'llm_answer']].to_string()
        
        # --- CHANGE: Define the structured output schema ---
        new_prompt_schema = ResponseSchema(name="new_prompt",
                                           description="The full, raw text of the new, improved prompt.")
        response_schemas = [new_prompt_schema]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        # --- CHANGE: Update the meta-prompt to include format instructions ---
        meta_prompt_template_str = """You are an expert prompt engineer. Your task is to analyze failed math problems and rewrite a prompt to fix them.

--- FAILURE ANALYSIS REPORT ---
{failed_examples}
--- END REPORT ---

Based on your analysis of the `llm_answer` column in the report, rewrite the Original Prompt to prevent these specific errors.
Your output MUST be a JSON object that follows the specified format.

{format_instructions}

Original Prompt:
---
{original_prompt}
---
"""
        meta_prompt = ChatPromptTemplate.from_template(meta_prompt_template_str)
        mutation_prompt = meta_prompt.format_messages(
            failed_examples=failed_examples_str,
            original_prompt=current_best_prompt_str,
            format_instructions=format_instructions
        )
        
        print("Asking LLM to generate an improved prompt in a structured format...")
        response = chat_model.invoke(mutation_prompt)
        
        try:
            # --- CHANGE: Parse the structured output ---
            parsed_output = output_parser.parse(response.content)
            new_candidate_prompt_str = parsed_output['new_prompt']
            print("Successfully parsed new candidate prompt.")
        except Exception as e:
            print(f"ERROR: Could not parse LLM output. Skipping this generation. Error: {e}")
            continue

        candidate_template = ChatPromptTemplate.from_template(new_candidate_prompt_str)

        print(f"Pre-screening new prompt on {len(failures_df)} known failures...")
        targeted_accuracy, _ = run_evaluation(candidate_template, chat_model, failures_df, "Targeted Re-Test")

        if targeted_accuracy > 0:
            print(f"Candidate fixed {targeted_accuracy*100:.2f}% of failures. Promoting to full regression test.")
            
            full_accuracy, new_failures = run_evaluation(candidate_template, chat_model, optimization_set, "Full Regression Test")
            
            if full_accuracy > current_best_score:
                print(f"IMPROVEMENT FOUND! New Score: {full_accuracy*100:.2f}% | Old Score: {current_best_score*100:.2f}%")
                print(new_candidate_prompt_str)
                current_best_score = full_accuracy
                current_best_prompt_str = new_candidate_prompt_str
                failures_df = new_failures
            else:
                print(f"No improvement in full test ({full_accuracy*100:.2f}%). Using new failure data to guide next generation.")
                failures_df = new_failures
        else:
            print("Candidate failed to fix any known issues. Discarding.")

    # --- 4. FINAL TEST ---
    print("\n--- Automated Optimization Complete ---")
    print(f"Final Optimized Prompt:\n{current_best_prompt_str}")
    
    print("\n--- Running Final Evaluation on the Held-Out Test Set ---")
    final_prompt_template = ChatPromptTemplate.from_template(current_best_prompt_str)
    # The final evaluation runs on the separate 'test_set'
    final_accuracy, final_results_df = run_evaluation(final_prompt_template, chat_model, test_set, "Final Test")
    
    print(f"\nFinal Accuracy on Test Set: {final_accuracy * 100:.2f}%")
    
    final_results_df.to_csv("Part3.csv", index=False)
    print("Final results saved to auto_prompt_results.csv")
    
    best_prompt_filename = "best_prompt1.txt"
    with open(best_prompt_filename, "w") as f:
        f.write(current_best_prompt_str)
    print(f"Final optimized prompt saved to {best_prompt_filename}")


if __name__ == '__main__':
    main()

