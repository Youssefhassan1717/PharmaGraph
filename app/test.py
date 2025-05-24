# app/test_run.py

from pipeline import run_pipeline

user_input = input("Please describe your drugs situation: ")
final_answer = run_pipeline(user_input)

print("\n\nFinal Response:\n", final_answer)
