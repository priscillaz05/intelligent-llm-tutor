import google.generativeai as genai
from datasets import load_dataset
import json
import time
import ast
import re

if __name__ == "__main__":
    api_key = ""
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model_name="gemini-2.0-flash")

    with open('confusion_generation_prompt.txt', 'r') as file:
        confusion_prompt = file.read()

    with open('help_generation_prompt.txt', 'r') as file:
        help_prompt = file.read()

    ds = load_dataset("BAAI/TACO", split="train")

    eval_samples = []


    # generate confusions

    count = 0

    for sample in ds:
        try:
            solutions = json.loads(sample["solutions"])
            input_output = json.loads(sample["input_output"])

            if not solutions:  # skip if no solutions
                print("No solution")
                continue

            problem = sample["question"]
            solution = solutions[0]
            tags = eval(sample["tags"])
            skill_types = eval(sample["skill_types"])

            confusion_generation_prompt = f"""{confusion_prompt}. 

            Plese generate questions as a list of strings.

            Coding question:
            {problem}

            Question tags:
            {tags}

            Skill types:
            {skill_types}
            """

            raw_confusions = model.generate_content(confusion_generation_prompt)
            print(raw_confusions.text)


            # format confusions into list of strings

            match = re.search(r'questions\s*=\s*(\[.*\])', raw_confusions.text, re.DOTALL)

            if match:
                list_str = match.group(1)
                try:
                    confusions_list = ast.literal_eval(list_str)
                    print(confusions_list)
                except Exception as e:
                    print("Error parsing list:", e)
            else:
                print("List not found in model output for confusions.")

            entry = {
            "problem": problem,
            "solution": solution,
            "input_output": input_output,
            "tags": tags,
            "skill_types": skill_types,
            }

            entry["confusions"] = confusions_list

            eval_samples.append(entry)

            count += 1
            if count == 4:
                break

        except Exception as e:
            print("Skipping due to error:", e)
            continue


    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"eval_output_{timestamp}.json"

    with open(output_filename, "w") as f:
        json.dump(eval_samples, f, indent = 2)


    # generate help

    with open(output_filename, "r") as f:
        data = json.load(f)

    for entry in data:
        problem = entry["problem"]
        solution = entry["solution"]
        tags = entry["tags"]
        skill_types = entry["skill_types"]
        confusions = entry["confusions"]

        help_generation_prompt = f"""
            {help_prompt}

            Plese generate leading questions as a list of strings.

            Coding question:
            {problem}

            Coding question solution:
            {solution}

            Student questions:
            {confusions}

            Question tags:
            {tags}

            Skill types:
            {skill_types}
            """

        raw_help = model.generate_content(help_generation_prompt)

        match = re.search(r'(\[\s*".*?"\s*(?:,\s*".*?"\s*)*\])', raw_help.text, re.DOTALL)

        if match:
            list_str = match.group(1)
            try:
                help_list = ast.literal_eval(list_str)
                print(help_list)
            except Exception as e:
                print("Error parsing list:", e)
        else:
            print("List not found in model output for help.")
        
        entry["helps"] = help_list

    with open(output_filename, "w") as f:
        json.dump(data, f, indent=2)
