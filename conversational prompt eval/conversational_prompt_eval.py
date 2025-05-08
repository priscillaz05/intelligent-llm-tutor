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

    count = 0

    for sample in ds:
        try:
            solutions = json.loads(sample["solutions"])

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
                print("List not found in model output.")

            entry = {
            "problem": problem,
            "solution": solution,
            "tags": tags,
            "skill_type": skill_types,
            }

            entry["confusion"] = confusions_list

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



    # with open('help_generation_prompt.txt', 'r') as file:
    #     help_prompt = file.read()

    # help_generation_prompt = f"{help_prompt} \n\n {coding_question} \n\n {confusions}"

    # help = model.generate_content(help_generation_prompt)

    # print(help.text)

    # output_data = {
    # 'access_to_solution': "false",
    # 'coding_question': coding_question,
    # 'confusions': confusions.text,
    # 'help': help.text
    # }

    # timestamp = time.strftime("%Y%m%d_%H%M%S")
    # output_filename = f"result_{timestamp}.json"

    # with open(output_filename, 'w') as json_file:
    #     json.dump(output_data, json_file, indent=4)

    # print(f"Response saved to {output_filename}")


    # output_filename = f"result_{timestamp}.txt"
    # with open(output_filename, 'w') as txt_file:
    #     txt_file.write("Restricted student's access to ground truth solution.\n")
    #     txt_file.write(coding_question + "\n\n")

    #     txt_file.write("Confusions:\n")
    #     txt_file.write(confusions.text + "\n\n")

    #     txt_file.write("Help:\n")
    #     txt_file.write(help.text + "\n")

    # print(f"Response saved to {output_filename}")
