# TODO: include TACO. iterate through the first 10 questions of TACO. 

import google.generativeai as genai
import json
import time

if __name__ == "__main__":
    api_key = "AIzaSyBgsBHalFqsEIedpRUBS1S__3zd2W0FrOs"
    genai.configure(api_key=api_key)

    with open('confusion_generation_prompt.txt', 'r') as file:
        confusion_prompt = file.read()

    with open('coding_question.txt', 'r') as file:
        coding_question = file.read()

    # only gives student QUESTION, no ground truth solution
    confusion_generation_prompt = f"{confusion_prompt}. Please generate a list of questions. \n\n 'Coding Problem: Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. The order of the elements may be changed. Then return the number of elements in nums which are not equal to val.'"

    model = genai.GenerativeModel(model_name="gemini-2.0-flash")

    confusions = model.generate_content(confusion_generation_prompt)

    print(confusions.text)
    print("\n\n")

    with open('help_generation_prompt.txt', 'r') as file:
        help_prompt = file.read()

    help_generation_prompt = f"{help_prompt} \n\n {coding_question} \n\n {confusions}"

    help = model.generate_content(help_generation_prompt)

    print(help.text)

    output_data = {
    'access_to_solution': "false",
    'coding_question': coding_question,
    'confusions': confusions.text,
    'help': help.text
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"result_{timestamp}.json"

    with open(output_filename, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

    print(f"Response saved to {output_filename}")


    output_filename = f"result_{timestamp}.txt"
    with open(output_filename, 'w') as txt_file:
        txt_file.write("Restricted student's access to ground truth solution.\n")
        txt_file.write(coding_question + "\n\n")

        txt_file.write("Confusions:\n")
        txt_file.write(confusions.text + "\n\n")

        txt_file.write("Help:\n")
        txt_file.write(help.text + "\n")

    print(f"Response saved to {output_filename}")
