import time
import re


def get_match_items(items, str):
    match_time = 0
    str = str.lower()
    for i in items:
        i = i.strip().lower()
        if i in str:
            match_time += 1
    return match_time


def locate_ans(query, output):
    input_index = query.rfind('Input')
    input_line = query[input_index:]
    index = input_line.find('\n')
    input_line = input_line[:index]
    input_line = input_line.replace('Sentence 1:', ' ')
    input_line = input_line.replace('Sentence 2:', ' ')
    input_line = input_line.strip()
    inputs = input_line.split()
    output_lines = output.split('\n')
    ans_line = ''
    max_match_time = 0
    for i in range(len(output_lines)):
        line = output_lines[i]
        cur_match_time = get_match_items(inputs, line)
        if cur_match_time > max_match_time:
            max_match_time = cur_match_time
            ans_line = line
            if i < len(output_lines) - 1:
                ans_line += output_lines[i+1]
            if i < len(output_lines) - 2:
                ans_line += output_lines[i+2]
    return ans_line


api_num = 5


def get_response_from_llm(llm_model, queries, task, few_shot, api_num=4):
    model_outputs = []

    if llm_model.lower() == 't5':
        import torch
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(device)

        for q in queries:
            input_ids = tokenizer(q, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(input_ids)
            out_text = tokenizer.decode(outputs[0])
            out_text = out_text.replace('<pad>', '')
            out_text = out_text.replace('</s>', '')
            out_text = out_text.replace('<s>', '')
            out_text = out_text.strip()
            print('Model Output: ', out_text)
            model_outputs.append(out_text)

    elif llm_model.lower() == 'vicuna':
        import openai
        openai.api_key = "EMPTY"
        openai.api_base = "http://0.0.0.0:8000/v1"

        model = "vicuna-13b-v1.1"

        def get_completion(prompt):
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200)
            return response.choices[0].message.content

        for q in queries:
            output = None
            while output is None or output.isspace() or bool(re.search('[a-zA-Z]', output)) == False:
                try:
                    output = get_completion(q)
                    output = output.strip()
                except Exception as e:
                    print(e)
                    print('Retrying...')
                    time.sleep(5)
            index = output.find('.')
            if index > 0:
                output = output[:index]
            print('Model Input: ', q)
            print('Model Output: ', output)
            model_outputs.append(output)

    elif llm_model.lower() == 'chatgpt':
        import openai

        def get_completion(prompt):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                api_key='',  # add your api key here
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)
            return response["choices"][0]["message"]['content']

        for q in queries:
            output = None
            times = 0
            while output is None and times <= 10:
                try:
                    times += 1
                    output = get_completion(q)
                except Exception as e:
                    print(e)
                    print('Retrying...')
                    time.sleep(5)
            if times >= 10:
                print('Failed! Model Input: ', q)
                output = ''
            print('Model Output: ', output)
            model_outputs.append(output)

    elif llm_model.lower() == 'gpt4':
        import openai

        def get_completion(prompt):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                api_key="",  # add your api key here
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)
            return response["choices"][0]["message"]['content']

        for q in queries:
            output = None
            times = 0
            while output is None and times <= 10:
                try:
                    times += 1
                    output = get_completion(q)
                except Exception as e:
                    print(e)
                    print('Retrying...')
                    time.sleep(5)
            if times >= 10:
                print('Failed! Model Input: ', q)
                output = ''
            print('Model Output: ', output)
            model_outputs.append(output)

    elif llm_model.lower() == 'llama2':
        import torch
        from transformers import LlamaForCausalLM, LlamaTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        tokenizer = LlamaTokenizer.from_pretrained("../Llama-2-13b-chat-hf")
        model = LlamaForCausalLM.from_pretrained("../Llama-2-13b-chat-hf").to(device)

        for q in queries:
            input_ids = tokenizer(q, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(input_ids, max_new_tokens=50, temperature=0.7, do_sample=True)
            out_text = tokenizer.decode(outputs[0])
            out_text = out_text.replace('<pad>', '')
            out_text = out_text.replace('</s>', '')
            out_text = out_text.replace('<s>', '')
            out_text = out_text.strip()
            out_list = out_text.split('\n')
            for i in range(len(out_list)):
                line = out_list[i]
                if 'Answer:' in line:
                    cur_line = line.replace('Answer:', '').strip()
                    if cur_line != '':
                        out_text = cur_line
                        break
                if 'Output:' in line:
                    cur_line = line.replace('Output:', '').strip()
                    if cur_line != '':
                        out_text = cur_line
                        break
                if 'Answer:' in line:
                    if i < len(out_list) - 1:
                        next_line = out_list[i+1].strip()
                        if next_line != '':
                            out_text = next_line
                            break
                        elif i < len(out_list) - 2:
                            next_line = out_list[i+2].strip()
                            if next_line != '':
                                out_text = next_line
                                break
                if 'Output:' in line:
                    if i < len(out_list) - 1:
                        next_line = out_list[i+1].strip()
                        if next_line != '':
                            out_text = next_line
                            break
                        elif i < len(out_list) - 2:
                            next_line = out_list[i+2].strip()
                            if next_line != '':
                                out_text = next_line
                                break
            out_text = out_text.strip()
            if task == 'cause_and_effect':
                out_text = 'Sentence ' + out_text
            print('Model Output: ', out_text)
            model_outputs.append(out_text)

    else:
        import openai

        def get_completion(prompt):
            response = openai.Completion.create(
                model=llm_model,
                api_key=API_SET[api_num],
                prompt=prompt,
                temperature=0,
                max_tokens=1,
                stop=None,
                echo=False,
                logprobs=2)
            return response["choices"][0]["text"]

        for q in queries:
            output = None
            times = 0
            while output is None and times <= 10:
                try:
                    times += 1
                    output = get_completion(q)
                except Exception as e:
                    print(e)
                    print('Retrying...')
                    time.sleep(5)
            if times >= 10:
                print('Failed! Model Input: ', q)
                output = ''
            print('Model Output: ', output)
            model_outputs.append(output)

    return model_outputs
