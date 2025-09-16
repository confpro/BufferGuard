import pandas as pd
from openai import OpenAI
df = pd.read_csv('***.csv')

client = OpenAI(
    base_url='https://api.deepseek.com',
    api_key='sk-***',
)

def analyze_code(code):
    messages = [
        {
            "role": "user",
            "content": "Task Desctiption & Instructions "
                       "Analyze the information about the buffer overflow in the code snippet provided below. "
                       "First, generate a summary description of this code. "
                       "Then, determine if there are any allocation/variable passing function safety issues. "
                       "After that, determine if these functions have boundary checks. "
                       "Finally, determine if there is a buffer size calculation error. Output Format Specification "
                       "Desired format: Description: ' <description> ' "
                       "Assign/Variable Pass API: line <line number>:' <API call> ' assign/pass ' <variable> ' variable "
                       "Whether the data boundaries of the API are checked: line <line number>:' <variable> ' check ' <boundary> ' specification "
                       "Whether data size or type passing matches: line <line number>:' <variable> ' pass ' <size or type> ' match "
                       "Code Placeholder Code Snippet: ```c {} ```".format(code),
        }
    ]
    response = client.chat.completions.create(
        messages=messages,
        model="deepseek-chat",
    )
    for message in response.choices:
        print(message.message.content)
    result = {}
    for message in response.choices:
        for line in message.message.content.split('\n'):
            if line.startswith('Description:'):
                result['description'] = line[len('Description: '):].strip()
            elif line.startswith('Assign/Variable Pass API:'):
                result['API'] = line[len('Assign/Variable Pass API: '):].strip()
            elif line.startswith('Whether the data boundaries of the API are checked:'):
                result['boundary'] = line[len('Whether the data boundaries of the API are checked: '):].strip()
            elif line.startswith('Whether data size or type passing matches:'):
                result['match'] = line[len('Whether data size or type passing matches: '):].strip()
    return result

results_df = pd.DataFrame(columns=['func_before', 'description', 'API', 'boundary', 'match'])
count = 0
start_index = 0
for index, row in df.iloc[start_index:].iterrows():
    result = analyze_code(row['func_before'])
    new_row = {
        'func_before': row['func_before'],
        'description': result.get('description', ''),
        'API': result.get('API', ''),
        'boundary': result.get('boundary', ''),
        'match': result.get('match', '')
    }
    new_df = pd.DataFrame([new_row])
    results_df = pd.concat([results_df, new_df], ignore_index=True)
    results_df = results_df.iloc[[-1]]
    results_df.to_csv('***.csv', index=False, mode='a', header=not index)
    count += 1
    print("num:{}".format(count))








