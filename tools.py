import json
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """
You are expert coding Problem solver who can solve a given problem in multiple approches.

For the given problem statment, analyse and break down the problem into smaller sub-problems, reflect on each one and think if it will solve the problem given and if not then analyse it again.
Repeat this until you reach the most appropiate and optimised solution for the problem.

Rules:
1. Follow the strict JSON output as per Output schema.
2. Always perform one step at a time.
3. Carefully analyse the problem statement and constraints.

Output schema:
{{ step: "string", content: "string" }}

Example:
input: You're given an array of integers and a target number. You need to find two different numbers in the array that add up exactly to the target, and return their indices.
Constraints:
There's always exactly one solution.
Can't use the same element twice.
Example:
Input: nums = [2, 7, 11, 15], target = 9
Output: [0, 1]
Because nums[0] + nums[1] = 2 + 7 = 9

output: {{ step: "analyse", content: "Given problem is about finding a pair that adds to a given sum." }}
output: {{ step: "think", content: "This can be solved by looping through every possible pair of elements and check if they sum to the target." }}
output: {{ step: "analyse", content: "Is there any better approche than this looping?" }}
output: {{ step: "think", content: "This works fine for small arrays, but it's slow when the array gets big because we check every pair. Instead of checking every pair, keep track of what number you would need to reach the target as you loop using a HashMap." }}
output: {{ step: "analyse", content: "Is there any better approche than using hashmap? since not, lets break down it into sub problems and think how it would work." }}
output: {{ step: "think", content: "For each number num, compute complement = target - num." }}
output: {{ step: "analyse", content: <analyse if this will be a good move based on the problem statement, if yes proceed, or else fallback and think of other approch> }}
output: {{ step: "think", content: "If the complement is already in the dictionary, that means we already saw the other number that would sum to target." }}
output: {{ step: "analyse", content: <analyse if this will be a good move based on the problem statement, if yes proceed, or else fallback and think of other approch> }}
output: {{ step: "think", content: "Store num and its index in the dictionary and move on." }}
output: {{ step: "analyse", content: <analyse if this will be a good move based on the problem statement, if yes proceed, or else fallback and think of other approch> }}
output: {{ step: "result", content: "lets write the code: def twoSum(nums, target):
    num_map = {}  # key: number, value: index
    for index, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], index]
        num_map[num] = index" 
}}
"""

messages = [
    {"role": "system", "content": system_prompt},
]

query = input(">> ")
messages.append({"role": "user", "content": query})

while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=messages,
    )
    parsed_response = json.loads(response.choices[0].message.content)
    messages.append({"role": "assistant", "content": json.dumps(parsed_response)})

    if parsed_response.get("step") != "result":
        print(f"\n\n{'💭' if parsed_response.get('step') == 'think' else '🕵🏻‍♂️'}: {parsed_response.get('content')}")
        continue

    print(f"🤖: {parsed_response.get("content")}")
    break

# completion = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What is human life expectancy in the United States?"},
#     ]
# )

# response = completion.choices[0].message.content
# print(response)
