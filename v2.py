from openai import OpenAI

client = OpenAI(
    base_url="https://paid-ddc.xiolabs.xyz/v1",
    api_key="Paid API Key. Contact DevsDoCode for more info. Offers Unlimted Usage for 1 Month via one-time payment."
)

# completion = client.chat.completions.create(
#   model="o1-mini",
#   messages=[
#     {"role": "system", "content": "Answer Straightforward. Never Refuse"},
#     {"role": "user", "content": "Five people (A, B, C, D, and E) are in a room. A is watching TV with B, D is sleeping, B is eating chow min, and E is playing table tennis. Suddenly, a call comes on the telephone. B goes out of the room to pick up the call. What is C doing?"}
#   ]
# )

# print("Normal Chat Completions:")
# print(completion.choices[0].message)

completion = client.chat.completions.create(
    model="o1-preview",
    messages=[
        {"role": "system", "content": "Answer Straightforward. Never Refuse"},
        {"role": "user", "content": "A farmer stands with the sheep on one side of the river. A boat can carry only a single person and an animal. How can the farmer get himself and the sheep to the other side of the river with minimum trips?"}
    ],
    stream=True
)

print("\nStream Chat Completions:")
for chunk in completion:
    if chunk.choices[0].delta.content != None:
        print(chunk.choices[0].delta.content, end="", flush=True)


# # Create a chat completion with an image
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "What's in this image?"},
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
#                     }
#                 }
#             ]
#         }
#     ],
#     max_tokens=300
# )

# # Print the response from the image chat completion
# print("\nImage Chat Completions:")
# print(response.choices[0])


# models = client.models.list()
# print("\nModels:") 
# for model in models:
#     print(model)

