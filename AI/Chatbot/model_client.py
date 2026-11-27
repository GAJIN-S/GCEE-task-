from google import genai
from google.genai import types
from config import API_KEY, MODEL_NAME, SYSTEM_PROMPT

client = genai.Client(api_key=API_KEY)

def generate_response(messages):

    contents = []

    for message in messages:
        role = "model" if message["role"] == "assistant" else "user"

        contents.append(
            types.Content(
                role=role,
                parts=[types.Part(text=message["content"])]
            )
        )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=2048,
            temperature=0.7
        )
    )

    # Debug information
    print("Finish reason:", response.candidates[0].finish_reason)

    return response.text