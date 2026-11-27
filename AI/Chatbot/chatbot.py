from model_client import generate_response

def chat(history, user_input):

    messages = history + [
        {"role": "user", "content": user_input}
    ]

    reply = generate_response(messages)

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})

    return reply