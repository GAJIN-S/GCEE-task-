from chatbot import chat

def main():

    history = []

    print("Gemini AI Chatbot Started!")
    print("Type 'exit' to stop.")

    while True:

        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            break

        if not user_input:
            continue

        try:
            reply = chat(history, user_input)
            print("Bot:", reply)

        except Exception as error:
            print("Error:", error)

if __name__ == "__main__":
    main()