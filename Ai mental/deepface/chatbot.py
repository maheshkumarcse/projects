# Step 2: Import libraries
from transformers import pipeline
import gradio as gr

# Step 3: Load the conversational model (DialoGPT-medium)
chatbot = pipeline('conversational', model='microsoft/DialoGPT-medium')

# Step 4: Function for text-based chat (console interaction)
def chat():
    print("Chat with the bot! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':

            print("Goodbye!")
            break
        # Get the chatbot's response
        response = chatbot(user_input)
        print(f"Bot: {response[0]['generated_text']}")

# Uncomment the following line to start the console chat loop
# chat()

# Step 5: Function for Gradio interface (web-based chat)
def respond_to_input(input_text):
    response = chatbot(input_text)
    return response[0]['generated_text']

# Step 6: Create and launch the Gradio interface for web-based interaction
gr.Interface(fn=respond_to_input, inputs="text", outputs="text").launch()
