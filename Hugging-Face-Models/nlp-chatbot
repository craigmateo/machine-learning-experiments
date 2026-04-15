"""
This script extends the basic BlenderBot example into a simple
multi-turn conversational chatbot that maintains context across turns.

Step-by-step:

1. Suppress Logging Noise
   - Reduces output clutter from the transformers library.

2. Load Model and Tokenizer
   - AutoTokenizer: Converts text to tokens and back.
   - BlenderbotForConditionalGeneration: A pretrained conversational model.

3. Initialize Conversation History
   - A Python list called `history` stores all previous messages.
   - This acts as the chatbot’s memory.

4. Start Interactive Loop
   - The program continuously waits for user input using input().
   - Typing "quit" or "exit" ends the conversation.

5. Store User Message
   - Each user message is added to the history list.
   - Messages are prefixed (e.g., "You: ...") for clarity.

6. Build Full Input Context
   - All past messages are joined into a single string.
   - This full conversation is sent to the model each turn.

7. Tokenize Input
   - The combined conversation text is converted into tensors.

8. Generate Response
   - model.generate(...) produces the chatbot’s reply based on the entire history.
   - max_length controls the maximum response size.

9. Decode Output
   - Converts generated tokens back into readable text.

10. Print and Store Bot Reply
   - The response is printed to the console.
   - It is also added to the history so future responses include it.

Key Concepts:

- Conversation Memory:
  The model itself does NOT remember past messages.
  Memory is simulated by resending the entire conversation each turn.

- Context Window:
  As history grows, input gets longer.
  The tokenizer may truncate older messages if too long.

- Stateless Model:
  Each call to model.generate() is independent.
  The illusion of memory comes from manually managing history.

Limitations:

- Responses may degrade as history grows too long.
- No speaker-awareness beyond simple text prefixes.
- No fine control over tone or personality.
- Not optimized for long or complex dialogues.

This is a simple but important pattern:
→ “Conversation” = repeatedly feeding accumulated context into a text generation model.
"""

from transformers.utils import logging
logging.set_verbosity_error()

from transformers import AutoTokenizer, BlenderbotForConditionalGeneration

# Load model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Store conversation history
history = []

print("Chatbot ready! Type 'quit' or 'exit' to stop.\n")

while True:
    # Get user input
    user_message = input("You: ")
    if user_message.lower() in ["quit", "exit"]:
        print("Ending conversation.")
        break

    # Add user message to history
    history.append(user_message)

    # Combine history into a single string (context)
    full_input = " ".join(history)

    # Tokenize input
    inputs = tokenizer(
        [full_input],
        return_tensors="pt",
        truncation=True  # prevents overly long inputs
    )

    # Generate response
    reply_ids = model.generate(
        **inputs,
        max_length=100
    )

    # Decode response
    bot_reply = tokenizer.batch_decode(
        reply_ids,
        skip_special_tokens=True
    )[0]

    # Print and store response
    print("Bot:", bot_reply)
    history.append("Bot: " + bot_reply)