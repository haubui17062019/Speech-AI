from transformers import pipeline
from groq import Groq

transcriber = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-large", device='cuda')
client = Groq(
    api_key="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",
)

path_audio = "bctv.wav"

output = transcriber(path_audio)['text']
print(output)

print("=" * 20)

system_prompt = """You are a helpful assistant when adding punctuation to text.
Keep the original words intact and only insert capital letters based on the context provided.
If context is not provided, say, 'No context provided'.
Just give the final result and no further explanation!.\n"""

chat_completion = client.chat.completions.create(
    messages=[
        {
          "role": "system",
          "content": system_prompt
        },
        {
            "role": "user",
            "content": output,
        }
    ],
    model="llama-3.1-70b-versatile",
)

print(chat_completion.choices[0].message.content)




