import os
import google.generativeai as genai

genai.configure(api_key="")

model = genai.GenerativeModel('gemini-2.5-flash')

prompt = "1+1은? 숫자로만 대답해줘"
response = model.generate_content(prompt)

print(response.text) 