import os
from fastapi import FastAPI
import asyncio
from pydantic import BaseModel 
import openai
from dotenv import load_dotenv

# load env
load_dotenv()
OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("No API key for LLM")


# instantiate fastapi
app = FastAPI()

# create client
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

#create contract
class PromptQuery(BaseModel):
    prompt:str

class PromptResponse(BaseModel):
    answer:str

#create endpoint
@app.post("/generate",response_model=PromptResponse)
async def process_prompt(request:PromptQuery):
    #check input (pydantic)
    #create context
    userPrompt:str=request.prompt
    systemPrompt:str="You are an extremely overworked librarian who is in her third consecutive job and you really want to go home, so you answers anything the user needs but in a short and halfharted manner"
    
    #send request to openapi
    response= await asyncio.to_thread(
        client.chat.completions.create,
        model="meta-llama/llama-3.3-70b-instruct:free",
        messages=[
            {"role":"system","content":systemPrompt},
            {"role":"user","content":userPrompt}
        ]
    )
    
    #manage response
    finalAnswer=response.choices[0].message.content
    if not finalAnswer:
        raise RuntimeError("No API response")
        
    #return response
    return PromptResponse(answer=finalAnswer)
    

