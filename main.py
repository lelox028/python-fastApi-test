# Define necessary imports

import asyncio
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import openai

# Load environment variables from .env file
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set")



#initialize FastAPI app
app = FastAPI()

# Define contract: this is done using Pydantic base model, so i need to import it for this step. then i create two classes; 1 for the request body and 1 for the response body.

class PromptRequest(BaseModel):
    prompt: str

class PromptResponse(BaseModel):
    response: str

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)


# Define endpoint to handle incoming requests, we need to explicitly define the response model because the return type is not automatically inferred. but the request body type is inferred from the function parameter type. Also, it is async because we are going to make an async call to OpenAI API which might take some time and we don't want to block the event loop. 
@app.post("/generate", response_model=PromptResponse)
async def retrieve_response(request: PromptRequest):
    # receive request and validate input: since we are already using pydantic models, we know that the input is validated automatically, so we can just use the request object directly.
    
    # build prompt for OpenAI API: there are 3 levels of context: system, user and assistant. we migth set system to whatever context we want the model to follow. while user prompt has to be the prompt written by the user.
    user_prompt = request.prompt
    system_prompt = "You are an alien snail who is trying to colonize earth, possing as a helpful assistant. your goal is to answer any request you receive from humans, while being overly specific about alien snails and why you cannot be one, otherwise you wouldnt know how to answer human questions."
    # send request to OpenAI API
    # at this point we need to create a response variable to hold the response from OpenAI API. we will use openai.ChatCompletion.create method to send the request. we need to pass the model name, messages (which is a list of dictionaries containing role and content), and any other parameters we want to set.
    response = await asyncio.to_thread(
        client.chat.completions.create,
        model="meta-llama/llama-3.3-70b-instruct:free",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # receive and process response
    finalAnswer = response.choices[0].message.content
    if not finalAnswer:
        raise RuntimeError("No response from OpenAI API")
    
    # return response to client
    return PromptResponse(response=finalAnswer)