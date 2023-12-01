from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import zhipuai
import logging
import json
import asyncio
from fastapi.responses import StreamingResponse
import time
import os
import random


app = FastAPI()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DialogueRequest(BaseModel):
    message: str

@app.post("/dialogue/")
async def dialogue(dialogue_request: DialogueRequest):
    try:
        zhipuai.api_key = "1a21c86a3aa8f435250194b3dc9dc6b8.2Aov2pnPfNB7lLPi"
        response = zhipuai.model_api.sse_invoke(
            model="chatglm_turbo",
            prompt=dialogue_request.message,
            temperature=0.2,
            top_p=0.7,
            incremental=True
        )
        logging.info(f"用户提问: {dialogue_request.message}")

        async def async_event_generator(response):
            for event in response.events():
                yield event

        async def event_generator():
            try:
                async for event in async_event_generator(response):
                    if event.event == "add":
                        formatted_data = f"data: {json.dumps(event.data)}\n\n"
                        yield formatted_data
                    elif event.event in ["error", "interrupted"]:
                        yield f"data: {json.dumps(event.data)}\n\n"
                    elif event.event == "finish":
                        break
            except Exception as e:
                logging.error(f"Event Generator Exception: {str(e)}")
                # 处理异常

        headers = {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }

        return StreamingResponse(event_generator(), headers=headers)

    except Exception as e:
        logging.error(f"Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
