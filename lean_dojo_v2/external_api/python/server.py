import gc
from typing import Optional

import torch
from external_models import *
from fastapi import FastAPI, HTTPException
from loguru import logger
from models import *
from pydantic import BaseModel

app = FastAPI()

models = {
    "kimina": VLLMTacticGenerator(
        model="AI-MO/Kimina-Prover-Preview-Distill-7B",
        tensor_parallel_size=1,
        temperature=0.6,
        max_tokens=1024,
        top_p=0.9,
        length_penalty=0,
        n=4,
        do_sample=True,
        output_scores=True,
        output_logits=False,
        return_dict_in_generate=True,
        device="auto",
    ),
    "deepseek": HFTacticGenerator(
        model="deepseek-ai/DeepSeek-Prover-V2-7B",
        temperature=0.6,
        max_new_tokens=256,
        top_p=0.9,
        length_penalty=0,
        num_return_sequences=4,
        do_sample=True,
        output_scores=True,
        output_logits=False,
        return_dict_in_generate=True,
        device="auto",
    ),
}


class GeneratorRequest(BaseModel):
    name: str
    input: str
    prefix: Optional[str]


class Generation(BaseModel):
    output: str
    score: float


class GeneratorResponse(BaseModel):
    outputs: List[Generation]


class EncoderRequest(BaseModel):
    name: str
    input: str


class EncoderResponse(BaseModel):
    outputs: List[float]


@app.post("/generate")
async def generate(req: GeneratorRequest) -> GeneratorResponse:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    try:
        model = models[req.name]
        target_prefix = req.prefix if req.prefix is not None else ""
        outputs = model.generate(req.input, target_prefix)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        return GeneratorResponse(
            outputs=[Generation(output=out[0], score=out[1]) for out in outputs]
        )
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        raise HTTPException(status_code=500, detail="GPU out of memory")
    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/encode")
async def encode(req: EncoderRequest) -> EncoderResponse:
    model = models[req.name]
    feature = model.encode(req.input)
    return EncoderResponse(outputs=feature.tolist())
