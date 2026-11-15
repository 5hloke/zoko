import json, os, threading, re, time, uuid
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
# Using the Qwen 3.5 model from hugging face to change the model change this can be an input
WEIGHTS = "model/weights"
MODEL_CARD = "Qwen/Qwen3.5-9B"
MODEL_ROOT = os.environ.get("LOCAL_MODEL_ROOT", MODEL_CARD) + "/" + WEIGHTS + "/" + MODEL_CARD
# device detection support
DEVICE = "cpu"
def select_device(): 
    global DEVICE
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"

select_device()
print(f"The selected DEVICE is: {DEVICE}")

# Check if the model weights are downloaded. If not download them
def download_weights():
    global MODEL_ROOT
    LOCAL_MODEL_ROOT = Path(MODEL_ROOT)
    if not LOCAL_MODEL_ROOT.exists(): 
        from huggingface_hub import snapshot_download 
        LOCAL_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        print("Downloading weights for the model... ")
        snapshot_download(MODEL_CARD, local_dir = LOCAL_MODEL_ROOT)
    else:
        print("Weights already donloaded skipping download")

    


download_weights()
# Setting up the toknizer and the model
tokenizer = AutoTokenizer.from_pretrained(MODEL_CARD)
model = AutoModelForCausalLM.from_pretrained(MODEL_CARD, torch_dtype="auto", device_map = DEVICE)

# Setting up the threading locks and the app objects
lock = threading.Lock() # One model call at a time 
app = FastAPI()


TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

def normalize(messages):
    """Convert OpenAI-format messages into what HF chat templates expect."""
    out = []
    for m in messages:
        m = dict(m)
        if isinstance(m.get("content"), list):            # content parts -> string
            m["content"] = "".join(p.get("text", "") for p in m["content"])
        if m.get("content") is None:
            m["content"] = ""
        for tc in m.get("tool_calls") or []:              # arguments: JSON string -> dict
            args = tc["function"].get("arguments")
            if isinstance(args, str):
                tc["function"]["arguments"] = json.loads(args or "{}")
        out.append(m)
    return out

def generate(body):
    messages = normalize(body["messages"])
    tools = body.get("tools") or None
    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    temp = body.get("temperature")
    kwargs = dict(max_new_tokens=body.get("max_tokens") or 2048)
    if temp:
        kwargs.update(do_sample=True, temperature=temp, top_p=body.get("top_p") or 0.95)
    else:
        kwargs.update(do_sample=False)
    with lock:
        out = model.generate(**inputs, **kwargs)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text, inputs["input_ids"].shape[1], len(new_tokens)

def parse_tool_calls(text):
    calls = []
    for m in TOOL_RE.finditer(text):
        try:
            d = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:8],
            "type": "function",
            "function": {"name": d["name"], "arguments": json.dumps(d.get("arguments", {}))},
        })
    return TOOL_RE.sub("", text).strip(), calls

def sse(obj):
    return f"data: {json.dumps(obj)}\n\n"

def stream_response(rid, name, content, calls):
    base = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": name}
    def chunk(delta, finish=None):
        return sse({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
    yield chunk({"role": "assistant", "content": ""})
    if content:
        yield chunk({"content": content})
    for i, c in enumerate(calls):
        yield chunk({"tool_calls": [{"index": i, **c}]})
    yield chunk({}, "tool_calls" if calls else "stop")
    yield "data: [DONE]\n\n"

@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "local-model", "object": "model", "owned_by": "me"}]}

@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    text, n_in, n_out = generate(body)
    content, calls = parse_tool_calls(text)
    rid, name = "chatcmpl-" + uuid.uuid4().hex[:12], body.get("model", "local-model")

    if body.get("stream"):
        return StreamingResponse(stream_response(rid, name, content, calls),
                                 media_type="text/event-stream")
    message = {"role": "assistant", "content": content or None}
    if calls:
        message["tool_calls"] = calls
    return JSONResponse({
        "id": rid, "object": "chat.completion", "created": int(time.time()), "model": name,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "tool_calls" if calls else "stop"}],
        "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out},
    })



