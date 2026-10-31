#!/usr/bin/env python3
"""Does vLLM reuse the prefix cache? Stdlib only.

  repeat mode:        python3 check_cache.py repeat fixtures/prompt64k.json --times 6
  conversation mode:  python3 check_cache.py conversation fixtures/conversation10.json

For every request it prints prompt tokens, prefix-cache hits (delta of
vllm:prefix_cache_hits_total on /metrics), hit ratio and time to first token.
Run one request at a time on an otherwise idle engine, or the deltas mix.
Set VLLM_API_KEY if your server uses --api-key.
"""
import argparse, json, os, re, sys, time, urllib.request

def req(url, body=None, stream=False):
    h = {"Content-Type": "application/json"}
    if os.environ.get("VLLM_API_KEY"):
        h["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]
    data = json.dumps(body).encode() if body is not None else None
    return urllib.request.urlopen(urllib.request.Request(url, data=data, headers=h), timeout=900)

def counters(base):
    text = req(base + "/metrics").read().decode()
    out = {}
    for name in ("prefix_cache_hits_total", "prefix_cache_queries_total"):
        vals = re.findall(r"^vllm:%s(?:\{[^}]*\})? ([0-9.eE+]+)$" % name, text, re.M)
        out[name] = sum(float(v) for v in vals)
    return out

def chat(base, model, messages, max_tokens):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}}
    t0, ttft, content, usage = time.monotonic(), None, [], {}
    with req(base + "/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if ttft is None and (d.get("content") or d.get("reasoning_content") or d.get("reasoning")):
                    ttft = time.monotonic() - t0
                if d.get("content"):
                    content.append(d["content"])
    return "".join(content), usage, ttft, time.monotonic() - t0

def measure(base, model, messages, max_tokens, label):
    before = counters(base)
    reply, usage, ttft, total = chat(base, model, messages, max_tokens)
    time.sleep(1)
    after = counters(base)
    hits = after["prefix_cache_hits_total"] - before["prefix_cache_hits_total"]
    prompt = usage.get("prompt_tokens") or 0
    ratio = hits / prompt if prompt else 0
    print(f"{label:>8}  prompt {prompt:>7}  hits {int(hits):>7}  ({ratio:5.1%})  "
          f"first token {ttft or 0:6.2f} s  total {total:6.1f} s", flush=True)
    return reply

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["repeat", "conversation"])
    ap.add_argument("fixture")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--times", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=4096)
    a = ap.parse_args()
    fx = json.load(open(a.fixture))
    if a.mode == "repeat":
        for i in range(a.times):
            measure(a.url, a.model, fx["messages"], a.max_tokens, f"rep {i + 1}")
    else:
        messages = []
        for i, turn in enumerate(fx["turns"], 1):
            messages.append({"role": "user", "content": turn["user"]})
            reply = measure(a.url, a.model, messages, a.max_tokens, f"turn {i}")
            messages.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    sys.exit(main())
