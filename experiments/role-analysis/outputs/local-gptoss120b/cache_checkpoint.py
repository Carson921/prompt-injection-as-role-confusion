from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib,json,requests,time
REV="b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
OUT=Path("/dev/shm/role-analysis-gptoss120b")
OUT.mkdir(exist_ok=True)
BASE=f"https://huggingface.co/openai/gpt-oss-120b/resolve/{REV}/"
started=time.time()
index=requests.get(BASE+"model.safetensors.index.json",timeout=30);index.raise_for_status()
(OUT/"model.safetensors.index.json").write_bytes(index.content)
files=sorted(set(index.json()["weight_map"].values()))
def download(name):
    target=OUT/name
    response=requests.head(BASE+name,allow_redirects=False,timeout=30)
    response.raise_for_status()
    expected=response.headers["x-linked-etag"].strip('"')
    size=int(response.headers["x-linked-size"])
    if target.exists() and target.stat().st_size==size:
        return {"name":name,"sha256":expected,"size":size,"cached":True}
    r=requests.get(BASE+name+"?download=true",stream=True,timeout=60);r.raise_for_status()
    digest=hashlib.sha256();n=0
    with target.with_suffix(".partial").open("wb") as f:
        for block in r.iter_content(8*1024*1024):
            f.write(block);digest.update(block);n+=len(block)
    assert n==size,(name,n,size)
    assert digest.hexdigest()==expected,(name,digest.hexdigest(),expected)
    target.with_suffix(".partial").replace(target)
    print(name,n,round(time.time()-started,1),flush=True)
    return {"name":name,"sha256":expected,"size":size}
results=[]
with ThreadPoolExecutor(max_workers=8) as pool:
    for f in as_completed([pool.submit(download,name) for name in files]):
        results.append(f.result())
for name in ["config.json","generation_config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","chat_template.jinja"]:
    r=requests.get(BASE+name,timeout=30);r.raise_for_status();(OUT/name).write_bytes(r.content)
(OUT/"verified-weights.json").write_text(json.dumps({"revision":REV,"files":results,"seconds":time.time()-started},indent=2))
print("COMPLETE",time.time()-started,flush=True)
