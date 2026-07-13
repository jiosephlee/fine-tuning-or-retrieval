"""Measure vLLM request throughput across powers-of-two concurrency levels."""
import argparse, concurrent.futures, time
from openai import OpenAI

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-url', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--requests', type=int, default=256)
    p.add_argument('--max-workers', type=int, nargs='+', default=[1,2,4,8,16,32,64,128,256])
    a = p.parse_args()
    for workers in a.max_workers:
        client = OpenAI(base_url=a.base_url, api_key='EMPTY', timeout=600, max_retries=0)
        def one(i):
            t = time.perf_counter()
            r = client.chat.completions.create(model=a.model, messages=[{'role':'user','content':f'Reply with exactly: {i}'}], max_tokens=16, reasoning_effort='low')
            return time.perf_counter()-t, bool((r.choices[0].message.content or '').strip())
        t0=time.perf_counter(); ok=0; lat=[]; errors=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in concurrent.futures.as_completed([ex.submit(one,i) for i in range(a.requests)]):
                try: l, valid=fut.result(); ok += int(valid); lat.append(l)
                except Exception as e: errors.append(type(e).__name__)
        elapsed=time.perf_counter()-t0
        print(f'workers={workers} requests={a.requests} ok={ok} errors={len(errors)} elapsed_s={elapsed:.2f} req_s={ok/elapsed:.2f} p50_s={sorted(lat)[len(lat)//2] if lat else 0:.2f}', flush=True)

if __name__ == '__main__': main()
