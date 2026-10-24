# http_worker (optional, future use)

A warm HTTP image server for a GPU box you start yourself (for example a Kaggle notebook).
Models load once; after that each request takes seconds instead of minutes.

**It is separate from the Kaggle batch flow.** Nothing in `worker/`, `notebooks/kaggle_runner.py` or Studio's
batch transport imports this package, and it does nothing until you start it.

## Safe defaults
- Binds to `127.0.0.1`, no tunnel (`tunnel: none`). Nothing is reachable from outside until you opt in.
- Refuses to start without `WORKER_TOKEN` (set it as a Kaggle Secret). `ALLOW_NO_AUTH=1` is for local tests only.
- Writes images to `/tmp/http_worker_out`, never `/kaggle/working`, so they are not bundled as kernel output.
- Hard session cap (`max_session_minutes: 360`) so a forgotten session cannot drain your weekly GPU quota.
  Optional dead-man switch: `idle_timeout_minutes: 20` stops the server when Studio stops sending requests.
- Honors `worker/models.yaml`: disabled, test-only and non-commercial engines are refused.

## Endpoints (all need `Authorization: Bearer <WORKER_TOKEN>`)
| Method and path | Purpose |
|---|---|
| `GET /health` | `ready`, loaded engines, per-GPU state, queue depth, session time left |
| `GET /v1/models` | OpenAI-style model list (the engines currently loaded) |
| `POST /v1/images/generations` | OpenAI-compatible: `model`, `prompt`, `size` (`1344x768`), `n`, `response_format` (`b64_json` or `url`), plus optional `seed`, `steps`, `output_format` |
| `POST /v1/jobs`, `GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/file` | Async job: submit, poll, download |
| `POST /v1/packs`, `GET /v1/packs/{id}`, `GET /v1/packs/{id}/download` | Many items in one request; download is a zip with `results.json` |
| `POST /v1/shutdown` | Stops the server (and the notebook cell) cleanly |

Example:
```
curl -s $URL/v1/images/generations -H "Authorization: Bearer $WORKER_TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"flux2_klein","prompt":"flat vector ship at sea","size":"1344x768"}'
```

## Settings
Edit `http_worker/config.yaml` (or the settings cell in the notebook). Key choices:
- `roles`: which engine runs on which GPU. FLUX (6.7 GB) plus SDXL (7.2 GB) do not fit together on one 15 GB T4,
  so the default is GPU 0 = `flux2_klein`, GPU 1 = `sdxl_lightning`. Use two FLUX roles for twice the FLUX speed.
- `output_format`: `webp` is about 5 to 10 times smaller than PNG, which matters for tunnel bandwidth.
- `tunnel`: `none`, `ngrok` or `cloudflared`. ngrok free plan limits: 1 GB/month out, 20k requests/month.
  Send whole packs and poll sparingly. The tunnel code is untested here.

## Connecting Studio later
Use Studio's custom source type `openai_images` (for `/v1/images/generations`) or `studio_http_worker`
(jobs and packs). Base URL is your tunnel URL plus `/v1`, the key is the same `WORKER_TOKEN`.

## Tested
`tests/test_http_worker.py` (14 tests, CPU, fake engine): auth, OpenAI-style generation (base64 and URL),
async jobs, packs and zip, size validation, request logging, shutdown, idle timeout, session cap, engine policy,
refusal to start without a token, and failed engine loads. **Not tested on a GPU or through a tunnel.**
