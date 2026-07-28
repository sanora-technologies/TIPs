# TIPs — Virtual Environments & CUDA Stack

Audited **2026-07-13** on the live box (RTX 5060 Ti, driver 595.71.05).

This is the reference for *which venv serves what*, *exactly which CUDA wheels are
pinned*, and *why the copy to the external SSD lost files*. The CUDA/wheel alignment
below was painful to reach — **do not "clean up" the duplicate `nvidia-*` wheels.**
Read [Why the venv looks like it has duplicate CUDA wheels](#why-the-venv-looks-like-it-has-duplicate-cuda-wheels) first.

---

## TL;DR

| Directory | Size | Status | Serves |
|---|---|---|---|
| `TIPs/venv` | 13 G | ✅ **LIVE** | `api_server_linux.py` — the whole API (FastAPI/uvicorn, nnU-Net, YOLO/OPG, CBCT) |
| `TIPs/ToothGroupNetwork/venv_tgn` | 8.7 G | ✅ **LIVE** | ToothGroupNetwork IOS_SCAN inference, spawned as a **subprocess** |
| `TIPs/venv_tgn` | 8.7 G | ❌ **DEAD** — broken copy | nothing. Safe to delete. |
| `TIPs/server_env_3.11` | 11 G | ❌ **DEAD** — broken copy, no `bin/` at all | nothing. Safe to delete. |

Two venvs are live. The other two are **wreckage from a round-trip through the exFAT SSD**
that lost their symlinks. Deleting both reclaims **~20 GB**.

All four are Python **3.11.15** from `/usr/bin/python3.11`.

---

## 1. Who serves what

### `TIPs/venv` — the main server

Everything in `api_server_linux.py` runs here. It is selected in three places, all
of which must agree:

- `start_server.sh` → `source .../TIPs/venv/bin/activate`
- `api_server_linux.py:214` → `VENV_BIN = $TIPS_VENV_BIN or REPO_ROOT/venv/bin`
- `tips-server.service` → `ExecStart=.../start_server.sh`

Serves: FastAPI + uvicorn on **port 7863** (the `--port` default in code is 7862;
`start_server.sh` overrides it to 7863), behind ngrok.

### `ToothGroupNetwork/venv_tgn` — the TGN subprocess

`api_server_linux.py:293` resolves it:

```python
TGN_VENV_PATH = Path(os.environ.get('TGN_VENV_PATH',
                     str(REPO_ROOT / 'ToothGroupNetwork' / 'venv_tgn')))
```

It exists as a **separate venv on purpose**: TGN needs a CUDA-compiled `pointops`
extension built against its own torch, and mixing it into the main venv broke the
nnU-Net stack. The main server never imports TGN — it shells out
(`_tgn_python()`, `api_server_linux.py:2624`) and talks over files + stdout.

**The default path is correct. `TGN_VENV_PATH` is commented out in `.env`, so the
default wins — and the default points at the *ToothGroupNetwork/* copy, not the root one.**

### Verified working

```
$ ToothGroupNetwork/venv_tgn/bin/python -c "from external_libs.pointops.functions import pointops; ..."
pointops import: OK
pointops CUDA kernel ran OK, idx shape: torch.Size([256])
```

`pointops` is **not** a top-level module. It is installed as an egg
(`pointops-0.0.0-py3.11-linux-x86_64.egg`, wired in via `easy-install.pth`) but
imported as `external_libs.pointops.functions.pointops`. `pip list` will never show
it and `import pointops` will always fail — **that is normal, not a bug.**
`torch_scatter` is likewise absent and not needed.

---

## 2. Hardware / system baseline

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti, 16311 MiB |
| Compute capability | **12.0** (`sm_120`, Blackwell) |
| Driver | **595.71.05** |
| System CUDA toolkit | `/usr/local/cuda` → **cuda-12.8** (`nvcc` not on PATH) |
| Python | 3.11.15 (`/usr/bin/python3.11`) |
| jemalloc | `/usr/lib/x86_64-linux-gnu/libjemalloc.so.2` ✅ |
| ngrok | `/usr/local/bin/ngrok` ✅ |

**`sm_120` is the whole reason this stack is fussy.** Blackwell needs CUDA 13 wheels;
anything older produces `no kernel image is available for execution on the device`.
Both live venvs ship torch built for **cu130**, whose `arch_list` includes `sm_120`:

```
['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

The **system** toolkit being 12.8 does not matter — torch uses the CUDA runtime from
its own `nvidia-*` wheels, not `/usr/local/cuda`. Do not "upgrade" system CUDA to fix
a GPU error; it is not in the path.

---

## 3. Torch / CUDA versions (the part that was hard to get right)

Both live venvs report the same runtime, but were **pinned by different pip
resolutions** — the wheel sets are *not* identical. Both work.

| | `venv` (main) | `ToothGroupNetwork/venv_tgn` |
|---|---|---|
| torch | **2.11.0+cu130** | **2.11.0+cu130** |
| torchvision | 0.26.0+cu130 | 0.26.0 |
| triton | 3.6.0 | 3.6.0 |
| torch `version.cuda` | **13.0** | **13.0** |
| cuDNN (runtime) | **91900** (9.19.0) | **91900** (9.19.0) |
| `cuda.is_available()` | ✅ True | ✅ True |
| numpy | 2.4.4 | 2.4.6 |
| scipy | 1.17.1 | 1.17.1 |
| `pip check` | clean | clean |

> ⚠️ `pip list` shows torch as `2.11.0` in the main venv and `2.11.0+cu130` in the TGN
> venv. **Both are the same cu130 build** — `torch.__version__` reports `2.11.0+cu130`
> in both. The bare `2.11.0` is only a metadata artifact of how it was installed.

### Main-venv application stack

| Package | Version |
|---|---|
| fastapi | 0.135.3 |
| uvicorn | 0.43.0 |
| starlette | 1.0.0 |
| pydantic | 2.12.5 |
| ultralytics (OPG/YOLO) | 8.4.33 |
| nnunetv2 | 2.6.4 |
| monai | 1.3.0 |
| SimpleITK | 2.5.3 |
| batchgenerators | 0.25.1 |
| einops | 0.8.2 |
| trimesh | 4.12.0 |
| gradio | 6.13.0 |

`flask` and `cupy` are **not installed and not needed** — the server is FastAPI, and
GPU resampling degrades gracefully (`api_server_linux.py:209`).

### TGN-venv stack

| Package | Version |
|---|---|
| open3d | 0.19.0 |
| trimesh | 4.12.2 |
| scikit-learn | 1.8.0 |
| pointops | 0.0.0 (egg, CUDA-compiled — see above) |

---

## 4. Why the venv looks like it has duplicate CUDA wheels

`pip list` in `venv` shows what looks like **three generations** of every NVIDIA
library stacked on top of each other:

```
nvidia-cublas                 13.1.0.3      ← cu13 (unsuffixed = CUDA 13)
nvidia-cublas-cu12            12.9.1.4      ← cu12 leftovers
nvidia-cudnn-cu12             9.10.2.21
nvidia-cudnn-cu13             9.19.0.56     ← the one actually loaded
nvidia-nccl-cu12              2.27.3
nvidia-nccl-cu13              2.28.9
...
```

**This is expected and must be left alone.** In the CUDA-13 era NVIDIA renamed the
wheels: the CUDA 13 packages dropped the suffix (`nvidia-cublas`) or use `-cu13`,
while the old `-cu12` wheels remain as separate, independently-named distributions.
pip has no idea they are the same library, so upgrading torch to a cu130 build
*adds* the new wheels without removing the old ones.

Consequences:

- `pip check` is **clean** — nothing is actually broken.
- `nvidia/` on disk holds **one directory per library** plus a `cu13/` subdir. The
  `-cu12` dist-infos are largely stale metadata pointing at files the cu13 wheels
  overwrote.
- torch loads from its own `torch/lib` + the `cu13` paths. The stale cu12 metadata
  is inert.

**Do not `pip uninstall` the `-cu12` packages to "clean up".** They share file paths
with the cu13 wheels; uninstalling them deletes `.so` files the cu13 wheels still own
and leaves the venv unbootable. This is the single easiest way to destroy this venv.
If you must rebuild, rebuild from scratch (§7), never by subtraction.

### The two venvs pinned *different* cu12 leftovers

Harmless, but worth recording so nobody "fixes" the drift:

| Wheel | `venv` | `ToothGroupNetwork/venv_tgn` |
|---|---|---|
| nvidia-cublas-cu12 | 12.9.1.4 | 12.8.4.1 |
| nvidia-cudnn-cu12 | 9.10.2.21 | **9.19.0.56** |
| nvidia-nccl-cu12 | 2.27.3 | 2.28.9 |
| nvidia-cuda-cupti-cu12 | 12.9.79 | 12.8.90 |
| nvidia-nvjitlink-cu12 | 12.9.86 | 12.8.93 |

The **cu13** wheels — the ones that actually load — are **identical across both venvs**
(cublas 13.1.0.3, cudnn 9.19.0.56, nccl 2.28.9, nvjitlink 13.0.88, cuda-runtime 13.0.96).
That is why both work.

---

## 5. Known latent issues (not currently breaking)

### `/usr/local/cuda-13.0` is hardcoded but does not exist

`api_server_linux.py:2900` appends `/usr/local/cuda-13.0/lib64` to the TGN
subprocess `LD_LIBRARY_PATH`. **That directory does not exist on this box**
(system CUDA is 12.8). It is silently ignored — a non-existent path in
`LD_LIBRARY_PATH` is a no-op — and TGN works because the wheel paths *before* it
supply everything. Harmless today; would bite if someone ever relies on it.

### `LD_LIBRARY_PATH` list omits two present libs

`api_server_linux.py:2896` iterates:

```python
for sub in ['cu13', 'cudnn', 'cublas', 'cusparse', 'cusolver', 'cufft', 'curand', 'nccl']:
```

but `nvidia/` also contains **`cusparselt`** and **`nvshmem`** (and `nvtx`, `cufile`,
`cuda_runtime`, `cuda_nvrtc`, `cuda_cupti`, `nvjitlink`). They are not added. Nothing
currently fails — torch's own `torch/lib` RPATH resolves them — but if a future torch
build starts dlopen-ing `libcusparseLt.so` lazily, this list is where to add it.

---

## 6. Service / run

```bash
sudo systemctl start tips-server     # unit installed at /etc/systemd/system/, currently disabled+inactive
journalctl -u tips-server -f
```

Manual: `./start_server.sh` (activates `venv`, runs on **7863**, opens ngrok).

Memory guards live in the unit + a drop-in (`memory.conf`, `restart.conf`):
`MemoryHigh=11G`, `MemoryMax=13G`, `OOMPolicy=stop`, with
`LD_PRELOAD=libjemalloc.so.2` + `PYTHONMALLOC=malloc` so freed pages actually return
to the OS.

---

## 7. Rebuilding from scratch (if a venv is ever lost)

Order matters. torch **must** land before the CUDA-compiled extensions.

```bash
# ── main venv ──────────────────────────────────────────────────────────
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# torch FIRST, from the cu130 index — sm_120/Blackwell needs CUDA 13
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# mamba kernels compile against the torch you just installed
pip install causal-conv1d mamba-ssm --no-build-isolation --no-cache-dir

# everything else (requirements.txt deliberately does NOT pin torch)
pip install -r requirements.txt

# ── TGN venv ───────────────────────────────────────────────────────────
python3.11 -m venv ToothGroupNetwork/venv_tgn
source ToothGroupNetwork/venv_tgn/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install open3d trimesh scikit-learn scipy numpy
./build_pointops.sh          # compiles the CUDA pointops egg — needs nvcc
```

`requirements.txt` intentionally carries **no torch pin** ("DO NOT INSTALL HERE") so
that a stray `pip install -r` can never downgrade you off cu130. Keep it that way.

Verify:

```bash
venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
# expect: 2.11.0+cu130 [... 'sm_120']
```

---

## 8. Backing up / transferring the venvs

**See [`TRANSFER.md`](TRANSFER.md).** Short version: the external SSD is **exFAT**,
which cannot store symlinks, and a venv's `bin/python` *is* a symlink. Copying a venv
onto it silently produces the broken `venv_tgn` / `server_env_3.11` you see in this
repo. **Do not copy venvs to the SSD — regenerate them from §7 instead.**
