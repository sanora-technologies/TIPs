# TIPs — Transferring to the external SSD

Audited **2026-07-13**.

## The short answer

**What didn't transfer: the 4 symlinks inside each venv.** The most important one is
`venv/bin/python`.

**Was it important? No — and you should not try to fix it.** Do not transfer the venvs
at all. They are **41 GB of the ~75 GB you copied**, they are *machine-specific*
(compiled against this box's GPU, driver and CUDA), and a venv copied to this SSD can
**never** be made to work. Regenerate them from
[`VENVS.md` §7](VENVS.md#7-rebuilding-from-scratch-if-a-venv-is-ever-lost) instead — it's
a 10-minute `pip install`.

**What you must NOT lose** is model weights and data, and those copied fine. See
[§4](#4-what-actually-matters-in-a-backup).

---

## 1. Why it failed: the SSD is exFAT

```
/dev/sda1  exfat  120G  110G used  9.7G free   →  /run/media/oaiz/SSD128
```

exFAT is a Microsoft filesystem with **no concept of a symbolic link**. Proven on the
live device:

```bash
$ ln -s /usr/bin/python3.11 /run/media/oaiz/SSD128/.__symlink_test
ln: Permission denied
```

A Python venv is *built* out of symlinks. Each one contains exactly **4**:

```
venv/lib64            -> lib
venv/bin/python       -> python3.11
venv/bin/python3      -> python3.11
venv/bin/python3.11   -> /usr/bin/python3.11     ← the interpreter itself
```

When you copied to exFAT, those 4 links could not be written. Depending on the copy
tool they were either skipped outright or replaced with empty stubs. **Without
`bin/python` a venv is not a venv** — nothing can activate it, nothing can run it.

That is the entire story. Nothing was corrupted; one specific *kind* of file simply
cannot exist on that disk.

### The "empty files" are a red herring

You may notice ~900 zero-byte files in each venv and assume the copy shredded them.
It didn't. They are `__init__.py` and `py.typed` package markers, which are *supposed*
to be empty:

| | local `venv` | SSD `venv` |
|---|---|---|
| 0-byte files | 915 | 915 |
| total files | 74,761 | 74,761 |

Identical. The copy was **byte-for-byte complete except for the 4 symlinks.**

---

## 2. The collateral damage this already caused

The failed transfer left two dead directories **in your working repo** — these are
copies that came *back* off the SSD (or were made in the same doomed way), and they
are why you have "three venvs":

| Directory | Size | What's wrong |
|---|---|---|
| `TIPs/venv_tgn` | **8.7 G** | 0 symlinks. All 39 files in `bin/` are **0 bytes**. Its own `pyvenv.cfg` admits it was created as `ToothGroupNetwork/venv_tgn` — it's a misplaced copy. |
| `TIPs/server_env_3.11` | **11 G** | Has **no `bin/` directory at all**. Only `lib/` and `include/` survived. |

Neither is referenced by `api_server_linux.py`, `start_server.sh`, `.env`, or the
systemd unit. Both are already in `.gitignore`. **Nothing serves from them.**

```bash
# reclaim ~20 GB — safe, verified unreferenced
rm -rf /home/oaiz/Documents/Sanora/dianexea_stack/TIPs/venv_tgn
rm -rf /home/oaiz/Documents/Sanora/dianexea_stack/TIPs/server_env_3.11
```

> The two that actually serve — `TIPs/venv` and `TIPs/ToothGroupNetwork/venv_tgn` —
> are healthy, have their 4 symlinks, and both pass a live CUDA smoke test. Don't
> touch those.

There is also a **21 GB copy of TIPs sitting in the SSD's trash**
(`/run/media/oaiz/SSD128/.Trash-1000/files/TIPs`, deleted 2026-07-12) from the first
aborted attempt. Emptying the trash frees 21 GB on a disk that is currently **92 % full
with only 9.7 GB left** — which is very likely why the transfer ran out of room and
was abandoned partway.

---

## 3. How to transfer it properly

### Option A — don't copy the venvs (recommended)

Back up the project, skip the environments, rebuild them on the far side. This is the
right answer for a venv on *any* filesystem: they are derived artifacts, not source.

```bash
cd /home/oaiz/Documents/Sanora/dianexea_stack

rsync -a --info=progress2 \
  --exclude 'venv/' \
  --exclude 'venv_tgn/' \
  --exclude 'server_env_3.11/' \
  --exclude '__pycache__/' \
  TIPs/ /run/media/oaiz/SSD128/TIPs/
```

Drops ~41 GB of venv and copies cleanly, because nothing left in the tree is a symlink.
Rebuild with [`VENVS.md` §7](VENVS.md#7-rebuilding-from-scratch-if-a-venv-is-ever-lost).

### Option B — if you truly need a byte-exact venv archive

Put it in a **tar**. A tarball is a single regular file, so exFAT never sees the
symlinks — tar stores them *inside* the archive and restores them on extract.

```bash
cd /home/oaiz/Documents/Sanora/dianexea_stack

# venvs only, preserving symlinks + permissions
tar -czf /run/media/oaiz/SSD128/tips-venvs.tar.gz \
    TIPs/venv TIPs/ToothGroupNetwork/venv_tgn

# restore on the target machine (same GPU/driver/CUDA, or it won't run)
tar -xzf /run/media/oaiz/SSD128/tips-venvs.tar.gz -C /path/to/dest/
```

Caveats, in order of how much they will hurt you:

1. **A venv is not portable.** `bin/python3.11 -> /usr/bin/python3.11` is an absolute
   path, and `pyvenv.cfg` hardcodes the original directory. Restore it to a *different*
   path or a machine without Python 3.11.15 at that exact location and it breaks.
2. The compiled CUDA extensions (`pointops`, mamba kernels) are built for **sm_120 /
   cu130 / driver 595**. On any other GPU they are dead weight.
3. You need ~15 GB free to write the archive. **You have 9.7 GB.** Empty the SSD trash
   (§2) first.

Option B is only worth it for an exact-clone/disaster-recovery snapshot of *this* box.
For moving the project anywhere else, use **Option A**.

### Option C — reformat the SSD

If this drive is only ever going to be used with Linux, reformat it **ext4** and the
whole class of problem disappears (symlinks, permissions, ownership all work). You lose
Windows/macOS interoperability. **This erases the disk** — copy the 75 GB off first.

---

## 4. What actually matters in a backup

The venvs are reproducible. These are **not** — verify they made it:

| | |
|---|---|
| `models/` | nnU-Net weights (DentalSegmentator) |
| `nnResults/` | trained checkpoints (`Dataset810_root_binarySDM`) |
| `ToothGroupNetwork/ckpts/` | `tgnet_fps`, `tgnet_bdl` |
| `.env` | secrets — API tokens, webhook HMAC. **Not in git** (`.gitignore` has `.env*`) |
| `api_server_linux.py` | the server itself |

Note `.gitignore` also excludes `*.pt`, `*.pth`, `*.ckpt`, `*.pkl`, `nnResults/`,
`models/` — **so none of the model weights are in git.** The SSD copy is their only
backup. That is what the transfer was actually for; that part succeeded.

---

## 5. Quick verification after any transfer

```bash
# on the destination — the one check that matters
venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.11.0+cu130 True
```

If you get `No such file or directory: venv/bin/python`, the symlinks were dropped —
you copied to a filesystem that can't hold them. Go back to §3.
