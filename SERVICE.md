# TIPs Server Service

## Install

```bash
sudo cp /home/oaiz/Documents/Sanora/dianexea_stack/TIPs/tips-server.service /etc/systemd/system/tips-server.service
sudo systemctl daemon-reload
sudo systemctl enable --now tips-server.service
```

- `enable` → starts automatically on every reboot / PC start
- `--now`  → also starts it right now (no reboot needed to test)

Verify it came up:

```bash
systemctl status tips-server
journalctl -u tips-server -f
```

Watch the log until `[server] YOLO warmup done` then `[server] Worker ready`
(~40s after start). The ngrok static URL is live as soon as the process is up.

## Commands

| Action | Command |
|--------|---------|
| Start | `sudo systemctl start tips-server` |
| Stop | `sudo systemctl stop tips-server` |
| Restart | `sudo systemctl restart tips-server` |
| Status | `sudo systemctl status tips-server` |
| Logs | `journalctl -u tips-server -f` |
| Disable autostart | `sudo systemctl disable tips-server` |

## Configuration

The systemd service file (`tips-server.service`) should be located at:
```
/etc/systemd/system/tips-server.service
```

**Important:** Ensure the paths in the service file match your installation directory:
- `WorkingDirectory`: points to the TIPs project directory
- `ExecStart`: points to the `start_server.sh` script

## What the service does

The `tips-server` service runs:
1. A Python API server on port 7863 (using the virtual environment at `/home/oaiz/Documents/Sanora/dianexea_stack/TIPs/venv`)
2. ngrok tunnel to expose the local server publicly

Environment variables:
- `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2` - memory optimization
- `PYTHONMALLOC=malloc` - Python memory allocation setting

## Prerequisites

- Python 3.11 with virtual environment at `/home/oaiz/Documents/Sanora/dianexea_stack/TIPs/venv`
- ngrok installed
- jemalloc library (`libjemalloc.so.2`)

## Troubleshooting

### Service fails with exit code 203/EXEC
This means the ExecStart path is incorrect or the script is not executable.

**Fix:**
1. Verify the path to `start_server.sh` in `/etc/systemd/system/tips-server.service`
2. Ensure the script is executable: `chmod +x start_server.sh`
3. Reload systemd: `sudo systemctl daemon-reload`
4. Restart the service: `sudo systemctl restart tips-server`

### Check service logs
```bash
journalctl -u tips-server -f
```

### Service shows "workers not ready"
The server is still initializing. Startup takes ~40s: ~3s model load + ~35s YOLO
warmup. The warmup pays a one-time Blackwell-GPU cuDNN kernel-build cost up front
so the first OPG job doesn't eat ~35s — expected, once per boot. Wait for
`[server] Worker ready` in the logs.

### ngrok "command not found" in the journal
systemd uses a minimal PATH. Pin the full path in `start_server.sh`: change
`ngrok http 7863` to `/usr/local/bin/ngrok http 7863`. (The venv `python` already
uses an absolute path, so only ngrok is affected.)

## Run manually

```bash
./start_server.sh
```
