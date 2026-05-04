# TIPs Server Service

## Install

```bash
sudo cp tips-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tips-server
sudo systemctl start tips-server
```

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
1. A Python API server on port 7863 (using the virtual environment at `/home/oaiz/envs/server_env_3.11`)
2. ngrok tunnel to expose the local server publicly

Environment variables:
- `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2` - memory optimization
- `PYTHONMALLOC=malloc` - Python memory allocation setting

## Prerequisites

- Python 3.11 with virtual environment at `/home/oaiz/envs/server_env_3.11`
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
The server is still initializing (loading models). Wait for initialization to complete.

## Run manually

```bash
./start_server.sh
```
