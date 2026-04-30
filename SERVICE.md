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

## Run manually

```bash
./start_server.sh
```
