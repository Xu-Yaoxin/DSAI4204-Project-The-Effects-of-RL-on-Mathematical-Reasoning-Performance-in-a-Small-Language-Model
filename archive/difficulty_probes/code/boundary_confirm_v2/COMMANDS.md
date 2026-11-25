# Upload and run / 上传与执行

Run on the Mac / 在本机 Mac 运行：

```bash
rsync -avh --progress --partial \
  --exclude='outputs/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='*.log' \
  --exclude='*.pid' \
  -e "ssh -p 42172 -i $HOME/.ssh/dsai4207_4090" \
  --rsync-path="mkdir -p /root/code/boundary_confirm_v2 && rsync" \
  "/Users/xu./Library/Mobile Documents/com~apple~CloudDocs/PolyU/Y4/DSAI 4207/project/code/boundary_confirm_v2/" \
  root@xj-member.bitahub.com:/root/code/boundary_confirm_v2/
```

Run in the server terminal / 登入服务器后运行：

```bash
cd /root/code/boundary_confirm_v2 || exit
nohup bash run_confirmation_parallel.sh >> confirmation.log 2>&1 < /dev/null &
echo $! > confirmation.pid
tail -n 50 -f confirmation.log
```

`Ctrl+C` stops `tail` only. The background evaluation continues after disconnecting or closing the Mac. Repeating the launch command after the previous process has exited resumes completed answers. See README.md for sequential fallback and clean shutdown. All 800 answers must complete before analysis; inspect both worker logs if completed-answer progress pauses during a long generation.
