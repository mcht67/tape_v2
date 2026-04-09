## Local Run on Docker with GitHub Authentication via SSH

Create ssh-key and add it to GitHub.

Add key to ssh-agent:
```bash
ssh-add ~/.ssh/id_ed25519
```

```bash
docker run --rm \
  -v $SSH_AUTH_SOCK:$SSH_AUTH_SOCK \
  -e SSH_AUTH_SOCK=$SSH_AUTH_SOCK \
  -v $(pwd):/home/app \
  train-bird-models \
  bash -c "mkdir -p ~/.ssh && ssh-keyscan -H github.com >> ~/.ssh/known_hosts 2>/dev/null && ./exp_workflow.sh"
```