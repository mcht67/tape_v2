## 0. Preparations on Cluster (HPC)
<details>
<summary>Show instructions...</summary>

Login into HPC from terminal with your TUB-Account credentials:
```bash
ssh <TUB-Account>@sshgate.tu-berlin.de
ssh gateway.hpc.tu-berlin.de
```

### Setup directories

<details>
<summary>Show more...</summary>

Create a personal subdirectory on /scratch, since space is limited on the user home directory:
```sh
cd /scratch
mkdir <username>
```
Update the global environment file [global.env](./../global.env) with the path to your HPC scratch directory:

```env
TUSTU_HPC_DIR=/scratch/<username>
```

Restrict permissions on your subdirectory (Optional):
```sh
chmod 700 <username>/
```

> **Info:** See [HPC Documentation](https://hpc.tu-berlin.de/doku.php?id=hpc:hardware:beegfs) for general information about the filesystem on [HPC Cluster - ZECM, TU Berlin](https://www.tu.berlin/campusmanagement/angebot/high-performance-computing-hpc).

Set up a temporary directory and hugging face and singularity cache directories on `/scratch` to get more space for temporary files. 
```sh
mkdir -p /beegfs/scratch/<TUB-username>/tmp
mkdir -p /beegfs/scratch/<TUB-username>/.singularity
mkdir -p /beegfs/scratch/<TUB-username>/.cache/huggingface/hub
mkdir -p /beegfs/scratch/<TUB-username>/.cache/huggingface/datasets
```

Then add the `TMPDIR` environment variable to your `.bashrc` so that singularity and other applications use this directory for temporary files. These can get quite large as singularity uses them to extract the image and run the container. Then change the cache directory of singularity with the `SINGULARITY_CACHEDIR` environment variable as well as hugging face directories with 'HF_HOME', 'HF_HUB_CACHE' and 'HF_DATASETS_CACHE'. (replace <TUB-username> with your actual username!)
```sh
cat >> ~/.bashrc << 'EOF'

# HPC Cluster cache configuration
export TMPDIR=/beegfs/scratch/<TUB-username>/tmp
export SINGULARITY_CACHEDIR=/beegfs/scratch/<TUB-username>/.singularity

# Hugging Face cache configuration
export HF_HOME=/beegfs/scratch/<TUB-username>/.cache/huggingface
export HF_HUB_CACHE=/beegfs/scratch/<TUB-username>/.cache/huggingface/hub
export HF_DATASETS_CACHE=/beegfs/scratch/<TUB-username>/.cache/huggingface/datasets
EOF
```

Apply changes and verify:
```sh
source ~/.bashrc

echo "TMPDIR: $TMPDIR"
echo "SINGULARITY_CACHEDIR: $SINGULARITY_CACHEDIR"
echo "HF_HOME: $HF_HOME"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"
echo "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
```

Update hugging face cache paths in your global.env if necessary:
```env
HF_HOME=/beegfs/scratch/.cache/huggingface
HF_HUB_CACHE=/beegfs/scratch/.cache/huggingface/hub
HF_DATASETS_CACHE=/beegfs/scratch/.cache/huggingface/datasets
```
</details>

### Configure Git

<details>
<summary>Show more...</summary>

Navigate to your user directory 
```sh
cd ~
```

Create .ssh dir and restrict and check permissions
```sh
mkdir -p ~/.ssh
chmod 700 ~/.ssh
ls -ld ~/.ssh
```

Generate ssh key
```sh
cd ~/.ssh
ssh-keygen -t ed25519 -C "github-2026"
```
If prompted set 'id_ed25519' as the name for your key, to make ssh automatically find the key, and enter a passphrase.

Copy the public ssh key and add it to GitHub via web interface.
IMPORTANT: never copy and post id_ed25519 (private)
```sh
cat id_ed25519.pub
```

Test SSH connection
```sh
ssh -T git@github.com
```

Configure git to always use ssh
```sh
git config --global url."git@github.com:".insteadOf "https://github.com/"
```

Clone into your scratch directory
```sh
cd /scratch/your_username/
git clone --recurse-submodules https://github.com/mcht67/Polyphonic-Bird-Call-Dataset.git
```

Create directory for logs
```sh
mkdir -p logs/slurm/
```

</details>

<!-- Assuming you have already configured Git on the HPC cluster, clone your Git repository to `/scratch/<username>`:

```sh
cd <username>
git clone git@github.com:<github_user>/<repository_name>.git
```

> **Info:** On the hpc cluster, the first time you log in a ssh key is generated for you (`~/.ssh/id_rsa`). You can use this key to access your git repository. -->

<!-- ### Install git lfs

<details>
<summary>Show more...</summary>

Install latest [Linux-amd64 release](https://github.com/git-lfs/git-lfs/releases).
```bash
cd ~
wget https://github.com/git-lfs/git-lfs/releases/download/v3.7.1/git-lfs-linux-amd64-v3.7.1.tar.gz  # pick a current version
tar -xvzf git-lfs-linux-amd64-v3.7.1.tar.gz
cd git-lfs-3.7.1
sed -i 's|^prefix="/usr/local"$|prefix="$HOME/.local"|' install.sh
mkdir -p ~/.local/bin
./install.sh
```

Export the path
```bash
export PATH="$HOME/.local/bin:$PATH"
```

</details>

<!-- ### Pull changes files from git submodule "resources"
```bash
cd /scratch/username/Polyphonic-Bird-Call-Dataset/resources
git lfs pull
``` -->
</details> -->


## 4. Create local.env
Add the local.env to the repository. 
You can use this bash command. Just replace placeholders with your information.
```bash
cat > local.env << 'EOF'
# Copy this to local.env and fill in your values
# DO NOT commit local.env to git!

GIT_REPO_URL=https://github.com/mcht67/Polyphonic-Bird-Call-Dataset.git
GIT_USERNAME="Your Username"
GIT_EMAIL="your.email@example.com"
HUGGINGFACE_TOKEN="your_hf_token_here"
DOCKERHUB_USERNAME="your_dockerhub_username"
EOF
```

## Add config.local
```bash
cat > config.local<< 'EOF'
# Content of your local config.local
EOF
```

## Add secrets/dvc-token.json [GoogleDrive Authentication]
```bash
cat > secrets/dvc-token.json << 'EOF'
# Content of your secrets/dvc-token.json
EOF
```

# Add python venv to run multi_submission.py
```bash
python3 -m venv venv
. venv/bin/activate
pip install huggingface_hub dotenv omegaconf pathlib hydra
```