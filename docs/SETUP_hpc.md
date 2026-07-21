# Setup [HPC Cluster]

This guide describes the setup on the HPC Cluster of the TU Berlin. For other SLURM-based HPC Clusters this might has to be adapted.

> **Info:** See [HPC Documentation](https://hpc.tu-berlin.de/doku.php?id=hpc:hardware:beegfs) for general information about the filesystem on [HPC Cluster - ZECM, TU Berlin](https://www.tu.berlin/campusmanagement/angebot/high-performance-computing-hpc).

## 1. Login

Login into HPC from terminal with your TUB-Account credentials:
```bash
ssh <TUB-Account>@sshgate.tu-berlin.de
ssh gateway.hpc.tu-berlin.de
```

## 2. Setup directories for temporary data
Create a personal subdirectory on /beegfs/scratch, since space is limited on the user home directory:
```sh
cd /beegfs/scratch
mkdir <username>
```
<!-- Once automatic rsync is used -> Update the global environment file [global.env](./../global.env) with the path to your HPC scratch directory:

```env
HPC_DIR=/scratch/<username>
``` -->

Restrict permissions on your subdirectory (Optional):
```sh
chmod 700 <username>/
```

Set up a temporary directory and hugging face and singularity cache directories on `/beegfs/scratch` to get more space for temporary files. (replace <TUB-username> with your actual username!)
```sh
mkdir -p /beegfs/scratch/<TUB-username>/tmp
mkdir -p /beegfs/scratch/<TUB-username>/.singularity
mkdir -p /beegfs/scratch/<TUB-username>/.cache/huggingface/hub
mkdir -p /beegfs/scratch/<TUB-username>/.cache/huggingface/datasets
```

Then add the `TMPDIR` environment variable to your `.bashrc` so that singularity and other applications use this directory for temporary files. These can get quite large as singularity uses them to extract the image and run the container. Then change the cache directory of singularity with the `SINGULARITY_CACHEDIR` environment variable.  
```sh
cat >> ~/.bashrc << 'EOF'

# HPC Cluster cache configuration
export TMPDIR=/beegfs/scratch/<TUB-username>/tmp
export SINGULARITY_CACHEDIR=/beegfs/scratch/<TUB-username>/.singularity
EOF
```

Apply changes and verify:
```sh
source ~/.bashrc

echo "TMPDIR: $TMPDIR"
echo "SINGULARITY_CACHEDIR: $SINGULARITY_CACHEDIR"
```

Hugging face directory environment variables are set from within the scripts with 'HF_HOME', 'HF_HUB_CACHE' and 'HF_DATASETS_CACHE' being defined in global.env. 

Update hugging face cache paths in your global.env if necessary:
```env
HF_HOME=/beegfs/scratch/<TUB-username>/.cache/huggingface
HF_HUB_CACHE=/beegfs/scratch/<TUB-username>/.cache/huggingface/hub
HF_DATASETS_CACHE=/beegfs/scratch/<TUB-username>/.cache/huggingface/datasets
```

## 3. Configure Git

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

## 4. Clone repository

Clone into your scratch directory
```sh
cd /beegfs/scratch/your_username/
git clone https://github.com/mcht67/tape_v2.git
```

Create directory for logs
```sh
mkdir -p logs/slurm/
```

## 5. Set git config

Set git config
```bash
git config --global user.name "$GIT_USERNAME"
git config --global user.email "$GIT_EMAIL"
git config --global safe.directory "$REPO_DIR"
```

## 5. Update global.env

Update the project name in global.env:
```file
# Name of the project
PROJECT_NAME=train-bird-models
```

## 6. Create local.env
Add the local.env to the repository. 
You can use this bash command. Just replace placeholders with your information.
```bash
cat > local.env << 'EOF'
# Copy this to local.env and fill in your values
# DO NOT commit local.env to git!

HUGGINGFACE_TOKEN="your_hf_token_here"
DOCKERHUB_USERNAME="your_dockerhub_username"
EOF
```

## 7. Setup dvc remote

### Setup with google drive locally
First
[setup Google Drive Project for dvc remote](https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended).

Then
[connect dvc with Google drive](https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive)
like this:
```sh
mkdir secrets
dvc remote add -d myremote gdrive://YOUR_FOLDER_ID
dvc remote modify --local myremote gdrive_acknowledge_abuse true
dvc remote modify --local myremote gdrive_client_id  "actual gdrive client id"
dvc remote modify --local myremote gdrive_client_secret "actual gdrive client secret"
dvc remote modify --local myremote gdrive_user_credentials_file ./secrets/dvc-token.json

dvc pull
``` 
To reset authentication delete dvc-token.json. This will open authentication in the browser on the next dvc pull.
The refresh token is valid for 7 days. If you want it to persist, you have to publish the Google Cloud App.

### Move your local setup to the Cluster

Move your local dvc config to the Cluster
```bash
cat > .dvc/config.local<< 'EOF'
# Content of your .dvc/config.local
EOF
```

Move your local dvc-token to the Cluster
```bash
cat > secrets/dvc-token.json << 'EOF'
# Content of your secrets/dvc-token.json
EOF
```

# Make files executable
Change permissions to be able to run files from bash
```bash
chmod +x build_singularity.sh
chmod +x multi_submission.py
```

## 8. Setup Docker
Sign Up for Docker Hub: If you do not have an account, register at Docker Hub.
Configure GitHub Secrets: In your GitHub repository, go to Settings → Security → Secrets and variables → Actions → New repository secret, and add secrets for:
DOCKER_USERNAME: Your Docker Hub username
DOCKER_PASSWORD: Your Docker Hub password

Your docker image will be buidl automatically by GitHub actions once you opush your changes in requirements, Dockerfile or global.env. The singularity container will also be build automatically.