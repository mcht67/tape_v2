<!-- # Run exp_workflow.sh

Run from within singularity container -> Not helpful running on frontend node???
```bash
singularity exec /path/to/singularity_image ./exp_workflow.sh
``` -->

# Run multi_submission.py

Run from within singularity container
```bash
chmod +x multi_submission.py
singularity exec \
    --bind $HF_HOME:$HF_HOME \
    --bind /beegfs/scratch/cohrt/tape_v2 \
    --pwd /beegfs/scratch/cohrt/tape_v2 \
    train-bird-models-image-latest \
    python3 ./multi_submission.py
```