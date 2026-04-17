# Towards Avian Population Estimation [TAPE]

This repository provides a reproducible training pipeline for acoustic bird counting on a SLURM-based HPC cluster. It was developed as part of a thesis investigating deep learning–based methods for estimating vocal polyphony — the number of concurrently vocalizing birds in a given time window — as a proxy for population density in passive acoustic monitoring (PAM) scenarios.

The pipeline includes numerous state-of-the-art deep learning models covering both implicit (detection-based counting) and explicit (direct regression) polyphony estimation strategies, scripts for pre-computing embeddings, multi-job submission for sweeping across datasets and hyperparameter configurations, and notebooks for model evaluation. Training runs and evaluation metrics are logged to an archive that can be synced to local machine. The synthetic mixture datasets used for training are generated separately and available at [GitHub: Polyphonic-Bird-Call-Dataset](https://github.com/mcht67/Polyphonic-Bird-Call-Dataset).

# Pipeline Flowchart
```mermaid
flowchart TD
    
    A["Multi submission script"] -- 1. runs --> C
    A --2. runs--> D

    B[("☁️ HuggingFace Hub<br/>Audio Dataset")] -- Load dataset --> C

    C["Dataset preparation [embed audio]"]
    C -- Update dataset --> B   

    D["DVC Pipeline"] --runs--> E     
   
    B -- Load dataset --> E["Training"]
    subgraph dvc-experiment
        E --> F[("📊 Logs, Metrics &<br/>Checkpoints")]
        F -- dvc push --> I[("🗄️ DVC<br/>Remote")]
    end
    F -- save to --> G[("💾 HPC archive")]
    G -- manual rsync--> H[("💾 Local archive")]

    style A fill:#dbeafe,stroke:#2563eb,color:#1f2937
    style C fill:#fef3c7,stroke:#b45309,color:#1f2937
    style E fill:#fef3c7,stroke:#b45309,color:#1f2937
    style F fill:#dbeafe,stroke:#2563eb,color:#1f2937
    style H fill:#fef3c7,stroke:#b45309,color:#1f2937
    style I fill:#d1fae5,stroke:#047857,color:#1f2937
    style B fill:#ede9fe,stroke:#6d28d9,color:#1f2937
    style D fill:#ede9fe,stroke:#6d28d9,color:#1f2937
    style G fill:#ede9fe,stroke:#6d28d9,color:#1f2937
    style I fill:#d1fae5,stroke:#047857,color:#1f2937
    style B fill:#ede9fe,stroke:#6d28d9,color:#1f2937
    style D fill:#ede9fe,stroke:#6d28d9,color:#1f2937
    style G fill:#ede9fe,stroke:#6d28d9,color:#1f2937
  ```

# Setup

- [Setup on Local Machine](docs/SETUP_local.md)
- [Setup on HPC Cluster](docs/SETUP_hpc.md)

# Usage

- [Usage on Local Machine](docs/USAGE_local.md)
- [Usage on HPC Cluster](docs/USAGE_hpc.md)
- [Evaluation Notebooks](source/notebooks/README.md)

# Implementation Details
- [Implementation Details](docs/IMPLEMENTATION.md)

# Credits
This project builds on the following open-source work:

- [BirdSet](https://github.com/DBD-research-group/BirdSet) — Bird call embedding models
- [Perch](https://github.com/google-research/perch) — Bird call embedding models
- [TU Studio HPC Cluster ML Workflow](https://github.com/tu-studio/hpc-cluster-ml-workflow/tree/main) — SLURM pipeline architecture
- [Polyphonic Bird Call Dataset](https://github.com/mcht67/Polyphonic-Bird-Call-Dataset) - Polyphonic bird call data

# Citations
**BirdSet**: Rauch et al., 2024. *BirdSet: A Large-Scale Dataset for Audio Classification in Avian Bioacoustics*. arXiv:2403.10380.
    <details>
    <summary>BibTeX</summary>
    bibtex@misc{rauch2024birdset,
        title={BirdSet: A Large-Scale Dataset for Audio Classification in Avian Bioacoustics}, 
        author={Lukas Rauch et al.},
        year={2024},
        eprint={2403.10380},
        archivePrefix={arXiv},
        url={https://arxiv.org/abs/2403.10380}, 
    }
    </details>

**Perch 2.0**: van Merriënboer et al., 2026. *Perch 2.0: The Bittern Lesson for Bioacoustics*. arXiv:2508.04665.
<details>
    <summary>BibTeX</summary>
    @misc{vanmerriënboer2026perch20bitternlesson,
      title={Perch 2.0: The Bittern Lesson for Bioacoustics}, 
      author={Bart van Merriënboer and Vincent Dumoulin and Jenny Hamer and Lauren Harrell and Andrea Burns and Tom Denton},
      year={2026},
      eprint={2508.04665},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.04665}, 
}
</detauils>


