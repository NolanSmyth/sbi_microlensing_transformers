# Transformer Embeddings for Fast Microlensing Inference

This repository contains the complete pipeline for the paper *"Transformer Embeddings for Fast Microlensing Inference"*. It provides tools to simulate, train, and evaluate a neural posterior estimator for characterizing gravitational microlensing events, with a focus on free-floating planets. The pipeline leverages a Transformer-based architecture to handle noisy, sparse, and irregularly-sampled astronomical time-series data.

## Features

-   **End-to-End Pipeline**: From data simulation to model evaluation and application on real astronomical data.
-   **High-Fidelity Simulations**: Generates microlensing events using `VBMicrolensing`, including finite-source effects.
-   **Robust Data Augmentation**: On-the-fly augmentation simulates realistic observing conditions, including seasonal gaps, dropouts, and photometric noise.
-   **Modern Architecture**: A Transformer encoder learns a summary representation of the light curve, naturally handling irregular time sampling and variable-length inputs.
-   **Fast & Accurate Inference**: Amortized inference provides posterior estimates orders of magnitude faster than traditional methods like MCMC.
-   **Rigorous Validation**: Includes code for performance validation using injected-recovered plots and coverage diagnostics.
-   **Real-World Application**: A dedicated script to analyze the short-duration FFP candidate event KMT-2019-BLG-2073.

## The Model

The pipeline estimates the posterior distribution for 5 key physical parameters of a finite-source, point-lens (FSPL) microlensing event:

| Parameter                 | Symbol            | Description                            |
| ------------------------- | ----------------- | -------------------------------------- |
| Time of Closest Approach  | $t_0$             | The time of peak magnification.        |
| Impact Parameter          | $u_0$             | Minimum lens-source separation.        |
| Einstein Crossing Time    | $\log_{10}(t_E)$  | Timescale of the event (log scale).    |
| Normalized Source Radius  | $\log_{10}(\rho)$ | Source radius in units of $\theta_E$ (log scale). |
| Blend Flux Fraction       | $f_s$             | Fraction of baseline flux from the lensed source. |

## Project Structure

```
.
├── 1_generate_base_simulations.py  # Step 1: Create a master set of noiseless simulations.
├── 2_train_model.py                # Step 2: Train the Transformer & Normalizing Flow.
├── 3_eval_ivr_tarp.py              # Step 3: Evaluate model accuracy and calibration.
├── 3b_see_outliers.py              # Diagnostic script to visualize failure modes.
├── 4_analyze_kmt_2073.py           # Step 4: Apply the trained model to real data.
├── 5_mcmc_compare.py               # Compares NPE performance against emcee.
├── 6_create_schematic.py           # Generates the methodology schematic for the paper.
├── config.py                       # Central configuration for paths, priors, and hyperparameters.
├── model.py                        # Defines the TransformerEmbeddingNet architecture.
├── utils.py                        # Core simulation, data processing, and helper functions.
├── drp.py                          # Implementation of the TARP diagnostic.
├── run_job.sh                      # Example SLURM script to run the full pipeline.
└── test_overfit.py                 # A sanity check to test the training loop.
```

## Getting Started

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <REPO.git>
    cd <REPO NAME>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    ```

3.  **Install Python dependencies:**
    ```
    pip install -r requirements.txt
    ``` 

## Running the Pipeline

The pipeline is designed to be run sequentially. Key settings can be adjusted in `config.py`.

### Step 1: Generate Base Simulations

This script creates a master dataset of dense, noiseless microlensing light curves from the prior. These will be augmented on-the-fly during training.

```bash
python 1_generate_base_simulations.py --num-sims 100000
```

### Step 2: Train the Model

This script trains the neural posterior estimator. It uses a `MicrolensingAugmentationDataset` to generate fresh, noisy, and gappy light curves for every training sample. The best model state is saved based on validation loss.

```bash
python 2_train_model.py --max-epochs 400 --patience 30 --lr 1e-4
```

### Step 3: Evaluate Performance

After training, evaluate the model's accuracy and calibration on a test set of simulated events. This script generates the injected-vs-recovered plots and the TARP diagnostic figure.

```bash
python 3_eval_ivr_tarp.py --num-events 2000 --num-samples 5000
```
You can use `3b_see_outliers.py` to visually inspect cases where the model performed poorly, which is useful for debugging and understanding model limitations.

### Step 4: Analyze KMT-2019-BLG-2073

Apply the trained posterior estimator to real astronomical data for the FFP candidate KMT-2019-BLG-2073. This script will download the data (if needed), process it, and generate a corner plot of the inferred posterior and a figure showing the model fit.

```bash
python 4_analyze_kmt_2073.py
```

### (Optional) Compare with MCMC

To quantify the speed and accuracy gains, compare the NPE results against a traditional MCMC fit using `emcee`.

```bash
python 5_mcmc_compare.py
```

## Configuration

All major parameters are centralized in `config.py` for easy modification:

-   `TOTAL_DURATION`, `MAX_NUM_POINTS`: Simulation and data window settings.
-   `PRIOR_LOW`, `PRIOR_HIGH`: Defines the parameter space for simulation and inference.
-   `AUG_*_RANGE`: Controls the on-the-fly data augmentation (gaps, dropout).
-   `D_MODEL`, `N_HEAD`, `N_LAYERS`: Hyperparameters for the Transformer architecture.
-   `BATCH_SIZE`, `MAX_EPOCHS`, `LEARNING_RATE`: Training settings.
-   `*_PATH`: File paths for data, models, and figures.

## Usage on a SLURM Cluster

The `run_job.sh` script provides a template for running the simulation and training steps on a cluster managed by SLURM. You can submit it directly:

```bash
sbatch run_job.sh
```