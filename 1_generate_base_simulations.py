# 1_generate_base_simulations.py
import argparse, torch
from tqdm import trange
from config import PRIOR, BASE_SIMS_PATH
from utils import generate_base_simulation, ensure_dir, set_global_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-sims", type=int, default=50_000)
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10_000,
        help="Process sims in chunks to keep memory stable",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_global_seeds(args.seed)
    ensure_dir(BASE_SIMS_PATH)

    n, shard = args.num_sims, args.shard_size
    print(f"Generating {n} base simulations (shard size {shard})...")

    thetas = PRIOR.sample((n,))  # shape [n, D]
    xs_shards = []

    for start in trange(0, n, shard, desc="Simulating"):
        end = min(start + shard, n)
        xs_list = []
        for i in range(start, end):
            theta_i = thetas[i]
            sim_out = generate_base_simulation(theta_i)
            xs_list.append(sim_out)
        xs_shards.append(torch.stack(xs_list, dim=0))

    xs = torch.cat(xs_shards, dim=0)  # [n, ...]
    print(f"Shape of final xs tensor: {tuple(xs.shape)}")
    print(f"Saving base simulations to {BASE_SIMS_PATH}...")
    torch.save({"thetas": thetas, "xs": xs}, BASE_SIMS_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
