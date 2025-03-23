import datetime
import torch
import os


print("Distributed available", torch.distributed.is_available())
print("Distributed available NCCL", torch.distributed.is_nccl_available())
print("Distributed available GLOO", torch.distributed.is_gloo_available())
print("Distributed available MPI", torch.distributed.is_mpi_available())





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank",
                        type=int,
                        default=0,
                        dest='rank',
                        )
    parser.add_argument("--backend",
                        type=str,
                        default="nccl",
                        dest='backend',
                        )

    args = parser.parse_args()


    rank = int(os.environ.get("RANK", 0))
    print("Hello from rank", rank)
    torch.distributed.init_process_group(backend=args.backend, timeout=datetime.timedelta(seconds=30))
    print("Connected rank", rank)
    torch.distributed.barrier()
    print("Barrier passed rank", rank)

