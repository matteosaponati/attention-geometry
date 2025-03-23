import datetime
import torch
import os

rank = int(os.environ.get("RANK", 0))

print("Hello from rank", rank)
torch.distributed.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=30))
print("Connected rank", rank)
torch.distributed.barrier()
print("Barrier passed rank", rank)

