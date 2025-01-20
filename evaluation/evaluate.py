import argparse
import os
import os.path as osp
import queue
import random
import threading
import traceback
from typing import Any, Dict, List, Union

import datetime
import json
import numpy as np
import torch
import torch.distributed as dist
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import sys
sys.path.append(".")
from evaluation.register import INFERENCES
from evaluation.datasets import build_dataset
from videollama3 import disable_torch_init, model_init, mm_infer


class BackgroundGenerator(threading.Thread):
    """
    the usage is below
    >> for batch in BackgroundGenerator(my_minibatch_iterator):
    >>    doit()
    More details are written in the BackgroundGenerator doc
    >> help(BackgroundGenerator)
    """
    def __init__(self, generator, max_prefetch=10):
        """
        This function transforms generator into a background-thead generator.
        :param generator: generator or genexp or any
        It can be used with any minibatch generator.
        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended (may raise GIL and zero-out the
        benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it
        outputs is passed through queue.
        There's no restriction on doing weird stuff, reading/writing files, retrieving
        URLs [or whatever] wlilst iterating.
        :param max_prefetch: defines, how many iterations (at most) can background generator keep
        stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until
        one of these batches is dequeued.
        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!
        Setting max_prefetch to -1 lets it store as many batches as it can, which will work
        slightly (if any) faster, but will require storing
        all batches in memory. If you use infinite generator with max_prefetch=-1, it will exceed the RAM size
        unless dequeued quickly enough.
        """
        super().__init__()
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.exit_event = threading.Event()
        self.start()

    def run(self):
        for item in self.generator:
            if self.exit_event.is_set():
                break
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class CUDADataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream = torch.cuda.Stream() # create a new cuda stream in each process

    def __iter__(self):
        self.iter = super().__iter__()
        self.iter = BackgroundGenerator(self.iter)
        self.preload()
        return self

    def _shutdown_background_thread(self):
        if not self.iter.is_alive():
            # avoid re-entrance or ill-conditioned thread state
            return
        # Set exit event to True for background threading stopping
        self.iter.exit_event.set()
        # Exhaust all remaining elements, so that the queue becomes empty,
        # and the thread should quit
        for _ in self.iter:
            pass
        # Waiting for background thread to quit
        self.iter.join()

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            frames = self.batch['video'][0][0]
            for idx, frame in enumerate(frames):
                frames[idx]['pixel_values'] = frame['pixel_values'].to(device="cuda", non_blocking=True)
                frames[idx]['image_grid_thw'] = frame['image_grid_thw'].to(device="cuda", non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)  # wait tensor to put on GPU
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

    # Signal for shutting down background thread
    def shutdown(self):
        # If the dataloader is to be freed, shutdown its BackgroundGenerator
        self._shutdown_background_thread()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "--model_path", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)

    parser.add_argument("--data-root", "--data_root", type=str, required=True)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max-frames", "--max_frames", type=int, default=128)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=8)

    parser.add_argument("--save-path", "--save_path", type=str, default=None)

    return parser.parse_args()


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def show_metrics(metrics: Dict[str, Any], benchmark: str):
    if all(isinstance(metric, dict) for metric in metrics.values()):
        for task_name, metric in metrics.items():
            show_metrics(metric, f"{benchmark}_{task_name}")
    elif all(isinstance(metric, (int, float)) for metric in metrics.values()):
        table = PrettyTable(["Task Type", "Accuracy"])
        for task_name, metric in metrics.items():
            table.add_row([task_name, round(metric, 2)])
        table.align["Task Type"] = "l"
        print(f"Results on {benchmark}:")
        print(table)
        print("\n")
    else:
        raise ValueError


def main():
    dist.init_process_group(backend="gloo", timeout=datetime.timedelta(minutes=120))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()

    seed_everything()
    args = parse_args()

    disable_torch_init()
    model_init, mm_infer = INFERENCES(args.model_path)
    model, processor, tokenizer = model_init(args.model_path, device_map={"": f"cuda:{local_rank}"})

    dataset = build_dataset(
        args.benchmark,
        data_root=args.data_root,
        processor=processor["video"],
        num_splits=dist.get_world_size(),
        split_idx=global_rank,
        fps=args.fps,
        max_frames=args.max_frames,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=lambda x: x[0],  # asume the batch_size is always 1
        pin_memory=True,
    )

    results = []
    for idx, data in enumerate(tqdm(dataloader, desc=f"Rank {global_rank}", total=len(dataloader), position=local_rank)):
        data_ids = data["data_ids"]
        instructions = data["instructions"]
        for data_id, instruction in zip(data_ids, instructions):
            try:
                response = mm_infer(
                    data["video"],
                    instruction,
                    model=model,
                    tokenizer=tokenizer,
                    modal="video",
                    do_sample=False,
                )
                prediction = dataset.process_response(data_id, response)
            except Exception as e:
                traceback.print_exc()
                print(f"Error in data_id: {data_id}")
                exit(0)

            results.append(
                {
                    "data_id": data_id,
                    "response": response,
                    "prediction": prediction,
                }
            )
    
    assert len(results) == dataset.n_samples

    del model, data
    torch.cuda.empty_cache()
    gathered_results = [None for _ in range(dist.get_world_size())]
    dist.gather_object(
        obj=results,
        object_gather_list=gathered_results if global_rank == 0 else None,
        dst=0,
    )

    if global_rank == 0:
        results = sum(gathered_results, [])
        metrics, infos = dataset.evaluate(results)

        print("\n" * dist.get_world_size())  # prevent unexpected progress bars
        show_metrics(metrics, args.benchmark)

        if args.save_path:
            os.makedirs(osp.dirname(args.save_path), exist_ok=True)
            if args.save_path.endswith(".json"):
                with open(args.save_path, "w") as f:
                    json.dump(infos, f, indent=4)
            elif args.save_path.endswith(".jsonl"):
                with open(args.save_path, "w") as f:
                    for info in infos:
                        f.write(json.dumps(info) + "\n")
            else:
                raise ValueError("Unsupported file format.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
