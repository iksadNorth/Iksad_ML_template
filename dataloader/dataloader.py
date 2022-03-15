from torch.utils.data import DataLoader, Dataset, Sampler

from typing import Callable, Optional, T_co, Callable, Sequence

from config.adapter import adapter


class BaseLoader(DataLoader): 
    @adapter
    def __init__(self, 
                 dataset: Dataset, 
                 batch_size: Optional[int] = 1, 
                 shuffle: bool = False, 
                 sampler: Optional[Sampler] = None, 
                 batch_sampler: Optional[Sampler[Sequence]] = None, 
                 num_workers: int = 0, 
                 collate_fn: Optional[Callable] = None, 
                 pin_memory: bool = False, 
                 drop_last: bool = False, 
                 timeout: float = 0, 
                 worker_init_fn: Optional[Callable] = None, 
                 multiprocessing_context=None, 
                 generator=None, 
                 prefetch_factor: int = 2,
                 persistent_workers: bool = False 
                 ):
        super().__init__(dataset, batch_size, shuffle, 
                         sampler, batch_sampler, num_workers, 
                         collate_fn, pin_memory, drop_last, timeout, 
                         worker_init_fn, multiprocessing_context, generator, 
                         prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
