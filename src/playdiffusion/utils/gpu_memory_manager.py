import time
import gc
import torch
import asyncio
import threading

class GPUMemoryManager:
    def __init__(self, threshold_percent, min_interval_seconds):
        """
        Initialize the GPU memory manager.

        Args:
            threshold_percent (float): Default memory usage threshold to trigger cleanup
            min_interval_seconds (float): Default minimum seconds between cleanups
        """
        self.threshold_percent = threshold_percent
        self.min_interval_seconds = min_interval_seconds
        self.max_interval_seconds = 5 * 60 # Cleanup every 5 minutes if no other condition is met

        self.fragmentation_threshold = 0.7
        self.segment_count_threshold = 350

        self.last_cleanup_time = time.perf_counter()
        self.second_until_gc = 20 * 60 # 20 minutes until next GC
        self.lock = threading.Lock()

    def is_memory_fragmented(self):
        try:
            stats = torch.cuda.memory_stats()

            allocated_bytes = stats.get('allocated_bytes.all.current', 0)
            reserved_bytes = stats.get('reserved_bytes.all.current', 0)
            segment_count = stats.get('segment.all.current', 0)

            ratio = 0
            if reserved_bytes > 0:
                ratio = 1.0 - (allocated_bytes / reserved_bytes)

            print(f"Memory fragmentation metrics - Ratio: {ratio:.4f}, "
                f"Segments: {segment_count}")

            # Condition 1: Fragmentation ratio exceeds threshold
            if ratio > self.fragmentation_threshold:
                print(f"Fragmentation: {ratio:.4f} > {self.fragmentation_threshold}")
                return True
            # Condition 2: Excessive memory segments
            elif segment_count > self.segment_count_threshold and ratio > 0.5:
                print(f"Excessive segments: {segment_count} > {self.segment_count_threshold}")
                return True

            return False
        except Exception as e:
            print(f"Error checking fragmentation: {e}")
            return False

    def defragment_memory(self, free):
        try:
            # Force allocator to coalesce memory blocks
            tensor_size = int(free * 0.7)
            if tensor_size > 1024:  # Only if we have enough free memory
                temp_tensor = torch.empty(tensor_size, dtype=torch.uint8, device=torch.cuda.current_device())
                del temp_tensor
                torch.cuda.empty_cache()
                print("Performed memory defragmentation")
        except Exception as e:
            print(f"Error in defragmentation: {e}")


    def check_and_cleanup(self):
        threshold = self.threshold_percent
        min_interval = self.min_interval_seconds
        max_interval = self.max_interval_seconds

        try:
            with self.lock:
                current_time = time.perf_counter()
                time_since_last_cleanup = current_time - self.last_cleanup_time
                if time_since_last_cleanup < min_interval:
                    return None, None, None

                device = torch.cuda.current_device()
                free, total = torch.cuda.mem_get_info(device)
                percent = 100 - (free / total) * 100

                should_gc = False
                should_cleanup = percent > threshold or self.is_memory_fragmented() or time_since_last_cleanup > max_interval
                if should_cleanup:
                    if percent > threshold:
                        print(f"GPU mem({percent:.2f}%) > ({threshold}%). Performing cleanup...")
                    elif time_since_last_cleanup > max_interval:
                        print(f"Interval ({time_since_last_cleanup:.2f}s) > ({max_interval}s). Performing cleanup...")
                    else:
                        print(f"Fragmentation detected. Performing cleanup...")

                    # Update timestamp before cleanup to prevent other threads from starting cleanup
                    self.last_cleanup_time = current_time
                    self.second_until_gc -= time_since_last_cleanup
                    if self.second_until_gc <= 0:
                        should_gc = True
                        self.second_until_gc = 20 * 60 # 20 minutes until next GC

            if should_cleanup:
                if should_gc:
                    gc.collect()
                torch.cuda.empty_cache()

                free, total = torch.cuda.mem_get_info(device)

                if self.is_memory_fragmented():
                    print("Memory fragmentation after cleanup detected.")
                    self.defragment_memory(free)

                percent = 100 - (free / total) * 100
                print(f"GPU mem after cleanup: {percent:.2f}%, {(total - free) / (1024 ** 3):.2f}GB / {total / (1024 ** 3):.2f}GB")
            else:
                print(f"GPU mem: {percent:.2f}% <= ({threshold}%), {(total - free) / (1024 ** 3):.2f}GB / {total / (1024 ** 3):.2f}GB")

            return percent, (total - free) / (1024 ** 3), total / (1024 ** 3)
        except Exception as e:
            print(f"Error in GPU memory check: {e}")
            return None, None, None

    async def async_check_and_cleanup(self):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.check_and_cleanup()
        )
