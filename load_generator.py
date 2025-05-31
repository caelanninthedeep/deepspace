import os
import time
import random
import subprocess
import multiprocessing
import mmap

# 1️⃣ 内存压力：分配随机大小内存块
def generate_memory_load():
    mem_mb = random.randint(100, 900)
    print(f"[Memory Load] Allocating {mem_mb}MB...")
    block = bytearray(mem_mb * 1024 * 1024)
    for i in range(0, len(block), 4096):
        block[i] = 1
    time.sleep(random.randint(3, 7))

# 2️⃣ CPU 压力：使用随机核数运行计算密集型任务
def generate_cpu_load():
    cpu_cores = random.randint(1, os.cpu_count())
    duration = random.randint(5, 15)
    print(f"[CPU Load] Using {cpu_cores} cores for {duration}s...")
    subprocess.run(['stress', '--cpu', str(cpu_cores), '--timeout', str(duration)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 3️⃣ 磁盘 I/O 压力：写入随机大小的数据到临时文件
def generate_io_load():
    io_mb = random.randint(100, 1000)
    print(f"[IO Load] Writing {io_mb}MB to disk...")
    subprocess.run(['dd', 'if=/dev/urandom', 'of=/tmp/testfile', 'bs=1M', f'count={io_mb}'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 4️⃣ 网络 I/O 压力：多次 ping 外部主机
def generate_network_load():
    ping_count = random.randint(10, 100)
    print(f"[Network Load] Pinging google.com {ping_count} times...")
    try:
        subprocess.run(['ping', '-c', str(ping_count), 'google.com'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("[Warning] 'ping' command not found.")

# 5️⃣ 并发内存分配：多个子进程同时分配大内存
def parallel_memory_allocations():
    print("[Parallel Load] Spawning memory-heavy processes...")
    def worker():
        _ = bytearray(random.randint(100, 500) * 1024 * 1024)
        time.sleep(3)
    procs = [multiprocessing.Process(target=worker) for _ in range(3)]
    for p in procs: p.start()
    for p in procs: p.join()

# 6️⃣ 文件缓存压力：反复创建/读写/删除文件
def file_cache_pressure():
    print("[File Cache] Stressing filesystem cache...")
    for _ in range(30):
        with open("/tmp/cache_testfile", "w") as f:
            f.write("a" * 1024 * 1024)
        with open("/tmp/cache_testfile", "r") as f:
            f.read()
        os.remove("/tmp/cache_testfile")
    time.sleep(2)

# 7️⃣ drop_caches 操作
def drop_caches():
    print("[Sys] Dropping filesystem caches...")
    subprocess.run(['sudo', 'sync'])
    subprocess.run(['sudo', 'bash', '-c', 'echo 3 > /proc/sys/vm/drop_caches'])

# 8️⃣ mmap 文件模拟
def mmap_access():
    print("[mmap] Accessing via mmap...")
    with open("/tmp/mmap_test", "wb") as f:
        f.write(b'\x00' * 1024 * 1024)
    with open("/tmp/mmap_test", "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        for _ in range(100000):
            mm[random.randint(0, 1024 * 1024 - 1)] = b'\x01'[0]
        mm.close()
    os.remove("/tmp/mmap_test")
    time.sleep(3)

# 🔀 随机选择压力函数
def generate_random_load():
    load_functions = [
        generate_memory_load,
        generate_cpu_load,
        generate_io_load,
        generate_network_load,
        parallel_memory_allocations,
        file_cache_pressure,
        drop_caches,
        mmap_access,
    ]
    random.choice(load_functions)()

# 主循环
if __name__ == "__main__":
    while True:
        generate_random_load()
        interval = random.randint(5, 10)
        print(f"[Sleep] Waiting {interval}s before next load...\n")
        time.sleep(interval)














