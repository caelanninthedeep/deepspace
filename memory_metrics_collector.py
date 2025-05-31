import psutil
import time
import csv
from datetime import datetime

OUTPUT_CSV = "memory_metrics.csv"
INTERVAL = 5  # 采样间隔（秒）

# 初始化 CSV 文件
def init_csv():
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "mem_total", "mem_free", "mem_available",
            "swap_total", "swap_free", "pgfault", "pgmajfault",
            "slab_unreclaim", "cpu_usage",
            "load_avg_1m", "load_avg_5m", "load_avg_15m",
            "context_switches", "disk_read", "disk_write",
            "net_recv_rate", "net_sent_rate"  # 网络吞吐速率（bytes/sec）
        ])

# 获取单次系统状态
def get_system_metrics(prev_net, prev_time):
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    cpu = psutil.cpu_percent(interval=1)
    load_avg = psutil.getloadavg()
    ctx_switches = psutil.cpu_stats().ctx_switches
    disk_io = psutil.disk_io_counters()
    net_io = psutil.net_io_counters()
    now = time.time()

    interval = now - prev_time
    net_recv_rate = (net_io.bytes_recv - prev_net.bytes_recv) / interval
    net_sent_rate = (net_io.bytes_sent - prev_net.bytes_sent) / interval

    # 从 /proc/vmstat 中读取页错误和 slab 信息
    pgfault = pgmajfault = slab_unreclaim = 0
    with open("/proc/vmstat") as f:
        for line in f:
            if line.startswith("pgfault"):
                pgfault = int(line.split()[1])
            elif line.startswith("pgmajfault"):
                pgmajfault = int(line.split()[1])
            elif line.startswith("slab_unreclaimable"):
                slab_unreclaim = int(line.split()[1])

    metrics = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        mem.total, mem.free, mem.available,
        swap.total, swap.free,
        pgfault, pgmajfault,
        slab_unreclaim, cpu,
        load_avg[0], load_avg[1], load_avg[2],
        ctx_switches,
        disk_io.read_bytes, disk_io.write_bytes,
        int(net_recv_rate), int(net_sent_rate)
    ]

    return metrics, net_io, now

# 持续采集并写入数据
def collect_loop():
    print(f"[start] Collecting memory metrics every {INTERVAL}s...")
    prev_net_io = psutil.net_io_counters()
    prev_time = time.time()

    while True:
        row, prev_net_io, prev_time = get_system_metrics(prev_net_io, prev_time)
        with open(OUTPUT_CSV, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print(f"[log] Snapshot taken at {row[0]}")
        time.sleep(INTERVAL)

if __name__ == "__main__":
    init_csv()
    collect_loop()


