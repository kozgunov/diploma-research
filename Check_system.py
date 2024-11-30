import psutil
import platform
import os
import GPUtil

def get_cpu_info():
    """Get CPU information."""
    print("\n=== CPU Information ===")
    print(f"Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"Total cores: {psutil.cpu_count(logical=True)}")
    print(f"Max Frequency: {psutil.cpu_freq().max}Mhz")
    print(f"Min Frequency: {psutil.cpu_freq().min}Mhz")
    print(f"Current Frequency: {psutil.cpu_freq().current}Mhz")
    print(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")

def get_memory_info():
    """Get memory information."""
    print("\n=== Memory Information ===")
    memory = psutil.virtual_memory()
    print(f"Total Memory: {memory.total / (1024 ** 3):.2f} GB")
    print(f"Available Memory: {memory.available / (1024 ** 3):.2f} GB")
    print(f"Used Memory: {memory.used / (1024 ** 3):.2f} GB")
    print(f"Memory Usage: {memory.percent}%")

def get_swap_info():
    """Get swap memory information."""
    print("\n=== Swap Memory Information ===")
    swap = psutil.swap_memory()
    print(f"Total Swap: {swap.total / (1024 ** 3):.2f} GB")
    print(f"Used Swap: {swap.used / (1024 ** 3):.2f} GB")
    print(f"Free Swap: {swap.free / (1024 ** 3):.2f} GB")
    print(f"Swap Usage: {swap.percent}%")

def get_disk_info():
    """Get disk information."""
    print("\n=== Disk Information ===")
    partitions = psutil.disk_partitions()
    for partition in partitions:
        try:
            print(f"Device: {partition.device}")
            print(f"Mountpoint: {partition.mountpoint}")
            print(f"File system type: {partition.fstype}")
            usage = psutil.disk_usage(partition.mountpoint)
            print(f"Total Size: {usage.total / (1024 ** 3):.2f} GB")
            print(f"Used: {usage.used / (1024 ** 3):.2f} GB")
            print(f"Free: {usage.free / (1024 ** 3):.2f} GB")
            print(f"Percentage Used: {usage.percent}%\n")
        except PermissionError as e:
            print(f"WARNING: Cannot access {partition.mountpoint}. {str(e)}")
        except Exception as e:
            print(f"ERROR: An error occurred while accessing {partition.mountpoint}. {str(e)}")

def get_gpu_info():
    """Get GPU information if available."""
    print("\n=== GPU Information ===")
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU ID: {gpu.id}")
            print(f"GPU Name: {gpu.name}")
            print(f"GPU Load: {gpu.load * 100}%")
            print(f"GPU Free Memory: {gpu.memoryFree}MB")
            print(f"GPU Used Memory: {gpu.memoryUsed}MB")
            print(f"GPU Total Memory: {gpu.memoryTotal}MB")
            print(f"GPU Temperature: {gpu.temperature} °C\n")
    except ImportError:
        print("GPUtil is not installed. Install it using 'pip install gputil' to get GPU information.")

def get_os_info():
    """Get operating system information."""
    print("\n=== Operating System Information ===")
    print(f"OS: {platform.system()}")
    print(f"OS Version: {platform.version()}")
    print(f"OS Release: {platform.release()}")
    print(f"Architecture: {platform.architecture()[0]}")
    print(f"Processor: {platform.processor()}")

def main():
    """Main function to gather system information."""
    get_os_info()
    get_cpu_info()
    get_memory_info()
    get_swap_info()
    get_disk_info()
    get_gpu_info()

if __name__ == "__main__":
    main() 
