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
    print("CPU information gathered successfully.")

def get_memory_info():
    """Get memory information."""
    print("\n=== Memory Information ===")
    memory = psutil.virtual_memory()
    print(f"Total Memory: {memory.total / (1024 ** 3):.2f} GB")
    print(f"Available Memory: {memory.available / (1024 ** 3):.2f} GB")
    print(f"Used Memory: {memory.used / (1024 ** 3):.2f} GB")
    print(f"Memory Usage: {memory.percent}%")
    print("Memory information gathered successfully.")

def check_model_memory_requirements():
    """Check memory requirements for loading LLaMA 2.7B model."""
    print("\n=== Checking Memory Requirements for LLaMA 2.7B Model ===")
    model_memory_size = 2.7 * 4 / (1024 ** 2)  # in GB
    additional_memory_factor = 2  # Estimate for activations and other overhead
    total_memory_required = model_memory_size * additional_memory_factor
    
    print(f"Estimated memory required to load LLaMA 2.7B model: {total_memory_required:.2f} GB")
    
    memory = psutil.virtual_memory()
    if memory.available < total_memory_required * (1024 ** 2):  # Convert GB to bytes
        print("WARNING: Insufficient memory to load the LLaMA 2.7B model.")
    else:
        print("Sufficient memory available to load the LLaMA 2.7B model.")

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

def check_minimal_requirements():
    """Check minimal requirements for running the system."""
    print("\n=== Checking Minimal Requirements ===")
    
    # Check CPU
    cpu_cores = psutil.cpu_count(logical=True)
    if cpu_cores < 2:
        print("WARNING: Minimum of 2 CPU cores is recommended.")
    
    # Check Memory
    memory = psutil.virtual_memory()
    if memory.total < 4 * (1024 ** 3):  # 4 GB
        print("WARNING: Minimum of 4 GB RAM is required.")
    
    # Check Disk Space
    print("\n=== Checking Disk Space ===")
    partitions = psutil.disk_partitions()
    total_free_space = 0
    for partition in partitions:
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            total_free_space += usage.free
            print(f"Partition {partition.mountpoint}: Free Space: {usage.free / (1024 ** 3):.2f} GB")
        except PermissionError as e:
            print(f"WARNING: Cannot access {partition.mountpoint}. {str(e)}")
        except Exception as e:
            print(f"ERROR: An error occurred while accessing {partition.mountpoint}. {str(e)}")
    
    if total_free_space < 50 * (1024 ** 3):  # 50 GB
        print("WARNING: Minimum of 50 GB free disk space is required.")
    
    # Check Internet Connection
    print("\n=== Checking Internet Connection ===")
    try:
        response = os.system("ping -c 1 google.com")  # For Linux/Mac
        # response = os.system("ping -n 1 google.com")  # For Windows
        if response != 0:
            print("WARNING: Internet connection is not available.")
        else:
            print("Internet connection is available.")
    except Exception as e:
        print(f"ERROR: Could not check internet connection: {str(e)}")

def main():
    """Main function to gather system information."""
    get_os_info()
    get_cpu_info()
    get_memory_info()
    get_swap_info()
    get_disk_info()
    get_gpu_info()
    check_minimal_requirements()  # Check minimal requirements
    check_model_memory_requirements()  # Check memory requirements for LLaMA 2.7B model

if __name__ == "__main__":
    main() 
