# echogrid/utils/device_utils.py
"""
Device and hardware utilities for EchoGrid.

This module provides utilities for detecting and managing hardware resources.
"""

import logging
import platform
import psutil
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_available_devices() -> List[str]:
    """
    Get list of available compute devices.
    
    Returns:
        List[str]: Available devices (e.g., ["cpu", "cuda:0", "mps"])
    """
    devices = ["cpu"]
    
    try:
        # Check for CUDA
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                devices.append(f"cuda:{i}")
            devices.append("cuda")  # Generic CUDA
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append("mps")
            
    except ImportError:
        logger.debug("PyTorch not available, only CPU device detected")
    
    return devices


def get_device_info(device: str = "auto") -> Dict[str, Any]:
    """
    Get detailed information about a device.
    
    Args:
        device: Device string (e.g., "cpu", "cuda:0", "mps")
    
    Returns:
        Dict[str, Any]: Device information
    """
    if device == "auto":
        devices = get_available_devices()
        device = devices[1] if len(devices) > 1 else "cpu"  # Prefer GPU if available
    
    info = {
        "device": device,
        "available": device in get_available_devices(),
        "type": _get_device_type(device)
    }
    
    if device == "cpu":
        info.update(_get_cpu_info())
    elif device.startswith("cuda"):
        info.update(_get_cuda_info(device))
    elif device == "mps":
        info.update(_get_mps_info())
    
    return info


def get_memory_info(device: str = "auto") -> Dict[str, Any]:
    """
    Get memory information for a device.
    
    Args:
        device: Device string
    
    Returns:
        Dict[str, Any]: Memory information in GB
    """
    if device == "auto":
        devices = get_available_devices()
        device = devices[1] if len(devices) > 1 else "cpu"
    
    if device == "cpu":
        return _get_system_memory_info()
    elif device.startswith("cuda"):
        return _get_cuda_memory_info(device)
    elif device == "mps":
        return _get_mps_memory_info()
    else:
        return {"total_gb": 0, "available_gb": 0, "used_gb": 0}


def get_optimal_device(
    memory_requirement_gb: Optional[float] = None,
    prefer_gpu: bool = True
) -> str:
    """
    Get the optimal device for a given task.
    
    Args:
        memory_requirement_gb: Required memory in GB
        prefer_gpu: Whether to prefer GPU over CPU
    
    Returns:
        str: Optimal device string
    """
    devices = get_available_devices()
    
    # If only CPU available, return it
    if len(devices) == 1:
        return "cpu"
    
    # Check memory requirements
    if memory_requirement_gb:
        for device in devices:
            if device == "cpu":
                continue  # Check GPU devices first if preferring GPU
            
            memory_info = get_memory_info(device)
            if memory_info["available_gb"] >= memory_requirement_gb:
                return device
        
        # Fall back to CPU if GPUs don't have enough memory
        cpu_memory = get_memory_info("cpu")
        if cpu_memory["available_gb"] >= memory_requirement_gb:
            return "cpu"
        
        logger.warning(f"No device has sufficient memory ({memory_requirement_gb:.1f}GB required)")
    
    # Default selection based on preference
    if prefer_gpu:
        # Return first GPU device
        for device in devices:
            if device != "cpu":
                return device
    
    return "cpu"


def check_device_compatibility(device: str) -> Dict[str, Any]:
    """
    Check if a device is compatible and get capability info.
    
    Args:
        device: Device string to check
    
    Returns:
        Dict[str, Any]: Compatibility information
    """
    result = {
        "compatible": False,
        "available": False,
        "issues": [],
        "capabilities": {}
    }
    
    available_devices = get_available_devices()
    result["available"] = device in available_devices
    
    if not result["available"]:
        result["issues"].append(f"Device {device} not available")
        return result
    
    try:
        if device == "cpu":
            result.update(_check_cpu_compatibility())
        elif device.startswith("cuda"):
            result.update(_check_cuda_compatibility(device))
        elif device == "mps":
            result.update(_check_mps_compatibility())
        
        result["compatible"] = len(result["issues"]) == 0
        
    except Exception as e:
        result["issues"].append(f"Error checking compatibility: {e}")
    
    return result


def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information.
    
    Returns:
        Dict[str, Any]: System information
    """
    try:
        info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "available_devices": get_available_devices()
        }
        
        # Add GPU information if available
        try:
            import torch
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
                info["pytorch_version"] = torch.__version__
                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_names"] = [
                    torch.cuda.get_device_name(i) 
                    for i in range(torch.cuda.device_count())
                ]
        except ImportError:
            pass
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return {"error": str(e)}


def _get_device_type(device: str) -> str:
    """Get device type category."""
    if device == "cpu":
        return "cpu"
    elif device.startswith("cuda"):
        return "gpu"
    elif device == "mps":
        return "gpu"
    else:
        return "unknown"


def _get_cpu_info() -> Dict[str, Any]:
    """Get CPU-specific information."""
    try:
        return {
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
            "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "usage_percent": psutil.cpu_percent(interval=1),
            "architecture": platform.machine(),
            "supports_avx": _check_cpu_features()
        }
    except Exception as e:
        logger.warning(f"Failed to get CPU info: {e}")
        return {}


def _get_cuda_info(device: str) -> Dict[str, Any]:
    """Get CUDA device information."""
    try:
        import torch
        
        if device == "cuda":
            device_id = 0
        else:
            device_id = int(device.split(":")[1])
        
        if device_id >= torch.cuda.device_count():
            return {"error": f"CUDA device {device_id} not available"}
        
        properties = torch.cuda.get_device_properties(device_id)
        
        return {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_gb": properties.total_memory / (1024**3),
            "multiprocessor_count": properties.multi_processor_count,
            "max_threads_per_block": properties.max_threads_per_block,
            "supports_fp16": properties.major >= 7,  # Simplified check
            "supports_tensor_cores": properties.major >= 7,
            "cuda_version": torch.version.cuda
        }
        
    except ImportError:
        return {"error": "PyTorch not available"}
    except Exception as e:
        return {"error": str(e)}


def _get_mps_info() -> Dict[str, Any]:
    """Get MPS (Apple Silicon) device information."""
    try:
        import torch
        
        if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
            return {"error": "MPS not available"}
        
        return {
            "name": "Apple Metal Performance Shaders",
            "architecture": platform.machine(),
            "supports_fp16": True,
            "pytorch_version": torch.__version__
        }
        
    except ImportError:
        return {"error": "PyTorch not available"}
    except Exception as e:
        return {"error": str(e)}


def _get_system_memory_info() -> Dict[str, Any]:
    """Get system memory information."""
    try:
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent_used": memory.percent
        }
    except Exception as e:
        logger.warning(f"Failed to get memory info: {e}")
        return {"total_gb": 0, "available_gb": 0, "used_gb": 0}


def _get_cuda_memory_info(device: str) -> Dict[str, Any]:
    """Get CUDA memory information."""
    try:
        import torch
        
        if device == "cuda":
            device_id = 0
        else:
            device_id = int(device.split(":")[1])
        
        if device_id >= torch.cuda.device_count():
            return {"total_gb": 0, "available_gb": 0, "used_gb": 0}
        
        # Get memory info
        total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
        cached = torch.cuda.memory_reserved(device_id) / (1024**3)
        
        return {
            "total_gb": total,
            "allocated_gb": allocated,
            "cached_gb": cached,
            "available_gb": total - cached,
            "used_gb": allocated
        }
        
    except ImportError:
        return {"total_gb": 0, "available_gb": 0, "used_gb": 0}
    except Exception as e:
        logger.warning(f"Failed to get CUDA memory info: {e}")
        return {"total_gb": 0, "available_gb": 0, "used_gb": 0}


def _get_mps_memory_info() -> Dict[str, Any]:
    """Get MPS memory information."""
    try:
        # MPS shares system memory, so return system memory info
        # but indicate it's shared
        memory_info = _get_system_memory_info()
        memory_info["shared_with_system"] = True
        return memory_info
        
    except Exception as e:
        logger.warning(f"Failed to get MPS memory info: {e}")
        return {"total_gb": 0, "available_gb": 0, "used_gb": 0}


def _check_cpu_compatibility() -> Dict[str, Any]:
    """Check CPU compatibility."""
    result = {
        "compatible": True,
        "issues": [],
        "capabilities": {}
    }
    
    try:
        # Check minimum CPU requirements
        cpu_count = psutil.cpu_count(logical=False)
        if cpu_count < 2:
            result["issues"].append("Minimum 2 CPU cores recommended")
        
        # Check available memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 4:
            result["issues"].append("Minimum 4GB RAM recommended")
        
        # Check CPU features
        result["capabilities"]["avx_support"] = _check_cpu_features()
        
    except Exception as e:
        result["issues"].append(f"CPU compatibility check failed: {e}")
    
    return result


def _check_cuda_compatibility(device: str) -> Dict[str, Any]:
    """Check CUDA compatibility."""
    result = {
        "compatible": True,
        "issues": [],
        "capabilities": {}
    }
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            result["issues"].append("CUDA not available")
            result["compatible"] = False
            return result
        
        device_id = 0 if device == "cuda" else int(device.split(":")[1])
        
        if device_id >= torch.cuda.device_count():
            result["issues"].append(f"CUDA device {device_id} not found")
            result["compatible"] = False
            return result
        
        properties = torch.cuda.get_device_properties(device_id)
        
        # Check compute capability
        if properties.major < 6:
            result["issues"].append(f"Compute capability {properties.major}.{properties.minor} is old (6.0+ recommended)")
        
        # Check memory
        memory_gb = properties.total_memory / (1024**3)
        if memory_gb < 2:
            result["issues"].append(f"Low GPU memory: {memory_gb:.1f}GB (4GB+ recommended)")
        
        result["capabilities"] = {
            "compute_capability": f"{properties.major}.{properties.minor}",
            "memory_gb": memory_gb,
            "supports_fp16": properties.major >= 7,
            "supports_tensor_cores": properties.major >= 7
        }
        
    except ImportError:
        result["issues"].append("PyTorch not available")
        result["compatible"] = False
    except Exception as e:
        result["issues"].append(f"CUDA compatibility check failed: {e}")
        result["compatible"] = False
    
    return result


def _check_mps_compatibility() -> Dict[str, Any]:
    """Check MPS compatibility."""
    result = {
        "compatible": True,
        "issues": [],
        "capabilities": {}
    }
    
    try:
        import torch
        
        if not hasattr(torch.backends, 'mps'):
            result["issues"].append("MPS not supported in this PyTorch version")
            result["compatible"] = False
            return result
        
        if not torch.backends.mps.is_available():
            result["issues"].append("MPS not available on this system")
            result["compatible"] = False
            return result
        
        # Check macOS version (MPS requires macOS 12.3+)
        if platform.system() == "Darwin":
            version = platform.mac_ver()[0]
            major, minor = map(int, version.split(".")[:2])
            if major < 12 or (major == 12 and minor < 3):
                result["issues"].append(f"macOS {version} detected, MPS requires macOS 12.3+")
        
        result["capabilities"] = {
            "supports_fp16": True,
            "unified_memory": True,
            "macos_version": platform.mac_ver()[0] if platform.system() == "Darwin" else None
        }
        
    except ImportError:
        result["issues"].append("PyTorch not available")
        result["compatible"] = False
    except Exception as e:
        result["issues"].append(f"MPS compatibility check failed: {e}")
        result["compatible"] = False
    
    return result


def _check_cpu_features() -> Dict[str, bool]:
    """Check CPU feature support."""
    features = {
        "avx": False,
        "avx2": False,
        "sse4_1": False,
        "sse4_2": False
    }
    
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        flags = info.get("flags", [])
        
        features["avx"] = "avx" in flags
        features["avx2"] = "avx2" in flags
        features["sse4_1"] = "sse4_1" in flags
        features["sse4_2"] = "sse4_2" in flags
        
    except ImportError:
        # Fallback check using platform-specific methods
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo_text = f.read()
                    features["avx"] = "avx" in cpuinfo_text
                    features["avx2"] = "avx2" in cpuinfo_text
            except Exception:
                pass
    except Exception:
        pass
    
    return features


def clear_gpu_cache():
    """Clear GPU memory cache."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to clear GPU cache: {e}")


def set_memory_fraction(device: str, fraction: float):
    """
    Set memory fraction for GPU device.
    
    Args:
        device: GPU device string
        fraction: Memory fraction (0.0-1.0)
    """
    try:
        import torch
        
        if not device.startswith("cuda"):
            logger.warning(f"Memory fraction only applicable to CUDA devices, got {device}")
            return
        
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"Memory fraction must be between 0.0 and 1.0, got {fraction}")
        
        device_id = 0 if device == "cuda" else int(device.split(":")[1])
        
        if device_id < torch.cuda.device_count():
            torch.cuda.set_per_process_memory_fraction(fraction, device_id)
            logger.info(f"Set memory fraction to {fraction} for {device}")
        else:
            logger.warning(f"CUDA device {device_id} not available")
            
    except ImportError:
        logger.warning("PyTorch not available, cannot set memory fraction")
    except Exception as e:
        logger.error(f"Failed to set memory fraction: {e}")


def benchmark_device(device: str, duration: float = 5.0) -> Dict[str, Any]:
    """
    Benchmark device performance.
    
    Args:
        device: Device to benchmark
        duration: Benchmark duration in seconds
    
    Returns:
        Dict[str, Any]: Benchmark results
    """
    try:
        import torch
        import time
        
        results = {
            "device": device,
            "duration": duration,
            "operations_per_second": 0,
            "memory_bandwidth_gb_s": 0,
            "error": None
        }
        
        if device == "cpu":
            results.update(_benchmark_cpu(duration))
        elif device.startswith("cuda"):
            results.update(_benchmark_cuda(device, duration))
        elif device == "mps":
            results.update(_benchmark_mps(duration))
        else:
            results["error"] = f"Benchmarking not supported for device: {device}"
        
        return results
        
    except ImportError:
        return {"device": device, "error": "PyTorch not available"}
    except Exception as e:
        return {"device": device, "error": str(e)}


def _benchmark_cpu(duration: float) -> Dict[str, Any]:
    """Benchmark CPU performance."""
    import torch
    import time
    
    # Matrix multiplication benchmark
    size = 1000
    device = torch.device("cpu")
    
    start_time = time.time()
    operations = 0
    
    while time.time() - start_time < duration:
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.mm(a, b)
        operations += 1
    
    elapsed = time.time() - start_time
    ops_per_second = operations / elapsed
    
    return {
        "operations_per_second": ops_per_second,
        "matrix_size": size,
        "total_operations": operations
    }


def _benchmark_cuda(device: str, duration: float) -> Dict[str, Any]:
    """Benchmark CUDA performance."""
    import torch
    import time
    
    device_obj = torch.device(device)
    
    # Warm up
    a = torch.randn(1000, 1000, device=device_obj)
    b = torch.randn(1000, 1000, device=device_obj)
    torch.mm(a, b)
    torch.cuda.synchronize()
    
    # Benchmark
    size = 2000  # Larger size for GPU
    start_time = time.time()
    operations = 0
    
    while time.time() - start_time < duration:
        a = torch.randn(size, size, device=device_obj)
        b = torch.randn(size, size, device=device_obj)
        c = torch.mm(a, b)
        torch.cuda.synchronize()  # Ensure operation completes
        operations += 1
    
    elapsed = time.time() - start_time
    ops_per_second = operations / elapsed
    
    # Memory bandwidth test
    memory_gb_s = _benchmark_memory_bandwidth(device_obj)
    
    return {
        "operations_per_second": ops_per_second,
        "memory_bandwidth_gb_s": memory_gb_s,
        "matrix_size": size,
        "total_operations": operations
    }


def _benchmark_mps(duration: float) -> Dict[str, Any]:
    """Benchmark MPS performance."""
    import torch
    import time
    
    device = torch.device("mps")
    
    # Matrix multiplication benchmark
    size = 1500
    start_time = time.time()
    operations = 0
    
    while time.time() - start_time < duration:
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.mm(a, b)
        operations += 1
    
    elapsed = time.time() - start_time
    ops_per_second = operations / elapsed
    
    return {
        "operations_per_second": ops_per_second,
        "matrix_size": size,
        "total_operations": operations
    }


def _benchmark_memory_bandwidth(device) -> float:
    """Benchmark memory bandwidth for GPU."""
    try:
        import torch
        import time
        
        # Large tensor for memory bandwidth test
        size = 10000
        iterations = 100
        
        # Create tensors
        a = torch.randn(size, size, device=device)
        b = torch.zeros_like(a)
        
        # Warm up
        for _ in range(10):
            b.copy_(a)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark memory copy
        start_time = time.time()
        for _ in range(iterations):
            b.copy_(a)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        
        # Calculate bandwidth
        bytes_transferred = size * size * 4 * iterations * 2  # float32 * iterations * read+write
        gb_transferred = bytes_transferred / (1024**3)
        bandwidth_gb_s = gb_transferred / elapsed
        
        return bandwidth_gb_s
        
    except Exception:
        return 0.0


def get_recommended_batch_size(
    device: str,
    model_size_gb: float,
    sequence_length: int = 512
) -> int:
    """
    Get recommended batch size for a device and model.
    
    Args:
        device: Target device
        model_size_gb: Model size in GB
        sequence_length: Input sequence length
    
    Returns:
        int: Recommended batch size
    """
    memory_info = get_memory_info(device)
    available_memory_gb = memory_info.get("available_gb", 4.0)
    
    # Reserve some memory for overhead
    usable_memory_gb = available_memory_gb * 0.8
    
    # Estimate memory per sample (rough heuristic)
    if device == "cpu":
        memory_per_sample_gb = model_size_gb * 0.1 + sequence_length * 0.000001
        base_batch_size = 4
    else:  # GPU
        memory_per_sample_gb = model_size_gb * 0.2 + sequence_length * 0.000002
        base_batch_size = 8
    
    # Calculate maximum batch size based on memory
    if memory_per_sample_gb > 0:
        max_batch_size = int(usable_memory_gb / memory_per_sample_gb)
        batch_size = min(max_batch_size, base_batch_size * 4)  # Cap at reasonable size
    else:
        batch_size = base_batch_size
    
    # Ensure minimum batch size
    batch_size = max(1, batch_size)
    
    return batch_size


def monitor_device_usage(device: str, interval: float = 1.0) -> Dict[str, Any]:
    """
    Monitor device usage statistics.
    
    Args:
        device: Device to monitor
        interval: Monitoring interval in seconds
    
    Returns:
        Dict[str, Any]: Usage statistics
    """
    import time
    
    stats = {
        "device": device,
        "timestamp": time.time(),
        "cpu_percent": 0,
        "memory_percent": 0,
        "temperature": None
    }
    
    try:
        if device == "cpu":
            stats["cpu_percent"] = psutil.cpu_percent(interval=interval)
            memory = psutil.virtual_memory()
            stats["memory_percent"] = memory.percent
            
            # Try to get CPU temperature
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get first available temperature sensor
                    for name, entries in temps.items():
                        if entries:
                            stats["temperature"] = entries[0].current
                            break
            except (AttributeError, OSError):
                pass
                
        elif device.startswith("cuda"):
            try:
                import torch
                device_id = 0 if device == "cuda" else int(device.split(":")[1])
                
                if device_id < torch.cuda.device_count():
                    # Memory usage
                    memory_allocated = torch.cuda.memory_allocated(device_id)
                    memory_total = torch.cuda.get_device_properties(device_id).total_memory
                    stats["memory_percent"] = (memory_allocated / memory_total) * 100
                    
                    # Try to get GPU utilization using nvidia-ml-py if available
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        stats["gpu_percent"] = utilization.gpu
                        stats["memory_percent"] = utilization.memory
                        
                        # Temperature
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        stats["temperature"] = temp
                        
                    except ImportError:
                        pass  # nvidia-ml-py not available
                    except Exception:
                        pass  # NVIDIA ML failed
                        
            except ImportError:
                pass
                
    except Exception as e:
        stats["error"] = str(e)
    
    return stats