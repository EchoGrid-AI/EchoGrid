# echogrid/utils/logging.py
"""
Comprehensive logging system for EchoGrid.

This module provides structured logging with performance tracking, context management,
and multiple output formats designed for both development and production use.
"""

import os
import sys
import json
import logging
import threading
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union, TextIO
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from contextlib import contextmanager


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Outputs log records as JSON objects for easy parsing by log aggregation systems.
    """
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add process and thread info
        log_obj["process_id"] = os.getpid()
        log_obj["thread_id"] = threading.get_ident()
        log_obj["thread_name"] = threading.current_thread().name
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if enabled
        if self.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in {
                    "name", "msg", "args", "levelname", "levelno", "pathname",
                    "filename", "module", "exc_info", "exc_text", "stack_info",
                    "lineno", "funcName", "created", "msecs", "relativeCreated",
                    "thread", "threadName", "processName", "process", "getMessage"
                }:
                    try:
                        # Ensure JSON serializable
                        json.dumps(value)
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        extra_fields[key] = str(value)
            
            if extra_fields:
                log_obj["extra"] = extra_fields
        
        return json.dumps(log_obj, ensure_ascii=False)


class ContextFormatter(logging.Formatter):
    """
    Formatter that includes contextual information like model names, operations, etc.
    """
    
    def __init__(self, include_context: bool = True):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with contextual information."""
        # Add context prefix if available
        context_parts = []
        
        if hasattr(record, 'model_name'):
            context_parts.append(f"model={record.model_name}")
        if hasattr(record, 'operation'):
            context_parts.append(f"op={record.operation}")
        if hasattr(record, 'user_id'):
            context_parts.append(f"user={record.user_id}")
        if hasattr(record, 'request_id'):
            context_parts.append(f"req={record.request_id}")
        
        if context_parts and self.include_context:
            context_str = f"[{','.join(context_parts)}] "
            record.msg = f"{context_str}{record.msg}"
        
        return super().format(record)


class PerformanceFilter(logging.Filter):
    """
    Filter that tracks performance metrics and adds timing information.
    """
    
    def __init__(self):
        super().__init__()
        self._operation_times = {}
        self._lock = threading.Lock()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance information to log records."""
        with self._lock:
            # Track operation timing
            if hasattr(record, 'operation') and hasattr(record, 'duration'):
                op_name = record.operation
                duration = record.duration
                
                if op_name not in self._operation_times:
                    self._operation_times[op_name] = []
                
                self._operation_times[op_name].append(duration)
                
                # Add performance stats
                times = self._operation_times[op_name]
                record.avg_duration = sum(times) / len(times)
                record.min_duration = min(times)
                record.max_duration = max(times)
                record.operation_count = len(times)
        
        return True
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all operations."""
        with self._lock:
            stats = {}
            for op_name, times in self._operation_times.items():
                if times:
                    stats[op_name] = {
                        "count": len(times),
                        "total_time": sum(times),
                        "avg_time": sum(times) / len(times),
                        "min_time": min(times),
                        "max_time": max(times),
                        "last_time": times[-1] if times else 0
                    }
            return stats


class EchoGridLogger:
    """
    Enhanced logger for EchoGrid with context management and performance tracking.
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._context_stack = []
        self._local = threading.local()
    
    def _get_context(self) -> Dict[str, Any]:
        """Get current context dictionary."""
        if not hasattr(self._local, 'context'):
            self._local.context = {}
        return self._local.context
    
    def debug(self, msg: str, **kwargs):
        """Log debug message with context."""
        self._log(logging.DEBUG, msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        """Log info message with context."""
        self._log(logging.INFO, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message with context."""
        self._log(logging.WARNING, msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        """Log error message with context."""
        self._log(logging.ERROR, msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        """Log critical message with context."""
        self._log(logging.CRITICAL, msg, **kwargs)
    
    def _log(self, level: int, msg: str, **kwargs):
        """Internal logging method with context injection."""
        # Merge context
        context = self._get_context()
        extra = {**context, **kwargs}
        
        # Log with extra context
        self.logger.log(level, msg, extra=extra)
    
    @contextmanager
    def context(self, **kwargs):
        """Context manager for adding temporary context."""
        context = self._get_context()
        old_values = {}
        
        # Save old values and set new ones
        for key, value in kwargs.items():
            if key in context:
                old_values[key] = context[key]
            context[key] = value
        
        try:
            yield
        finally:
            # Restore old values
            for key in kwargs:
                if key in old_values:
                    context[key] = old_values[key]
                else:
                    context.pop(key, None)
    
    def set_context(self, **kwargs):
        """Set persistent context for this thread."""
        context = self._get_context()
        context.update(kwargs)
    
    def clear_context(self):
        """Clear all context for this thread."""
        self._local.context = {}
    
    def get_context(self) -> Dict[str, Any]:
        """Get current context."""
        return self._get_context().copy()


class LogManager:
    """
    Central logging manager for EchoGrid.
    
    Manages loggers, handlers, and configuration across the application.
    """
    
    def __init__(self):
        self._loggers = {}
        self._handlers = {}
        self._performance_filter = PerformanceFilter()
        self._configured = False
    
    def get_logger(self, name: str) -> EchoGridLogger:
        """Get or create a logger with the given name."""
        if name not in self._loggers:
            self._loggers[name] = EchoGridLogger(name)
        return self._loggers[name]
    
    def configure(
        self,
        level: str = "INFO",
        format_type: str = "context",  # "context", "json", "simple"
        log_to_file: bool = False,
        log_file: Optional[str] = None,
        log_dir: Optional[str] = None,
        max_file_size: str = "10MB",
        backup_count: int = 5,
        rotation_type: str = "size",  # "size", "time"
        time_rotation: str = "midnight",
        console_output: bool = True,
        enable_performance: bool = True,
        capture_warnings: bool = True
    ):
        """
        Configure logging system.
        
        Args:
            level: Logging level
            format_type: Formatter type
            log_to_file: Enable file logging
            log_file: Specific log file path
            log_dir: Directory for log files
            max_file_size: Maximum size before rotation
            backup_count: Number of backup files to keep
            rotation_type: Type of rotation ("size" or "time")
            time_rotation: Time-based rotation schedule
            console_output: Enable console output
            enable_performance: Enable performance tracking
            capture_warnings: Capture Python warnings
        """
        if self._configured:
            return
        
        # Set root logger level
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create formatter
        formatter = self._create_formatter(format_type)
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            console_handler.setFormatter(formatter)
            
            if enable_performance:
                console_handler.addFilter(self._performance_filter)
            
            root_logger.addHandler(console_handler)
            self._handlers['console'] = console_handler
        
        # File handler
        if log_to_file:
            file_handler = self._create_file_handler(
                log_file, log_dir, max_file_size, backup_count, 
                rotation_type, time_rotation
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            
            if enable_performance:
                file_handler.addFilter(self._performance_filter)
            
            root_logger.addHandler(file_handler)
            self._handlers['file'] = file_handler
        
        # Capture warnings
        if capture_warnings:
            logging.captureWarnings(True)
            warnings_logger = logging.getLogger('py.warnings')
            warnings_logger.setLevel(logging.WARNING)
        
        self._configured = True
    
    def _create_formatter(self, format_type: str) -> logging.Formatter:
        """Create formatter based on type."""
        if format_type == "json":
            return JSONFormatter()
        elif format_type == "context":
            return ContextFormatter()
        else:  # simple
            return logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def _create_file_handler(
        self, log_file: Optional[str], log_dir: Optional[str],
        max_file_size: str, backup_count: int,
        rotation_type: str, time_rotation: str
    ) -> logging.Handler:
        """Create appropriate file handler."""
        # Determine log file path
        if log_file:
            log_path = Path(log_file)
        else:
            if log_dir:
                log_path = Path(log_dir) / "echogrid.log"
            else:
                log_path = Path.home() / ".echogrid" / "logs" / "echogrid.log"
        
        # Create directory
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create handler based on rotation type
        if rotation_type == "time":
            handler = TimedRotatingFileHandler(
                str(log_path),
                when=time_rotation,
                backupCount=backup_count,
                encoding='utf-8'
            )
        else:  # size-based rotation
            # Parse size string
            size_bytes = self._parse_size(max_file_size)
            handler = RotatingFileHandler(
                str(log_path),
                maxBytes=size_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
        
        return handler
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '10MB' to bytes."""
        size_str = size_str.upper().strip()
        
        multipliers = {
            'B': 1,
            'KB': 1024,
            'MB': 1024 ** 2,
            'GB': 1024 ** 3
        }
        
        for unit, multiplier in multipliers.items():
            if size_str.endswith(unit):
                try:
                    value = float(size_str[:-len(unit)])
                    return int(value * multiplier)
                except ValueError:
                    break
        
        # Default to 10MB if parsing fails
        return 10 * 1024 * 1024
    
    def add_handler(self, name: str, handler: logging.Handler):
        """Add a custom handler."""
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        self._handlers[name] = handler
    
    def remove_handler(self, name: str):
        """Remove a handler by name."""
        if name in self._handlers:
            root_logger = logging.getLogger()
            root_logger.removeHandler(self._handlers[name])
            del self._handlers[name]
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics."""
        return self._performance_filter.get_performance_stats()
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self._performance_filter._operation_times.clear()
    
    def set_level(self, level: str):
        """Set logging level for all handlers."""
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        
        for handler in self._handlers.values():
            handler.setLevel(numeric_level)


# Global log manager instance
_log_manager = LogManager()


def configure_logging(**kwargs):
    """Configure the global logging system."""
    _log_manager.configure(**kwargs)


def get_logger(name: str) -> EchoGridLogger:
    """Get a logger instance."""
    return _log_manager.get_logger(name)


def set_log_level(level: str):
    """Set global logging level."""
    _log_manager.set_level(level)


def get_performance_stats() -> Dict[str, Dict[str, float]]:
    """Get performance statistics from logging."""
    return _log_manager.get_performance_stats()


def reset_performance_stats():
    """Reset performance statistics."""
    _log_manager.reset_performance_stats()


@contextmanager
def log_operation(operation: str, logger: Optional[EchoGridLogger] = None, level: str = "INFO"):
    """
    Context manager for logging operations with timing.
    
    Args:
        operation: Name of the operation
        logger: Logger to use (default: get logger for calling module)
        level: Log level for the operation
    
    Example:
        >>> with log_operation("model_loading", logger):
        ...     model = load_model("alice/voice")
        # Logs: "Starting model_loading" and "Completed model_loading in 2.34s"
    """
    if logger is None:
        import inspect
        frame = inspect.currentframe().f_back
        module_name = frame.f_globals.get('__name__', 'unknown')
        logger = get_logger(module_name)
    
    start_time = datetime.now()
    
    # Log start
    logger._log(getattr(logging, level.upper()), f"Starting {operation}", operation=operation)
    
    try:
        yield
        # Log success
        duration = (datetime.now() - start_time).total_seconds()
        logger._log(
            getattr(logging, level.upper()),
            f"Completed {operation} in {duration:.2f}s",
            operation=operation,
            duration=duration,
            status="success"
        )
    except Exception as e:
        # Log failure
        duration = (datetime.now() - start_time).total_seconds()
        logger._log(
            logging.ERROR,
            f"Failed {operation} after {duration:.2f}s: {e}",
            operation=operation,
            duration=duration,
            status="error",
            error_type=type(e).__name__,
            error_message=str(e)
        )
        raise


def log_function_call(logger: Optional[EchoGridLogger] = None, level: str = "DEBUG"):
    """
    Decorator for logging function calls with arguments and timing.
    
    Args:
        logger: Logger to use
        level: Log level
    
    Example:
        >>> @log_function_call()
        ... def my_function(arg1, arg2=None):
        ...     return "result"
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if logger is None:
                func_logger = get_logger(func.__module__)
            else:
                func_logger = logger
            
            # Log function call
            func_logger._log(
                getattr(logging, level.upper()),
                f"Calling {func.__name__}",
                function=func.__name__,
                args=str(args),
                kwargs=str(kwargs)
            )
            
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                
                func_logger._log(
                    getattr(logging, level.upper()),
                    f"Completed {func.__name__} in {duration:.3f}s",
                    function=func.__name__,
                    duration=duration,
                    status="success"
                )
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                func_logger._log(
                    logging.ERROR,
                    f"Error in {func.__name__} after {duration:.3f}s: {e}",
                    function=func.__name__,
                    duration=duration,
                    status="error",
                    error_type=type(e).__name__
                )
                raise
        
        return wrapper
    return decorator