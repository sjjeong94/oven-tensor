"""
Command-line interface for oven-tensor
"""

import argparse
import sys
from pathlib import Path
from oven_tensor import (
    clear_kernel_cache,
    reload_kernels,
    list_available_functions,
    get_kernel_manager,
)


def cache_command():
    """Main cache management command"""
    parser = argparse.ArgumentParser(
        prog="oven-tensor-cache", description="Manage oven-tensor kernel cache"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Clear cache command
    clear_parser = subparsers.add_parser("clear", help="Clear kernel cache")

    # List functions command
    list_parser = subparsers.add_parser("list", help="List available functions")

    # Reload command
    reload_parser = subparsers.add_parser("reload", help="Reload all kernels")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show cache information")

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    if args.command == "clear":
        print("Clearing kernel cache...")
        clear_kernel_cache()
        print("Cache cleared successfully!")

    elif args.command == "list":
        print("Available GPU functions:")
        functions = list_available_functions()
        if functions:
            for func in sorted(functions):
                print(f"  - {func}")
        else:
            print("  No functions available. Try compiling some kernels first.")

    elif args.command == "reload":
        print("Reloading kernels...")
        reload_kernels()
        print("Kernels reloaded successfully!")

    elif args.command == "info":
        kernel_manager = get_kernel_manager()
        cache_dir = kernel_manager.cache.cache_dir
        print(f"Cache directory: {cache_dir}")

        # Count cached files
        cached_files = list(cache_dir.glob("*.ptx"))
        print(f"Cached PTX files: {len(cached_files)}")

        if cached_files:
            print("Cached kernels:")
            for ptx_file in sorted(cached_files):
                name = ptx_file.stem
                size = ptx_file.stat().st_size
                print(f"  - {name} ({size} bytes)")

        functions = list_available_functions()
        print(f"Available functions: {len(functions)}")


if __name__ == "__main__":
    cache_command()
