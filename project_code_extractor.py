#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Code Extractor
------------------------
This interactive script traverses a given project directory (or a specified subfolder),
supports extracting code files with specified extensions (e.g. .py, .sh, .ipynb),
and allows you to choose in each folder whether to extract all files or only specific ones.
The extracted content is organized and written to an output file for easy review.

Usage:
    python project_code_extractor.py

After running, you will be prompted to enter:
    - The project base directory.
    - Whether to traverse the entire directory tree or specify a subfolder.
    - The file extensions to consider (default: .py, .sh, .ipynb).
    - For each folder, whether to extract all matching files or choose specific ones (file indices are space separated).
    - Whether to traverse each subdirectory.

The final output is saved to "project_code_extracted.txt" by default.
"""

import os
import sys

def get_input(prompt, default=None):
    """
    Get user input; if the user just presses Enter, return the default value.
    """
    if default is not None:
        prompt = f"{prompt} (default: {default}): "
    else:
        prompt = f"{prompt}: "
    result = input(prompt).strip()
    return result if result else default

def choose_file_extensions():
    """
    Ask the user for the file extensions to extract, and return a set of extensions (all in lowercase).
    """
    default_exts = ".py .sh .ipynb"
    exts_input = get_input("Enter file extensions to extract (separated by space)", default_exts)
    ext_list = []
    for ext in exts_input.split():
        ext = ext.strip().lower()
        if not ext.startswith("."):
            ext = "." + ext
        ext_list.append(ext)
    return set(ext_list)

def interactive_extract(folder, output_lines, extensions):
    """
    Recursively traverse a directory and interactively select code file contents to extract.
    
    Parameters:
        folder (str): Current folder path.
        output_lines (list[str]): List to store the final output content.
        extensions (set[str]): Set of file extensions to extract.
    """
    print("\n" + "=" * 80)
    print(f"Current folder: {folder}")
    print("=" * 80)
    
    try:
        items = os.listdir(folder)
    except Exception as e:
        print(f"Error listing folder '{folder}': {e}")
        return

    # Get all matching files in the current folder.
    matching_files = [f for f in items if os.path.isfile(os.path.join(folder, f)) 
                      and os.path.splitext(f)[1].lower() in extensions]
    
    if matching_files:
        print("Found the following matching files:")
        for idx, file in enumerate(matching_files):
            print(f"  [{idx}] {file}")
        choice = get_input("Extract ALL files or CHOOSE specific ones? (enter 'all' or 'choose')", "all").lower()
        selected_files = []
        if choice == "all":
            selected_files = matching_files
        elif choice == "choose":
            indices_str = get_input("Enter indices of files to extract (space separated)", "")
            try:
                indices = [int(x.strip()) for x in indices_str.split() if x.strip().isdigit()]
                for i in indices:
                    if 0 <= i < len(matching_files):
                        selected_files.append(matching_files[i])
            except Exception as e:
                print(f"Invalid input, skipping selection in this folder: {e}")
        else:
            print("Invalid choice, skipping extraction for this folder.")

        # Append the selected file contents to output_lines.
        for file in selected_files:
            file_path = os.path.join(folder, file)
            output_lines.append("=" * 80)
            output_lines.append(f"File: {file_path}")
            output_lines.append("=" * 80)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                output_lines.append(content)
            except Exception as e:
                output_lines.append(f"Error reading file: {e}")
            output_lines.append("\n\n")
    else:
        print("No matching files found in this folder.")

    # Process subdirectories
    subdirs = [d for d in items if os.path.isdir(os.path.join(folder, d))]
    for subdir in sorted(subdirs):
        subfolder_path = os.path.join(folder, subdir)
        traverse = get_input(f"Do you want to traverse subdirectory '{subfolder_path}'? (y/n)", "y").lower()
        if traverse == "y":
            interactive_extract(subfolder_path, output_lines, extensions)
        else:
            print(f"Skipping subdirectory: {subfolder_path}")

def main():
    print("Project Code Extractor")
    print("----------------------")
    
    base_dir = get_input("Enter the project base directory (absolute path)", "")
    if not base_dir or not os.path.isdir(base_dir):
        print("Invalid directory. Exiting.")
        sys.exit(1)
    
    # Decide whether to traverse the entire directory tree or a specific subfolder.
    traverse_mode = get_input("Do you want to traverse the entire directory tree or a specific subfolder? (enter 'all' or 'sub')", "all").lower()
    if traverse_mode == "sub":
        subfolder = get_input("Enter the subfolder path relative to the base directory", "")
        base_dir = os.path.join(base_dir, subfolder)
        if not os.path.isdir(base_dir):
            print("Invalid subfolder. Exiting.")
            sys.exit(1)
    
    extensions = choose_file_extensions()
    print(f"Using file extensions: {extensions}")
    
    # Output file path is fixed to "project_code_extracted.txt" by default.
    output_file = "project_code_extracted.txt"
    
    output_lines = []
    output_lines.append("Extracted Project Code Content")
    output_lines.append("=" * 80 + "\n")
    output_lines.append(f"Base directory: {base_dir}\n")
    
    # Start interactive traversal.
    interactive_extract(base_dir, output_lines, extensions)
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        print(f"\nExtraction complete. Output saved to: {output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    main()
