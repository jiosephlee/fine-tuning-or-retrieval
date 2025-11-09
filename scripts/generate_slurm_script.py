import argparse
import os
import re
from pathlib import Path

# The SLURM header template based on the user's example
SLURM_TEMPLATE = """#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=ai
#SBATCH --mem-per-gpu=160GB
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}.out
#SBATCH --error=logs/{job_name}.err

echo "➤ START"

echo "➤ SETTING UP HOST CUDA"
module unload cuda
module load cuda/12.4

# Define the path to your SIF file
YOUR_SIF_FILE="/gpfs/fs001/cbica/home/leejose/joseph/pytorch-2.4.0-cuda12.4-cudnn9-devel.sif"

echo "➤ RUNNING SCRIPT INSIDE APPTAINER: ${{YOUR_SIF_FILE}}"
"""

# The apptainer command template
APPTAINER_TEMPLATE = """
# Execute the python script INSIDE the container
# --nv: Mounts the host NVIDIA drivers
# --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES: Passes the GPU assignment from SLURM into the container
apptainer exec --cleanenv --nv \\
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \\
    ${{YOUR_SIF_FILE}} \\
    {command}
"""

def format_python_command(command_string):
    """
    Formats a python command string to have each argument on a new line.
    """
    # Split the command by ' --' to separate arguments
    parts = command_string.split(' --')
    if len(parts) <= 1:
        return command_string  # No arguments to format, return as is

    base_command = parts[0]
    args = parts[1:]

    # Each argument is prepended with '--' and a backslash for line continuation
    formatted_args = [f"    --{arg.strip()} \\" for arg in args]
    
    # Remove the trailing backslash from the very last argument
    if formatted_args:
        formatted_args[-1] = formatted_args[-1][:-2]

    # Combine the base command and the formatted arguments
    return f"{base_command} \\\n" + "\n".join(formatted_args)


def parse_bash_script(script_content):
    """
    Parses a simple bash script to extract python commands and their associated variables.
    """
    # First pass: handle line continuations
    processed_content = script_content.replace('\\\n', ' ')

    variables = {}
    python_commands = []

    # Second pass: extract variables and commands
    for line in processed_content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        var_match = re.match(r'^(\w+)=(.+)', line)
        if var_match:
            key, value = var_match.groups()
            variables[key.strip()] = value.strip()
            continue
        
        if line.startswith('python '):
            command_to_process = line
            # Substitute variables in the current command
            for var, val in variables.items():
                command_to_process = command_to_process.replace(f'${var}', val)
                command_to_process = command_to_process.replace(f'${{{var}}}', val)

            # Remove output redirection, as SLURM handles it
            command_without_redirect = re.split(r'\s+>\s+', command_to_process)[0]
            
            python_commands.append(command_without_redirect.strip())

    return python_commands

def main():
    parser = argparse.ArgumentParser(
        description="Convert a bash script into a SLURM apptainer script."
    )
    parser.add_argument(
        "--input_script",
        type=str,
        required=True,
        help="Path to the input bash script (e.g., E44_7B_200_epochs.sh)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./FT/",
        help="Directory to save the generated SLURM script."
    )
    args = parser.parse_args()

    input_path = Path(args.input_script)
    if not input_path.exists():
        print(f"Error: Input file not found at {args.input_script}")
        return

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output script path
    output_filename = input_path.stem + "_slurm.sh"
    output_path = output_dir / output_filename

    with open(input_path, 'r') as f:
        script_content = f.read()

    python_commands = parse_bash_script(script_content)

    if not python_commands:
        print(f"Warning: No python commands found in {args.input_script}")
        return

    # Generate SLURM header
    job_name = input_path.stem
    
    slurm_header = SLURM_TEMPLATE.format(
        job_name=job_name,
    )

    # Format python commands for readability and then generate apptainer commands
    formatted_commands = [format_python_command(cmd) for cmd in python_commands]
    apptainer_commands = [APPTAINER_TEMPLATE.format(command=cmd) for cmd in formatted_commands]

    final_script = slurm_header + "".join(apptainer_commands) + "\necho \"➤ DONE\"\n"

    with open(output_path, 'w') as f:
        f.write(final_script)
        
    # Make the script executable
    os.chmod(output_path, 0o755)

    print(f"✓ Successfully generated SLURM script at: {output_path}")
    print(f"To submit the job, run: sbatch {output_path}")

if __name__ == "__main__":
    main()
