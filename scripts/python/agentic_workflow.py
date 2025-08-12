#!/usr/bin/env python3
import subprocess
import json
import time
import os

class AgenticWorkflow:
    def __init__(self, command, output_file, timeout_seconds=60):
        self.command = command
        self.output_file = output_file
        self.timeout_seconds = timeout_seconds

    def run(self):
        """
        Executes the workflow in the background and monitors for the output file.
        """
        print(f"Executing workflow: {' '.join(self.command)}")
        
        # Clean up previous run
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

        try:
            process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            print(f"Waiting for {self.output_file} to be created...")
            start_time = time.time()
            while not os.path.exists(self.output_file):
                if time.time() - start_time > self.timeout_seconds:
                    print("Timeout reached. Terminating process.")
                    process.terminate()
                    process.wait()
                    print("Process terminated.")
                    return None

                print("Workflow running in the background...")
                time.sleep(5)
                
            print(f"{self.output_file} found. Workflow complete.")
            
            # Read and return the last line of the JSONL file
            last_line = None
            with open(self.output_file, 'r') as f:
                for line in f:
                    last_line = json.loads(line)
            return last_line
            
        except Exception as e:
            print(f"Error executing workflow: {e}")
            return None

def run_hmc_workflow():
    command = ["java", "-cp", "target/classes", "qualia.Core", "hmcmulti"]
    output_file = "hmc-smoke/summary.json"
    
    workflow = AgenticWorkflow(command, output_file)
    summary_data = workflow.run()
    
    if summary_data:
        print("HMC Summary:")
        print(json.dumps(summary_data, indent=2))

def run_psi_workflow():
    command = ["java", "-cp", "target/classes", "qualia.Core", "unified"]
    output_file = "data/logs/unified.jsonl"
    
    workflow = AgenticWorkflow(command, output_file)
    # The output of this workflow is a JSONL file, so we'll just confirm it ran
    result = workflow.run()
    
    if result is not None:
        print("PSI Workflow completed successfully.")
        # We could add more sophisticated processing of the JSONL file here
        # For now, we'll just confirm it's not empty
        with open(output_file, 'r') as f:
            for line in f:
                print(line.strip())
                break # Just print the first line

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "psi":
        run_psi_workflow()
    else:
        run_hmc_workflow()