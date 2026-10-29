import os
import json
import platform
import argparse
from salt import get_student_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-id", required=True, help="8-digit Student ID")
    args = parser.parse_args()

    profile = {
        "os": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version()
    }

    os.makedirs("results", exist_ok=True)
    with open("results/host_profile.json", "w") as f:
        json.dump(profile, f, indent=4)

    params = get_student_params(args.student_id)
    with open("results/salted_params.json", "w") as f:
        json.dump(params, f, indent=4)

    print(f"[PASS] Milestone 0 completed. Salted N = {params['N']}")

if __name__ == "__main__":
    main()