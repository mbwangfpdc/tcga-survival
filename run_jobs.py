import os
import subprocess
import time

# Run these different configuration combinations

pca_sizes = [4, 8, 16, 32, 64, 128, 256, 512]
models = ["cox"]
ensembles = ["mean", "cox"]

for pca_size in pca_sizes:
    for model in models:
        for ensemble in ensembles:
            # We can't handle these for now (linear model degrades with feature size)
            if ensemble == "cat" and pca_size > 128:
                continue
            rundir = f"{pca_size}{model}{ensemble}"
            cmd = [
                    "./main.py",
                    "--features",
                    "all",
                    "--solo_features",
                    "demo",
                    "project",
                    "cancer",
                    "--model",
                    f"{model}",
                    "--rundir",
                    rundir,
                    "--pca_pre_join",
                    f"{pca_size}",
                    "--ensemble",
                    f"{ensemble}",
                ]
            start = time.time()
            print(f"Running command '{' '.join(cmd)}'")
            result = subprocess.run(
                cmd,
                text=True,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE
            )
            if result.stdout:
                with open(
                    os.path.join("results", rundir, "stdout.txt"), "w+"
                ) as stdout:
                    stdout.write(result.stdout)
            if result.returncode != 0:
                print(f"ERROR {result.returncode}, stopping")
                exit(1)
            print(f"Finished after {int(time.time() - start)} seconds")
