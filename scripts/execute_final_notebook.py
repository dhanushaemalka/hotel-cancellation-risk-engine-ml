"""Run the master notebook with the project-local Python dependencies."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".notebook-runtime"
sys.path.insert(0, str(RUNTIME))
os.environ["PYTHONPATH"] = str(RUNTIME)
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(ROOT / ".notebook-cache" / "matplotlib")
os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / ".notebook-cache" / "jupyter")
os.environ["IPYTHONDIR"] = str(ROOT / ".notebook-cache" / "ipython")
# Bound parallel workers to avoid multiplying memory use in cross-validation.
os.environ["LOKY_MAX_CPU_COUNT"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec


def main():
    source = ROOT / "notebooks" / "hotel_booking_demand_master.ipynb"
    output = ROOT / "notebooks" / "hotel_booking_demand_final_executed.ipynb"
    resume_final = "--resume-final" in sys.argv
    nb = nbformat.read(output if resume_final else source, as_version=4)
    if not resume_final:
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
    km = KernelManager(kernel_name="python3")
    km._kernel_spec = KernelSpec(
        argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        display_name="Project Python", language="python",
    )

    def start(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"START cell {cell_index + 1}: {cell.source.splitlines()[0]}", flush=True)

    def done(cell, cell_index, **kwargs):
        nbformat.write(nb, output)
        for item in cell.get("outputs", []):
            if item.output_type == "stream":
                print(item.text, end="", flush=True)
        print(f"SAVED cell {cell_index + 1}", flush=True)

    client = NotebookClient(
        nb, km=km, timeout=None, resources={"metadata": {"path": str(ROOT)}},
        on_cell_start=start, on_cell_executed=done,
    )
    try:
        if resume_final:
            bootstrap = nbformat.v4.new_code_cell('''
import sys
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from src.data_loader import load_raw_data
from src.preprocessing import clean_raw_data, engineer_features
root = Path.cwd()
df = engineer_features(clean_raw_data(load_raw_data()))
X = df.drop(columns=["is_canceled"])
y = df["is_canceled"]
_, X_test_raw, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
preprocessor = joblib.load(root / "data/processed/preprocessor.joblib")
X_test = preprocessor.transform(X_test_raw)
champion_model = joblib.load(root / "data/processed/models/xgboost.joblib")
''')
            with client.setup_kernel(cwd=str(ROOT)):
                nb.cells.insert(0, bootstrap)
                try:
                    client.execute_cell(bootstrap, 0, execution_count=0)
                finally:
                    nb.cells.pop(0)
                cell = nb.cells[22]
                cell.outputs = []
                client.execute_cell(cell, 22, execution_count=12)
        else:
            client.execute(cwd=str(ROOT))
    finally:
        nbformat.write(nb, output)
    print(f"SUCCESS: {output}", flush=True)


if __name__ == "__main__":
    main()
