
"""
Extract Python code blocks from a directory of Markdown files, patch every save-figure call so it writes into one central folder (using each Markdown basename for the image), execute each snippet to generate and save the figures, and dump the extracted code as .py scripts.

Usage example
-------------
python extract_plots.py \
    --md_dir  /path/to/markdowns \
    --img_dir /path/to/output/images \
    --py_dir  /path/to/output/py
"""
import os
import re
import json
import argparse
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm  


def extract_python_code_from_md(file_path: str):
    """Return the list of ```python … ``` blocks inside a Markdown file."""
    with open(file_path, "r", encoding="utf-8") as fh:
        content = fh.read()
    return re.findall(r'```python(.*?)```', content, re.DOTALL)


def modify_code_to_save_in_desired_directory(code: str, save_path: str):
    """Patch every known save-figure call so it writes to *save_path*."""
    code = re.sub(r'fig\.write_image\(["\']([^"\']+)["\'](.*)\)',
                  rf'fig.write_image("{save_path}"\2)', code)
    code = re.sub(r'plt\.savefig\(["\']([^"\']+)["\'](.*)\)',
                  rf'plt.savefig("{save_path}"\2)', code)
    code = re.sub(r'fig\.savefig\(["\']([^"\']+)["\'](.*)\)',
                  rf'plt.savefig("{save_path}"\2)', code)
    code = re.sub(r'pio\.write_image\(([^,]+),\s*["\']([^"\']+)["\'](.*)\)',
                  rf'pio.write_image(\1, "{save_path}"\3)', code)
    code = re.sub(r'hv\.save\(([^,]+),\s*["\']([^"\']+)["\'](.*)\)',
                  rf'hv.save(\1, "{save_path}"\3)', code)
    code = re.sub(r'save\(["\']([^"\']+)["\'](.*)\)',
                  rf'save("{save_path}"\2)', code)
    code = re.sub(r'mpf\.plot\(([^)]*?),\s*savefig=["\']([^"\']+)["\'](.*?)\)',
                  rf'mpf.plot(\1, savefig="{save_path}"\3)', code)
    return code


def ensure_imports(code: str):
    """Prepend imports that snippets often forget."""
    mandatory = ["import io"]
    return "\n".join(mandatory) + "\n" + code


def run_code(code_blocks, save_path):
    """Execute each patched block; return True if *all* run successfully."""
    ok = True
    for code in code_blocks:
        try:
            patched = ensure_imports(modify_code_to_save_in_desired_directory(code, save_path))
            exec(patched, {})        # fresh namespace for safety
        except Exception as exc:
            print(f"[ERROR] while executing code for {save_path}: {exc}")
            ok = False
    return ok


def save_python_code(file_path, code_blocks, py_save_directory):
    """Dump every extracted block into <py_save_directory>/<basename>.py."""
    base = Path(file_path).stem
    dst  = Path(py_save_directory) / f"{base}.py"
    with open(dst, "w", encoding="utf-8") as fh:
        for block in code_blocks:
            fh.write(block.strip() + "\n\n")
    return dst



def process_md_directory(md_root_dir: str,
                         img_dir: str,
                         py_dir: str,
                         skip_first: int = 0,
                         checkpoint_every: int = 100,
                         checkpoint_path: str | None = None):
    """
    Walk *md_root_dir* recursively, render every Markdown file, and keep stats.
    Now uses the Markdown basename for image filenames to match the .py scripts.
    """
    md_files = [p for p in Path(md_root_dir).rglob("*.md")]
    md_files.sort()          # deterministic order

    img_dir = Path(img_dir);  img_dir.mkdir(parents=True, exist_ok=True)
    py_dir  = Path(py_dir);   py_dir.mkdir(parents=True, exist_ok=True)

    total, executed = 0, 0
    unexecutable = defaultdict(int)   # path → count

    for idx, md_path in enumerate(tqdm(md_files, desc="Rendering", unit="file")):
        total += 1
        if idx < skip_first:
            continue

        # Use the same basename as the .py file for the image:
        base = md_path.stem
        img_name = f"{base}.png"  # or .png if you prefer
        save_path = str(img_dir / img_name)

        code_blocks = extract_python_code_from_md(md_path)
        if not code_blocks:
            continue   # nothing to run

        # dump the .py script
        save_python_code(md_path, code_blocks, py_dir)

        # execute and render
        if run_code(code_blocks, save_path):
            executed += 1
        else:
            unexecutable[str(md_path)] += 1

        # periodic checkpoint
        if checkpoint_path and total % checkpoint_every == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as ck:
                json.dump(unexecutable, ck, indent=2)

    # final statistics
    success_pct = 100 * executed / total if total else 0
    print(f"\nFinished: {executed}/{total} ({success_pct:.2f} %) ran without error")

    if checkpoint_path:
        with open(checkpoint_path, "w", encoding="utf-8") as ck:
            json.dump(unexecutable, ck, indent=2)


def main():
    p = argparse.ArgumentParser(description="Render plots from Markdown files.")
    p.add_argument("--md_dir",  required=True, help="Directory containing .md files")
    p.add_argument("--img_dir", required=True, help="Where rendered images go")
    p.add_argument("--py_dir",  required=True, help="Where patched .py scripts go")
    p.add_argument("--resume",  type=int, default=0,
                   help="Skip the first N Markdown files (default: 0)")
    p.add_argument("--checkpoint", default=None,
                   help="Write unexecutable stats to this JSON file periodically")
    args = p.parse_args()

    process_md_directory(args.md_dir, args.img_dir, args.py_dir,
                         skip_first=args.resume,
                         checkpoint_every=100,
                         checkpoint_path=args.checkpoint)

if __name__ == "__main__":
    main()
