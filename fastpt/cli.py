# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import difflib
import filecmp
from glob import glob
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from . import hipify_python  

_DEFAULT_HIP_ARCHS = "gfx928;gfx936;gfx938"
_DEFAULT_THIRD_PARTY_IGNORES = ("3rdparty", "third_party")

def _parse_cli_list_arg(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)

    value = value.strip()
    if not value:
        return []

    if value[0] == "[" and value[-1] == "]":
        value = value[1:-1]

    if not value:
        return []

    items = []
    for item in re.split(r"[,;\s]+", value):
        item = item.strip().strip("\"'")
        if item:
            items.append(item)
    return items


def _resolve_ignore_list(value, base_dir=None):
    items = _parse_cli_list_arg(value)
    if len(items) == 1:
        candidate = items[0]
        candidate_path = candidate
        if base_dir and not os.path.isabs(candidate_path):
            candidate_path = os.path.join(base_dir, candidate_path)
        if os.path.isfile(candidate_path) and os.path.splitext(candidate_path)[1].lower() in {".txt", ".list", ".ignore"}:
            resolved = []
            with open(candidate_path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        resolved.append(line)
            return resolved
    return items


def _normalize_ignore_items(project_directory, ignores):
    if not ignores:
        return []

    project_directory = os.path.abspath(project_directory)
    project_basename = os.path.basename(os.path.normpath(project_directory))
    normalized_items = []

    for item in ignores:
        candidate = item
        if not os.path.isabs(candidate):
            candidate_posix = candidate.replace("\\", "/").strip("/")
            prefix = f"{project_basename}/"
            if candidate_posix.startswith(prefix):
                stripped = candidate_posix[len(prefix):]
                if stripped:
                    candidate = stripped
        normalized_items.append(candidate)

    return normalized_items


def _expand_ignore_patterns(project_directory, ignores):
    if not ignores:
        return []

    project_directory = os.path.abspath(project_directory)
    project_basename = os.path.basename(os.path.normpath(project_directory))
    expanded = []
    seen = set()
    all_files = []
    for abs_dirpath, _, filenames in os.walk(project_directory):
        for filename in filenames:
            all_files.append(os.path.join(abs_dirpath, filename))

    for item in ignores:
        candidate = item
        matched = False
        candidate_variants = [candidate]

        if not os.path.isabs(candidate):
            normalized = candidate.replace("\\", "/").strip("/")
            basename_prefix = f"{project_basename}/"
            if normalized.startswith(basename_prefix):
                stripped = normalized[len(basename_prefix):]
                if stripped and stripped not in candidate_variants:
                    candidate_variants.append(stripped)

        for variant in candidate_variants:
            if os.path.isabs(variant):
                if os.path.exists(variant) and variant not in seen:
                    expanded.append(variant)
                    seen.add(variant)
                    matched = True
                    break
            else:
                abs_candidate = os.path.abspath(os.path.join(project_directory, variant))
                if os.path.exists(abs_candidate) and abs_candidate not in seen:
                    expanded.append(abs_candidate)
                    seen.add(abs_candidate)
                    matched = True
                    break

        if matched:
            continue

        for path in all_files:
            rel_path = os.path.relpath(path, project_directory)
            rel_path_posix = rel_path.replace(os.sep, "/")
            for variant in candidate_variants:
                variant_posix = variant.replace("\\", "/")
                if (
                    os.path.basename(path) == variant
                    or rel_path == variant
                    or rel_path_posix == variant_posix
                ):
                    if path not in seen:
                        expanded.append(path)
                        seen.add(path)
                    matched = True
                    break
            if matched:
                break

        if not matched and candidate not in seen:
            expanded.append(candidate)
            seen.add(candidate)

    return expanded


def _warn_and_return(message):
    print(f"WARNING: {message}", file=sys.stderr)
    return


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _backup_file(path):
    backup_path = f"{path}.dtk_bak"
    shutil.copy2(path, backup_path)
    return backup_path


def _iter_backup_candidate_files(root_dir):
    extensions = (".cu", ".cuh", ".c", ".cc", ".cpp", ".h", ".hpp", ".in", ".cmake")
    for abs_dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            abs_path = os.path.join(abs_dirpath, filename)
            if abs_path.endswith(".dtk_bak"):
                continue
            if filename == "CMakeLists.txt" or filename.endswith(extensions):
                yield abs_path


def _files_equal(path_a, path_b):
    if not (os.path.exists(path_a) and os.path.exists(path_b)):
        return False
    return filecmp.cmp(path_a, path_b, shallow=False)


def _backup_changed_folder_files(original_root, target_root, ignored_files=None):
    original_root = os.path.abspath(original_root)
    target_root = os.path.abspath(target_root)
    ignored_rel_paths = {
        os.path.relpath(os.path.abspath(path), target_root).replace(os.sep, "/")
        for path in (ignored_files or [])
        if os.path.exists(os.path.abspath(path))
    }

    for original_path in _iter_backup_candidate_files(original_root):
        rel_path = os.path.relpath(original_path, original_root)
        rel_posix = rel_path.replace(os.sep, "/")
        if rel_posix in ignored_rel_paths:
            continue

        target_same_path = os.path.join(target_root, rel_path)
        target_hip_rel = hipify_python.get_hip_file_path_replace(rel_posix, True).replace("/", os.sep)
        target_hip_path = os.path.join(target_root, target_hip_rel)

        if os.path.exists(target_same_path):
            if not _files_equal(original_path, target_same_path):
                _backup_file(target_same_path)
            continue

        if os.path.exists(target_hip_path):
            bak_path = f"{target_same_path}.dtk_bak"
            _ensure_parent_dir(bak_path)
            shutil.copy2(original_path, bak_path)
            continue


def _run_hipify(project_directory, output_directory, includes, ignores=None, extra_files=None,
                header_include_dirs=None, hipify_extra_files_only=False, preserve_dir_structure=False,
                custom_map_list="",
                show_detailed=False):
    hipify_python.HIPIFY_FINAL_RESULT.clear()
    return hipify_python.hipify(
        project_directory=project_directory,
        output_directory=output_directory,
        includes=includes,
        ignores=ignores or [],
        header_include_dirs=header_include_dirs or [],
        custom_map_list=custom_map_list,
        extra_files=extra_files or [],
        hipify_extra_files_only=hipify_extra_files_only,
        show_detailed=show_detailed,
        show_progress=True,
        is_pytorch_extension=True,
        add_dtk_macros=False,
        preserve_dir_structure=preserve_dir_structure,
    )


def _materialize_single_file(source_file, generated_file, output_file, inplace):
    output_file = os.path.abspath(output_file)
    _ensure_parent_dir(output_file)

    if inplace:
        changed = os.path.abspath(source_file) != output_file or not _files_equal(source_file, generated_file)
        if changed:
            _backup_file(source_file)
        if changed and os.path.abspath(source_file) != output_file and os.path.exists(source_file):
            os.remove(source_file)

    shutil.copy2(generated_file, output_file)


def _hipify_file_impl(source_file, output_file=None, inplace=False, show_detailed=False, custom_map_list=""):
    source_file = os.path.abspath(source_file)
    if not os.path.isfile(source_file):
        raise FileNotFoundError(f"{source_file} does not exist or is not a file")

    if not inplace and not output_file:
        _warn_and_return("fastpt.hipify.file requires either -inplace or -o.")
        return

    if inplace and output_file:
        output_file = None

    with tempfile.TemporaryDirectory(prefix="fastpt_hipify_file_src_") as tmp_src:
        tmp_name = os.path.basename(source_file)
        tmp_input = os.path.join(tmp_src, tmp_name)
        shutil.copy2(source_file, tmp_input)

        with tempfile.TemporaryDirectory(prefix="fastpt_hipify_file_out_") as tmp_out:
            _run_hipify(
                project_directory=tmp_src,
                output_directory=tmp_out,
                includes=['*'],
                preserve_dir_structure=True,
                show_detailed=show_detailed,
                custom_map_list=custom_map_list,
            )
            generated_name = hipify_python.get_hip_file_path_replace(tmp_name, True)
            generated_file = os.path.join(tmp_out, generated_name)
            if not os.path.exists(generated_file):
                raise FileNotFoundError(f"Hipified output not found: {generated_file}")

            if inplace:
                source_root, source_ext = os.path.splitext(source_file)
                inplace_output = source_root + ".hip" if source_ext == ".cu" else source_file
                _materialize_single_file(source_file, generated_file, inplace_output, inplace=True)
                return

            _materialize_single_file(source_file, generated_file, output_file, inplace=False)


def _collect_folder_extra_files(project_directory):
    extra_files = []
    for abs_dirpath, _, filenames in os.walk(project_directory):
        for filename in filenames:
            if filename == "CMakeLists.txt":
                extra_files.append(os.path.join(abs_dirpath, filename))
    return extra_files


def _cmake_postprocess_tree(project_directory, arch=_DEFAULT_HIP_ARCHS):
    project_directory = os.path.abspath(project_directory)
    files = _cmake_files(Path(project_directory))
    transformed = {}
    packages = set()

    for path in files:
        rel = path.relative_to(project_directory)
        old = path.read_text(encoding="utf-8", errors="surrogateescape")
        new, _ = _cmake_transform_text(old, rel, arch)
        transformed[rel] = new
        packages.update(_cmake_detect_packages(new))

    top = Path("CMakeLists.txt")
    if top in transformed:
        transformed[top] = _cmake_insert_top_level_block(transformed[top], arch, packages)

    for rel, text in transformed.items():
        (Path(project_directory) / rel).write_text(text, encoding="utf-8")


def _hipify_folder_impl(project_directory, output_directory=None, inplace=False, show_detailed=False,
                        ignores=None, custom_map_list=""):
    project_directory = os.path.abspath(project_directory)
    if not os.path.isdir(project_directory):
        raise NotADirectoryError(f"{project_directory} does not exist or is not a directory")

    extra_files = _collect_folder_extra_files(project_directory)
    raw_ignores = list(ignores or [])
    raw_ignores.extend(_DEFAULT_THIRD_PARTY_IGNORES)
    ignore_items = _normalize_ignore_items(project_directory, raw_ignores)

    if inplace and output_directory and os.path.abspath(output_directory) != project_directory:
        output_directory = os.path.abspath(output_directory)
        os.makedirs(output_directory, exist_ok=True)
        shutil.copytree(project_directory, output_directory, dirs_exist_ok=True)
        _hipify_folder_impl(
            project_directory=output_directory,
            output_directory=None,
            inplace=True,
            show_detailed=show_detailed,
            ignores=ignore_items,
            custom_map_list=custom_map_list,
        )
        return

    ignores = _expand_ignore_patterns(project_directory, ignore_items)

    if inplace or (output_directory and os.path.abspath(output_directory) == project_directory):
        with tempfile.TemporaryDirectory(prefix="fastpt_hipify_folder_src_") as tmp_src:
            tmp_project = os.path.join(tmp_src, os.path.basename(project_directory))
            shutil.copytree(project_directory, tmp_project, dirs_exist_ok=True)
            _run_hipify(
                project_directory=tmp_project,
                output_directory=project_directory,
                includes=['*'],
                ignores=ignores or [],
                extra_files=extra_files,
                custom_map_list=custom_map_list,
                preserve_dir_structure=True,
                show_detailed=show_detailed,
            )
            _cmake_postprocess_tree(project_directory)
            _backup_changed_folder_files(tmp_project, project_directory, ignored_files=ignores)
        return

    if not output_directory:
        output_directory = project_directory.rstrip("/") + "_dtk"

    output_directory = os.path.abspath(output_directory)
    _run_hipify(
        project_directory=project_directory,
        output_directory=output_directory,
        includes=['*'],
        ignores=ignores or [],
        extra_files=extra_files,
        custom_map_list=custom_map_list,
        preserve_dir_structure=True,
        show_detailed=show_detailed,
    )
    _cmake_postprocess_tree(output_directory)


_CMAKE_LIB_MAP = {
    "CUDA::cudart_static": "hip::host",
    "CUDA::cudart": "hip::host",
    "CUDA::cuda_driver": "hip::host",
    "CUDA::cublasLt": "roc::hipblaslt",
    "CUDA::cublas": "roc::hipblas",
    "CUDA::cufft": "hip::hipfft",
    "CUDA::curand": "hip::hiprand",
    "CUDA::cusparse": "roc::hipsparse",
    "CUDNN::cudnn": "hipdnn",
    "NCCL::nccl": "roc::rccl",
    "cudart_static": "hip::host",
    "cudart": "hip::host",
    "cublasLt": "roc::hipblaslt",
    "cublas": "roc::hipblas",
    "cufft": "hip::hipfft",
    "curand": "hip::hiprand",
    "cusparse": "roc::hipsparse",
    "cudnn": "hipdnn",
    "nccl": "roc::rccl",
}

_CMAKE_PACKAGE_MAP = {
    "roc::hipblas": "find_package(hipblas REQUIRED)",
    "roc::hipblaslt": "find_package(hipblaslt REQUIRED)",
    "hip::hipfft": "find_package(hipfft REQUIRED)",
    "hip::hiprand": "find_package(hiprand REQUIRED)",
    "roc::hipsparse": "find_package(hipsparse REQUIRED)",
    "roc::rccl": "find_package(rccl REQUIRED)",
}

_CMAKE_REVIEW_PATTERNS = [
    (re.compile(r"FindCUDA|cuda_compile", re.I), "Legacy FindCUDA/cuda_compile macros may require project-level replacement"),
    (re.compile(r"cutlass", re.I), "CUTLASS requires a decision to bypass it or integrate hytlass"),
    (re.compile(r"(?<![A-Za-z0-9_])/(?:usr/local/)?cuda(?:/|\b)", re.I), "A hardcoded CUDA path was detected; update it manually for the project's DTK path"),
    (re.compile(r"\bcuda[A-Za-z0-9_]*\b"), "CUDA-named variables or functions remain; verify whether they are compatibility switches"),
    (re.compile(r"\bCUDAToolkit\b|\bCUDA\b"), "CUDA/CUDAToolkit keywords remain; verify whether they must be preserved"),
]

_CMAKE_IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "build",
    "cmake-build-debug",
    "cmake-build-release",
    "dist",
    "__pycache__",
    ".pytest_cache",
}


def _cmake_files(root):
    files = []
    for path in root.rglob("*"):
        if any(part in _CMAKE_IGNORE_DIRS for part in path.parts):
            continue
        if path.name == "CMakeLists.txt" or path.suffix == ".cmake":
            files.append(path)
    return sorted(files)


def _cmake_replace_word(text, src, dst):
    if "::" in src:
        return text.replace(src, dst)
    return re.sub(rf"(?<![A-Za-z0-9_:+.-]){re.escape(src)}(?![A-Za-z0-9_:+.-])", dst, text)


def _cmake_strip_cuda_gencode(line, arch):
    out = re.sub(r"-gencode[= ][^\"')\s]+(?:,[^\"')\s]+)?", "", line)
    out = re.sub(r"--generate-code[= ][^\"')\s]+(?:,[^\"')\s]+)?", "", out)
    out = re.sub(r"-arch=sm_[0-9]+", f"--offload-arch={arch}", out)
    out = re.sub(r"-arch compute_[0-9]+", f"--offload-arch={arch}", out)
    return out


def _cmake_dedupe_link_libraries(line):
    if "target_link_libraries" not in line:
        return line

    match = re.search(r"target_link_libraries\s*\((.*)\)\s*$", line)
    if not match:
        return line

    body = match.group(1)
    parts = body.split()
    if len(parts) < 2:
        return line

    head = [parts[0]]
    if len(parts) >= 2 and parts[1] in {"PRIVATE", "PUBLIC", "INTERFACE"}:
        head.append(parts[1])
        lib_start = 2
    else:
        lib_start = 1

    libs = []
    seen = set()
    for item in parts[lib_start:]:
        if item not in seen:
            libs.append(item)
            seen.add(item)

    return f"{line[:match.start(1)]}{' '.join(head + libs)}{line[match.end(1):]}"


def _cmake_transform_line(line, arch):
    stripped = line.lstrip()
    if stripped.startswith("#"):
        return line

    out = line
    out = re.sub(r"\benable_language\(([^)]*)\bCUDA\b([^)]*)\)", r"enable_language(\1HIP\2)", out)
    out = re.sub(r"\bproject\(([^)]*)\bCUDA\b([^)]*)\)", r"project(\1HIP\2)", out)
    if re.search(r"set\s*\(\s*CMAKE_CUDA_ARCHITECTURES\s+", out, flags=re.I):
        indent = re.match(r"\s*", out).group(0)
        out = f'{indent}set(CMAKE_HIP_ARCHITECTURES "{arch}")\n'
        return out

    if re.search(r"set\s*\(\s*CUDA_ARCHITECTURES\s+", out, flags=re.I):
        indent = re.match(r"\s*", out).group(0)
        out = f'{indent}set(HIP_ARCHITECTURES "{arch}")\n'
        return out

    replacements = {
        "CMAKE_CUDA_ARCHITECTURES": "CMAKE_HIP_ARCHITECTURES",
        "CUDA_ARCHITECTURES": "HIP_ARCHITECTURES",
        "CMAKE_CUDA_STANDARD_REQUIRED": "CMAKE_HIP_STANDARD_REQUIRED",
        "CMAKE_CUDA_STANDARD": "CMAKE_HIP_STANDARD",
        "CMAKE_CUDA_EXTENSIONS": "CMAKE_HIP_EXTENSIONS",
        "CMAKE_CUDA_FLAGS": "CMAKE_HIP_FLAGS",
        "CMAKE_CUDA_COMPILER": "CMAKE_HIP_COMPILER",
        "CUDA_NVCC_FLAGS": "CMAKE_HIP_FLAGS",
        "CUDAToolkit_ROOT": "HIP_ROOT_DIR",
        "CUDA_TOOLKIT_ROOT_DIR": "HIP_ROOT_DIR",
        "CUDA_HOME": "ROCM_HOME",
        "CUDA_PATH": "ROCM_PATH",
    }
    for src, dst in replacements.items():
        out = _cmake_replace_word(out, src, dst)

    out = re.sub(
        r"(\b(?:CMAKE_HIP_ARCHITECTURES|HIP_ARCHITECTURES)\b\s+)([0-9][0-9; ]*)",
        lambda m: m.group(1) + arch,
        out,
    )

    if re.search(r"find_package\s*\(\s*(CUDA|CUDAToolkit)\b", out):
        indent = re.match(r"\s*", out).group(0)
        old = out.rstrip("\n").strip()
        out = f"{indent}# FASTPT_DTK: replaced `{old}`; HIP packages are injected at top level.\n"

    out = re.sub(r"\bnvcc\b", "hipcc", out, flags=re.I)
    out = re.sub(r"\.cu\b", ".hip", out)
    out = re.sub(r"\bcuda_add_library\s*\(", "add_library(", out)
    out = re.sub(r"\bcuda_add_executable\s*\(", "add_executable(", out)
    out = _cmake_strip_cuda_gencode(out, arch)

    for src, dst in _CMAKE_LIB_MAP.items():
        out = _cmake_replace_word(out, src, dst)

    out = _cmake_dedupe_link_libraries(out)

    return out


def _cmake_detect_packages(text):
    packages = {"find_package(hip REQUIRED)"}
    for target, package in _CMAKE_PACKAGE_MAP.items():
        if re.search(
            rf"(?<![A-Za-z0-9_:+.-]){re.escape(target)}(?![A-Za-z0-9_:+.-])",
            text,
        ):
            packages.add(package)
    return packages


def _cmake_transform_text(text, rel_path, arch):
    new_lines = []
    warnings = []
    for lineno, line in enumerate(text.splitlines(keepends=True), start=1):
        new_line = _cmake_transform_line(line, arch)
        for pattern, message in _CMAKE_REVIEW_PATTERNS:
            if "FASTPT_DTK:" in new_line:
                continue
            if pattern.search(new_line) and not new_line.lstrip().startswith("#"):
                warnings.append(f"{rel_path}:{lineno}: {message}: {new_line.strip()}")
                break
        new_lines.append(new_line)
    return "".join(new_lines), warnings


def _cmake_insert_top_level_block(text, arch, packages):
    if "FASTPT_DTK_CMAKE_BEGIN" in text:
        return text
    filtered_packages = []
    for package in sorted(packages):
        pkg_name = re.search(r"find_package\((\w+)", package)
        if pkg_name and re.search(rf"find_package\s*\(\s*{re.escape(pkg_name.group(1))}\b", text):
            continue
        filtered_packages.append(package)
    block = [
        "# FASTPT_DTK_CMAKE_BEGIN\n",
        "# Generated by fastpt.hipify.cmake.\n",
        f'set(CMAKE_HIP_ARCHITECTURES "{arch}" CACHE STRING "HIP GPU architectures" FORCE)\n',
    ]
    block.extend(line + "\n" for line in filtered_packages)
    block.append("# FASTPT_DTK_CMAKE_END\n\n")
    block_text = "".join(block)
    lines = text.splitlines(keepends=True)
    insert_after = -1
    for index, line in enumerate(lines):
        if re.match(r"\s*project\s*\(", line):
            insert_after = index
            break
    if insert_after < 0:
        for index, line in enumerate(lines):
            if re.match(r"\s*cmake_minimum_required\s*\(", line):
                insert_after = index
                break
    if insert_after >= 0:
        return "".join(lines[:insert_after + 1]) + block_text + "".join(lines[insert_after + 1:])
    return block_text + text


def _cmake_unified_diff(path, old, new):
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path) + ".fastpt-dtk",
        )
    )


_CMAKE_CU_SOURCE_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(?P<path>[^\s\"'()]+)\.cu\b")


def _cmake_classify_hip_sources(root, originals, transformed):
    missing = []
    unverified = []
    for rel, old in originals.items():
        new = transformed[rel]
        cmake_file = root / rel
        for lineno, (old_line, new_line) in enumerate(
            zip(old.splitlines(), new.splitlines()), start=1
        ):
            if ".cu" not in old_line or ".hip" not in new_line:
                continue

            converted_entries = 0
            for match in _CMAKE_CU_SOURCE_PATTERN.finditer(old_line):
                cuda_spec = match.group("path") + ".cu"
                hip_spec = cuda_spec[:-3] + ".hip"
                if hip_spec not in new_line:
                    continue
                converted_entries += 1

                resolved = hip_spec.replace("${CMAKE_CURRENT_SOURCE_DIR}", str(cmake_file.parent))
                resolved = resolved.replace("${CMAKE_CURRENT_LIST_DIR}", str(cmake_file.parent))
                if "$" in resolved:
                    unverified.append(
                        f"{rel}:{lineno}: cannot verify HIP source generated from `{cuda_spec}`: "
                        f"unresolved CMake variable in `{hip_spec}`"
                    )
                    continue

                hip_path = Path(resolved)
                if not hip_path.is_absolute():
                    hip_path = cmake_file.parent / hip_path
                if not glob(str(hip_path)):
                    missing.append(
                        f"{rel}:{lineno}: missing HIP source for `{cuda_spec}`; expected `{hip_spec}`"
                    )

            if old_line.count(".cu") > converted_entries:
                unverified.append(
                    f"{rel}:{lineno}: dynamic CUDA source expression was converted but cannot be statically verified"
                )
    return missing, unverified


def _hipify_cmake_impl(repo, arch=_DEFAULT_HIP_ARCHS, apply=False):
    repo = Path(repo).resolve()
    if not repo.exists():
        raise FileNotFoundError(f"repo not found: {repo}")

    files = _cmake_files(repo)
    originals = {}
    transformed = {}
    warnings = []
    packages = set()

    for path in files:
        rel = path.relative_to(repo)
        old = path.read_text(encoding="utf-8", errors="surrogateescape")
        new, file_warnings = _cmake_transform_text(old, rel, arch)
        originals[rel] = old
        transformed[rel] = new
        packages.update(_cmake_detect_packages(new))
        warnings.extend(file_warnings)

    source_transformed = dict(transformed)
    top = Path("CMakeLists.txt")
    if top in transformed:
        transformed[top] = _cmake_insert_top_level_block(transformed[top], arch, packages)

    changed = [rel for rel in sorted(transformed) if transformed[rel] != originals[rel]]
    diffs = [_cmake_unified_diff(rel, originals[rel], transformed[rel]) for rel in changed]
    missing_sources, unverified_sources = _cmake_classify_hip_sources(
        repo, originals, source_transformed
    )
    source_warnings = missing_sources + unverified_sources
    if apply:
        for rel in changed:
            target = repo / rel
            _backup_file(target)
            target.write_text(transformed[rel], encoding="utf-8")
        print(f"updated CMake files in: {repo}")
        print(f"changed CMake files: {len(changed)}")
        print(f"high-risk residual items: {len(warnings)}")
        if source_warnings:
            print("WARNING: CMake was updated with missing or unverified HIP source paths:", file=sys.stderr)
            print("\n".join(source_warnings), file=sys.stderr)
        return True

    print("".join(diffs) if diffs else "No CMake changes suggested.")
    if warnings:
        print("\n# High-risk residual items", file=sys.stderr)
        print("\n".join(warnings), file=sys.stderr)
    if source_warnings:
        print("\n# HIP source validation warnings", file=sys.stderr)
        print("\n".join(source_warnings), file=sys.stderr)
    return True

def hipify():
    parser = argparse.ArgumentParser(
        description="Hipify CUDA code to HIP.",
    )

    parser.add_argument(
        "-d",
        "--project_directory",
        type=str,
        required=True,
        help="Specify the project directory, which contains the CUDA code to be hipified."
    )

    parser.add_argument(
        "-o",
        "--output_directory",
        type=str,
        help="Specify the output directory, which will contain the hipified code. If not specified, the output will be ${project_directory}_dtk."
    )

    parser.add_argument(
        "-v",
        "--show_detailed",
        action="store_true",
        help="Whether to show detailed statistics."
    )

    parser.add_argument(
        "-i",
        "--includes",
        type=str,
        default=('*',),
        help="Specify the files to include for hipification. You can provide a list of files to include. Default is '*' (all files), '*' may solve almost all cases."
    )
    parser.add_argument(
        "-I",
        "--ignores",
        type=str,
        default=(),
        help="Specify the files to ignore during hipification. You can provide a list of files to ignore."
    )
    parser.add_argument(
        "--header_include_dirs",
        type=str,
        default=(),
        help="Specify header include directories that should also be scanned for header hipify."
    )
    parser.add_argument(
        "--custom_map_list",
        type=str,
        default="",
        help="Specify the path to custom_hipify_mappings.json or another custom mapping json file."
    )
    parser.add_argument(
        "--custom",
        type=str,
        default="",
        help="Alias of --custom_map_list."
    )
    parser.add_argument(
        "--extra_files",
        type=str,
        default=(),
        help="Specify extra files to hipify. These files have higher priority than ignores."
    )
    parser.add_argument(
        "--hide_progress",
        action="store_false",
        default=True,
        help="Whether to hide hipification progress. Default is False, "
    )
    parser.add_argument(
        "--is_pytorch_extension",
        type=bool,
        default=True,
        help="Whether to process torch-related code files. Default is True."
    )
    parser.add_argument(
        "--hipify_extra_files_only",
        action="store_true",
        help="Whether to hipify only the files provided by --extra_files."
    )

    parser.add_argument(
        "--add_dtk_macro",
        action="store_true",
        help="Whether to add DTK macro header file."
    )

    parser.add_argument(
        "--preserve_dir_structure",
        action="store_true",
        default=False,
        help="Whether to preserve the directory structure during hipification. Default is False."
    )
    
    
    # Show help when no arguments or only -h/--help are supplied.
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        parser.print_help()
        return

    args = parser.parse_args()
    includes = _parse_cli_list_arg(args.includes) or ['*']
    ignores = _parse_cli_list_arg(args.ignores)
    header_include_dirs = _parse_cli_list_arg(args.header_include_dirs)
    extra_files = _parse_cli_list_arg(args.extra_files)
    custom_map_list = args.custom or args.custom_map_list

    hipify_python.hipify(
        project_directory=args.project_directory,
        output_directory=args.output_directory,
        includes=includes,
        show_detailed=args.show_detailed,
        is_pytorch_extension=args.is_pytorch_extension,
        ignores=ignores,
        header_include_dirs=header_include_dirs,
        custom_map_list=custom_map_list,
        extra_files=extra_files,
        hipify_extra_files_only=args.hipify_extra_files_only,
        add_dtk_macros=args.add_dtk_macro,
        preserve_dir_structure=args.preserve_dir_structure
    )


def hipify_file():
    parser = argparse.ArgumentParser(
        description="Hipify a single CUDA/C++ source file.",
    )
    parser.add_argument("source_file", type=str, help="Source file to hipify.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file path. If same as input, overwrite the original file."
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Backup source to .dtk_bak and write hipified output in place."
    )
    parser.add_argument(
        "--custom",
        default="custom_hipify_mappings.json",
        help="Custom hipify mapping json file. Defaults to custom_hipify_mappings.json."
    )
    parser.add_argument(
        "-v",
        "--show_detailed",
        action="store_true",
        help="Whether to show detailed statistics."
    )

    args = parser.parse_args()
    _hipify_file_impl(
        source_file=args.source_file,
        output_file=args.output,
        inplace=args.inplace,
        show_detailed=args.show_detailed,
        custom_map_list=args.custom,
    )


def hipify_folder():
    parser = argparse.ArgumentParser(
        description="Hipify a source folder while preserving relative paths.",
    )
    parser.add_argument("project_directory", type=str, help="Project directory to hipify.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory. If omitted, defaults to <project_directory>_dtk."
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Backup source files to .dtk_bak and write hipified output in place."
    )
    parser.add_argument(
        "--ignore",
        default=(),
        help="Ignore files. Supports single file, comma/space/semicolon separated string, or a text file with one path per line."
    )
    parser.add_argument(
        "--custom",
        default="custom_hipify_mappings.json",
        help="Custom hipify mapping json file. Defaults to custom_hipify_mappings.json."
    )
    parser.add_argument(
        "-v",
        "--show_detailed",
        action="store_true",
        help="Whether to show detailed statistics."
    )

    args = parser.parse_args()
    ignores = _resolve_ignore_list(args.ignore, base_dir=args.project_directory)
    _hipify_folder_impl(
        project_directory=args.project_directory,
        output_directory=args.output,
        inplace=args.inplace,
        show_detailed=args.show_detailed,
        ignores=ignores,
        custom_map_list=args.custom,
    )


def hipify_cmake():
    parser = argparse.ArgumentParser(
        description="Inspect or apply CMake/CMakeLists conversion on an existing HIP source tree.",
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="Application repository root."
    )
    parser.add_argument(
        "--arch",
        default=os.environ.get("FASTPT_DTK_ARCH", _DEFAULT_HIP_ARCHS),
        help="HIP architecture, e.g. gfx936."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Modify CMake files in place and report missing or unverified HIP sources."
    )
    args = parser.parse_args()
    return 0 if _hipify_cmake_impl(
        repo=args.repo,
        arch=args.arch,
        apply=args.apply,
    ) else 1


if __name__ == "__main__":
    hipify()
