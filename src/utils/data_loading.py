import os
import posixpath
import tempfile
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import SimpleITK as sitk


def loading_data(path):
    base_dir = Path(path)
    df = pd.DataFrame(columns=["sample", "LS", "HS", "CT"])

    for sample_path in sorted(base_dir.glob('sample_*')):
        print(f"--- Traitement de : {sample_path.name} ---")
        high_edep_file = sample_path / 'high_edep.mhd'
        low_edep_file = sample_path / 'low_edep.mhd'
        ct_file = sample_path / 'ct.mhd'

        hs_img = sitk.ReadImage(str(high_edep_file))
        hs = sitk.GetArrayFromImage(hs_img)


        ls_img = sitk.ReadImage(str(low_edep_file))
        ls = sitk.GetArrayFromImage(ls_img)


        ct_img = sitk.ReadImage(str(ct_file))
        ct = sitk.GetArrayFromImage(ct_img)

        # Ajout au DataFrame
        df.loc[len(df)] = {
            "sample": sample_path.name,
            "HS": hs,
            "LS": ls,
            "CT": ct
        }

    return df

def loading_data_s3(path):
    fs = s3fs.S3FileSystem(
        client_kwargs={
            "endpoint_url": "https://" + os.environ["AWS_S3_ENDPOINT"]
        }
    )

    df = pd.DataFrame(columns=["sample", "LS", "HS", "CT"])
    sample_paths = sorted(fs.glob(f"{path.rstrip('/')}/sample_*"))

    total = len(sample_paths)
    start_time = time.perf_counter()

    print(f"Starting loading of {total} samples...", flush=True)

    for i, sample_path in enumerate(sample_paths):
        sample_name = posixpath.basename(sample_path.rstrip("/"))
        images = {}

        for key, filename in [
            ("HS", "high_edep.mhd"),
            ("LS", "low_edep.mhd"),
            ("CT", "ct.mhd"),
        ]:
            mhd_path = posixpath.join(sample_path, filename)

            with fs.open(mhd_path, "r") as f:
                header = f.read()

            dimensions = None
            raw_filename = None

            for line in header.splitlines():
                line = line.strip()

                if line.startswith("DimSize"):
                    dimensions = tuple(
                        map(int, line.split("=", 1)[1].split())
                    )
                elif line.startswith("ElementDataFile"):
                    raw_filename = line.split("=", 1)[1].strip()

            if dimensions is None or raw_filename is None:
                raise ValueError(f"Invalid MHD header: {mhd_path}")

            raw_path = posixpath.join(sample_path, raw_filename)

            with fs.open(raw_path, "rb") as f:
                raw_bytes = f.read()

            image = np.frombuffer(raw_bytes, dtype="<f4")
            expected_size = np.prod(dimensions)

            if image.size != expected_size:
                raise ValueError(
                    f"Unexpected image size in {raw_path}: "
                    f"expected {expected_size}, got {image.size}"
                )

            images[key] = image.reshape(dimensions[::-1])

        df.loc[len(df)] = {
            "sample": sample_name,
            "HS": images["HS"],
            "LS": images["LS"],
            "CT": images["CT"],
        }

        # Progress report every 100 samples and at completion
        if (i + 1) % 100 == 0 or (i + 1) == total:
            elapsed = time.perf_counter() - start_time
            percentage = 100 * (i + 1) / total
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate

            print(
                f"[{i + 1}/{total}] {percentage:.1f}% | "
                f"Elapsed: {elapsed / 60:.1f} min | "
                f"Estimated remaining: {remaining / 60:.1f} min",
                flush=True
            )

    print(
        f"Finished! {total} samples loaded in "
        f"{(time.perf_counter() - start_time) / 60:.2f} minutes.",
        flush=True
    )

    return df

def loading_data_s3local(path):
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]

    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": S3_ENDPOINT_URL}
    )

    df = pd.DataFrame(columns=["sample", "LS", "HS", "CT"])

    sample_paths = sorted(fs.glob(f"{path.rstrip('/')}/sample_*"))

    for sample_path in sample_paths:
        sample_name = posixpath.basename(sample_path.rstrip("/"))
        print(f"--- Traitement de : {sample_name} ---")

        images = {}

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            for key, filename in [
                ("HS", "high_edep.mhd"),
                ("LS", "low_edep.mhd"),
                ("CT", "ct.mhd"),
            ]:
                s3_mhd_path = posixpath.join(sample_path, filename)
                local_mhd_path = tmp_dir / filename

                # Download the MHD header
                fs.get(s3_mhd_path, str(local_mhd_path))

                # Find the raw data file referenced by the MHD header
                with open(local_mhd_path, "r", encoding="utf-8") as f:
                    header = f.readlines()

                data_file = None

                for line in header:
                    if line.strip().lower().startswith("elementdatafile"):
                        data_file = line.split("=", 1)[1].strip()
                        break

                if data_file is None:
                    raise ValueError(
                        f"ElementDataFile not found in {s3_mhd_path}"
                    )

                # Download the external RAW file, if there is one
                if data_file.upper() != "LOCAL":
                    if data_file.upper() == "LIST":
                        raise NotImplementedError(
                            f"ElementDataFile = LIST is not supported: "
                            f"{s3_mhd_path}"
                        )

                    relative_path = Path(data_file)

                    if relative_path.is_absolute() or ".." in relative_path.parts:
                        raise ValueError(
                            f"Unexpected raw file path in {s3_mhd_path}: "
                            f"{data_file}"
                        )

                    s3_raw_path = posixpath.normpath(
                        posixpath.join(
                            posixpath.dirname(s3_mhd_path),
                            data_file.replace("\\", "/")
                        )
                    )

                    local_raw_path = tmp_dir / relative_path
                    local_raw_path.parent.mkdir(
                        parents=True, exist_ok=True
                    )

                    fs.get(s3_raw_path, str(local_raw_path))

                # Read the image from the local MHD file
                img = sitk.ReadImage(str(local_mhd_path))
                images[key] = sitk.GetArrayFromImage(img)

        df.loc[len(df)] = {
            "sample": sample_name,
            "HS": images["HS"],
            "LS": images["LS"],
            "CT": images["CT"],
        }

    return df