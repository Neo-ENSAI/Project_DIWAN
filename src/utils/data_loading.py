from pathlib import Path
import pandas as pd
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
