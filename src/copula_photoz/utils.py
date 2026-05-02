import numpy as np
import pandas as pd
from astropy.table import Table


def load_photo_z_cats(set = 'validation', # or all/training
                      
            
                      validation_spec_uberids = '/Users/sp624AA/Downloads/qmost_spv/waves_specz/speccat-matched-v6new.fits',
                      training_spec_ubderids = '/Users/sp624AA/Downloads/qmost_spv/waves_specz/speccat-matched_v5.fits',
                      all_spec_uberids = '/Users/sp624AA/Downloads/qmost_spv/waves_specz/speccat-matched-v6.fits',
                      photo_z_waves_n_path = '/Users/sp624AA/Downloads/qmost_spv/WAVES-N_Photoz_Combined_Sabine Bellstedt_v3.parquet',
                      photo_z_waves_s_path = '/Users/sp624AA/Downloads/qmost_spv/WAVES-S_Photoz_Combined_Sabine Bellstedt_v3.parquet',
                      photo_z_waves_wd10_path = '/Users/sp624AA/Downloads/qmost_spv/WD10_Photoz_Combined_Sabine Bellstedt_v3.parquet',
                      photo_z_waves_wd01_path = '/Users/sp624AA/Downloads/qmost_spv/WD01_Photoz_Combined_Sabine Bellstedt_v3.parquet',
                      photo_z_waves_wd02_path = '/Users/sp624AA/Downloads/qmost_spv/WD02_Photoz_Combined_Sabine Bellstedt_v3.parquet',
                      photo_z_waves_wd03_path = '/Users/sp624AA/Downloads/qmost_spv/WD03_Photoz_Combined_Sabine Bellstedt_v3.parquet'
                      ):

    if set not in ['validation', 'training', 'all_speczs', 'all_photozs']:
        raise ValueError(f"Unknown set {set} (must be 'validation', 'training' or 'all')")
    
    photo_z_cols_to_read = ['uberID', 'P020_comb', 'P080_comb' ,'zphot_invar', 'zphot_err', 
                            'starmask', 'ghostmask', 'mask', 'class', 'duplicate']
    photo_z_cols_to_keep = ['uberID', 'P020_comb', 'P080_comb' ,'zphot_invar', 'zphot_err']

    #full_cat = load_elmo_spec_z_sample()
    if set == 'validation':
        subset = Table.read(validation_spec_uberids).to_pandas()
        subset.rename(columns={'UBERID': 'uberID'}, inplace=True)
        subset['uberID'] = subset['uberID'].astype("int64")
    elif set == 'training':
        subset = Table.read(training_spec_ubderids).to_pandas()
        subset.rename(columns={'UBERID': 'uberID'}, inplace=True)
        subset['uberID'] = subset['uberID'].astype("int64")
    elif set == 'all_speczs':
        subset = Table.read(all_spec_uberids).to_pandas()
        subset.rename(columns={'UBERID': 'uberID'}, inplace=True)
        subset['uberID'] = subset['uberID'].astype("int64")

    # filter full cat to only include uberIDs in validation_subset

    full_cat = pd.read_parquet(photo_z_waves_n_path, columns=photo_z_cols_to_read)

    wide_s = pd.read_parquet(photo_z_waves_s_path, columns=photo_z_cols_to_read)
    full_cat = pd.concat([full_cat, wide_s], ignore_index=True, axis=0)
    del wide_s

    wd10 = pd.read_parquet(photo_z_waves_wd10_path, columns=photo_z_cols_to_read)
    full_cat = pd.concat([full_cat, wd10], ignore_index=True, axis=0)
    del wd10

    wd01 = pd.read_parquet(photo_z_waves_wd01_path, columns=photo_z_cols_to_read)
    full_cat = pd.concat([full_cat, wd01], ignore_index=True, axis=0)
    del wd01

    wd02 = pd.read_parquet(photo_z_waves_wd02_path, columns=photo_z_cols_to_read)
    full_cat = pd.concat([full_cat, wd02], ignore_index=True, axis=0)
    del wd02

    wd03 = pd.read_parquet(photo_z_waves_wd03_path, columns=photo_z_cols_to_read)
    full_cat = pd.concat([full_cat, wd03], ignore_index=True, axis=0)
    del wd03

    mask = (
        (full_cat['mask'] == 0) *
        (full_cat['starmask'] == 0) *
        (full_cat['ghostmask'] == 0) *
        (full_cat['class'] !=  'star') *
        (full_cat['class'] != 'artefact') *
        (full_cat['duplicate'] == 0)
    )
    full_cat = full_cat[mask]
    full_cat = full_cat[photo_z_cols_to_keep]

    full_cat['uberID'] = full_cat['uberID'].astype("int64")
    
    full_cat.rename(columns={'zphot_invar': 'photoZ', 'zphot_err': 'photoZ_err'}, inplace=True)

    #full_cat = full_cat[full_cat['uberID'].isin(subset['uberID'])]
    # inner merge full_cat with subset on uberID to only keep rows in full_cat that are in subset
    if set != 'all_photozs': # for all_photozs we want to keep all rows in full_cat, not just those in subset
        full_cat = full_cat.merge(subset, on='uberID', how='inner')

    return full_cat


def flux2mag(flux):
    return 8.9 - 2.5 * np.log10(flux)


def add_photom(df_to_add_to, photometry_df):
    # find columns in common
    photometry_cols = set(photometry_df.columns)
    df_cols = set(df_to_add_to.columns)
    common_cols = photometry_cols.intersection(df_cols)
    # drop common cols from photometry_df apart from uberID
    common_cols.discard('uberID')

    photometry_df = photometry_df.drop(columns=common_cols)
    #print(photometry_df.columns, df_to_add_to.columns)
    return df_to_add_to.merge(photometry_df, on='uberID', how='left')


def load_waves_photom(region='All',
    wide_n_path = '/Users/sp624AA/Downloads/waves_data/reduced_cats/WAVES_N_Z_2125_reduced.parquet',
    wide_s_path = '/Users/sp624AA/Downloads/waves_data/reduced_cats/WAVES_S_Z_2125_reduced.parquet',
    wd10_path = '/Users/sp624AA/Downloads/waves_data/reduced_cats/WD10_Z_2125_reduced.parquet',
    wd01_path = '/Users/sp624AA/Downloads/waves_data/reduced_cats/WD01_Z_2125_reduced.parquet',
    wd02_path = '/Users/sp624AA/Downloads/waves_data/reduced_cats/WD02_Z_2125_reduced.parquet',
    wd03_path = '/Users/sp624AA/Downloads/waves_data/reduced_cats/WD03_Z_2125_reduced.parquet',
    ):
    
    cols_to_keep = ['uberID', 'RAmax', 'Decmax', 'mag_Zt',  'g-i', 'mag_ut', 'mag_it', 'mag_gt', 'mag_rt', 'mag_Jt', 'mag_Ht', 'mag_Yt', 'mag_Kt']

    if region == 'All':
        df = pd.read_parquet(wide_n_path)

        wide_s = pd.read_parquet(wide_s_path)
        df = pd.concat([df, wide_s], ignore_index=True, axis=0)
        #del wide_s

        wd10 = pd.read_parquet(wd10_path)
        df = pd.concat([df, wd10], ignore_index=True, axis=0)
        #del wd10
    
        wd01 = pd.read_parquet(wd01_path)
        df = pd.concat([df, wd01], ignore_index=True, axis=0)
        #del wd01

        wd02 = pd.read_parquet(wd02_path)
        df = pd.concat([df, wd02], ignore_index=True, axis=0)
        #del wd02

        wd03 = pd.read_parquet(wd03_path)
        df = pd.concat([df, wd03], ignore_index=True, axis=0)
        #del wd03
    elif region == 'wide':
        df = pd.read_parquet(wide_n_path)

        wide_s = pd.read_parquet(wide_s_path)
        df = pd.concat([df, wide_s], ignore_index=True, axis=0)
    
    elif region == 'deep':
        df = pd.read_parquet(wide_s_path)
        deep_mask = (
            (df['RAmax'] > 339) &
            (df['RAmax'] < 351) & 
            (df['Decmax'] > -35) &
            (df['Decmax'] < -30)
        )
        df = df[deep_mask]

    else:
        raise ValueError(f"Unknown region {region}")

    mask = (
        (df['mask'] == 0) * 
        (df['starmask'] == 0) * 
        (df['ghostmask'] == 0) * 
        (df['class'] !=  'star') * 
        (df['class'] != 'artefact') *
        (df['duplicate'] == 0)
    )
    df = df[mask]

    df['mag_ut'] = flux2mag(df['flux_ut'])
    df['mag_it'] = flux2mag(df['flux_it'])
    df['mag_gt'] = flux2mag(df['flux_gt'])
    df['mag_rt'] = flux2mag(df['flux_rt'])

    #df['mag_Zt'] = flux2mag(df['flux_Zt'])
    df['mag_Jt'] = flux2mag(df['flux_Jt'])
    df['mag_Ht'] = flux2mag(df['flux_Ht'])
    df['mag_Yt'] = flux2mag(df['flux_Yt'])
    df['mag_Kt'] = flux2mag(df['flux_Kt'])

    df['uberID'] = df['uberID'].astype("int64")

    df['g-i'] = df['mag_gt'] - df['mag_it']
    return df[cols_to_keep].reset_index(drop=True)