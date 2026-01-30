import numpy as np
import h5py
import hdf5plugin

from tqdm import tqdm

inf = h5py.File('/app/checkpoints/forecast_sequences_2k.h5', 'r')
out = h5py.File('/app/checkpoints/forecast_sequences_2k_blosc.h5', 'w')

try:
    meta = out.create_group('metadata')
except:
    meta = out['metadata']

for key in inf['metadata'].attrs.keys():
    print(key)
    meta.attrs[key] = inf['metadata'].attrs[key]

out.create_group('sequences')
for idx in tqdm(inf['sequences'].keys()):
    grp = out.create_group(f'sequences/{idx}')
    for key in ('params', 'timestamps', 'ssn'):
        grp.create_dataset(key, data=inf[f'sequences/{idx}'][key], compression='gzip', compression_opts=9)
    for key in ('maps',):
        grp.create_dataset(key, data=inf[f'sequences/{idx}'][key], **hdf5plugin.Blosc())
