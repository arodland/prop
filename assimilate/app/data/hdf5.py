import urllib.request
import io

import h5py

def get_data(url):
    with urllib.request.urlopen(url) as res:
        content = res.read()
        bio = io.BytesIO(content)
        h5 = h5py.File(bio, 'r')
        return h5

