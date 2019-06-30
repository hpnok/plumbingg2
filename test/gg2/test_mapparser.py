import numpy as np

from gg2.mapparser import GG2Map


def test_decompress():
    original_wm = [255, 255, 0, 0, 0, 0, 0, 255, 255, 0, 255, 255, 0, 0, 255, 0]
    metadata_wm = chr(10) + str(len(original_wm)) + chr(10) + '1' + chr(10)
    fill = 0
    numv = 0
    for i in original_wm:
        numv <<= 1
        if i == 0:
            numv += 1
        fill += 1
        if fill == 6:
            metadata_wm += chr(numv + 32)
            numv = 0
            fill = 0
    if fill > 0:
        for fill in range(fill, 6, 1):
            numv <<= 1
        metadata_wm += chr(numv + 32)

    wm = GG2Map._decompress_wm(metadata_wm)

    assert np.all(wm == original_wm)
