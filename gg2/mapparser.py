import numpy as np
import re
from PIL import Image
from PIL.PngImagePlugin import PngImageFile

GG2_METADATA_TAG = 'Gang Garrison 2 Level Data'
WM_TAG = '{WALKMASK}'
WM_END_TAG = '\n{END WALKMASK}'


class GG2Map(object):
    def __init__(self, file_name):
        with Image.open(file_name) as map_file:
            self.width, self.height = map_file.size
            self._mask = self.extract_wm(map_file)
            self._image = np.array(map_file)

        assert (self.height, self.width) == self._mask.shape
        self._mask[self.height - 1, 0] = 0

    @property
    def mask(self):
        return self._mask

    @property
    def image(self):
        return self._image

    @classmethod
    def extract_wm(cls, map_file: PngImageFile) -> np.ndarray:
        metadata = map_file.text[GG2_METADATA_TAG]
        index_wm_start = metadata.find(WM_TAG) + len(WM_TAG)
        index_wm_end = metadata[index_wm_start:].find(WM_END_TAG)

        return cls._decompress_wm(metadata[index_wm_start: index_wm_end])

    @classmethod
    def _decompress_wm(cls, metadata_wm: str) -> np.ndarray:
        """
        :param metadata_wm: content between the {WALKMASK} tag end the \n{WALKMASK}
        :return: a 2d array representing the wallmask
        """
        match = re.search(r'\n(?P<width>\d*)\n(?P<height>\d*)\n(?P<wm>.*)', metadata_wm)
        width = int(match.group('width'))
        height = int(match.group('height'))
        wm = np.empty(width*height, dtype=np.uint8)
        compressed_wm = match.group('wm')
        for chunk, c in zip(cls._get_wm_chunk_iterator(wm), compressed_wm):
            bit_code = ord(c) - 32
            for i in range(len(chunk)):
                if bit_code&(1 << (5 - i)):
                    chunk[i] = 0
                else:
                    chunk[i] = 255
        wm.shape = (height, width)
        return wm

    @staticmethod
    def _get_wm_chunk_iterator(wm):
        for i in range(0, len(wm), 6):
            yield wm[i:i + 6]
