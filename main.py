from gg2.mapparser import GG2Map
from stlwriter import STLWriter
from polygon.extractor import ImageToPolygon


if __name__ == '__main__':
    name = "gltex"
    image = GG2Map(name + ".png").mask
    loader = STLWriter()
    extractor = ImageToPolygon(image)
    for poly in extractor.get_polygons():
        loader.write(poly)
    loader.save(name + ".stl")
