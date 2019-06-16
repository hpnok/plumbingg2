from stlwriter import STLWriter
from polygon.extractor import get_gg2_image, ImageToPolygon


if __name__ == '__main__':
    name = "test"
    image = get_gg2_image(name + ".png")
    loader = STLWriter()
    extractor = ImageToPolygon(image)
    for poly in extractor.get_polygons():
        loader.write(poly)
    loader.save(name + ".stl")