from typing import List, Tuple, BinaryIO
from PIL import Image
import urllib.request
import urllib.error
import requests
import math
from multiprocessing.pool import ThreadPool
from io import BytesIO
import tempfile

# Latitude, longitude
WorldCoord = Tuple[float, float]

MAX_SIZE_PX = 4_000_000
MAX_WIDTH_PX = math.sqrt(MAX_SIZE_PX)


def validate_coord(coord: WorldCoord) -> bool:
    # TODO
    return True


# Top left, bottom right
WorldRect = Tuple[WorldCoord, WorldCoord]


def validate_rect(rect: WorldRect) -> bool:
    # TODO
    return True


URL = str
# Position of a tile in a tile grid
RowColPos = Tuple[int, int]
# Tile position and URL
TileRecord = Tuple[RowColPos, URL]
# Tile row count, tile col count
MapDimensions = Tuple[int, int]
# Complete tile map info
MapInfo = Tuple[MapDimensions, List[TileRecord]]


def build_rect_url(rect: WorldRect, scale: int):
    return f"https://render.openstreetmap.org/cgi-bin/export?bbox={rect[0][1]},{rect[0][0]},{rect[1][1]},{rect[1][0]}&scale={scale}&format=png"


def download_file(record: TileRecord) -> Tuple[TileRecord, BytesIO]:
    cookies = {"_osm_totp_token": "216681"}
    (_, url) = record
    file = BytesIO()
    with requests.get(url, cookies=cookies) as read:
        try:
            for chunk in read.iter_content(chunk_size=512):
                if chunk:
                    file.write(chunk)
                    file.flush()
            return record, file
        except urllib.error.URLError as e:
            print(record, "failed!!!", e)
            file.close()
            return None


'''
def open_temp_files(rect_records: List[TileRecord]) -> List[Tuple[TileRecord, tempfile._TemporaryFileWrapper]]:
    return [(record, tempfile.TemporaryFile(delete=True)) for record in rect_records]



def close_temp_files(rects: List[Tuple[TileRecord, tempfile._TemporaryFileWrapper]]) -> None:
    for _, io in rects:
        io.close()
'''


def download_multiple_tiles(rect_records: List[TileRecord]) -> List[Tuple[TileRecord, BytesIO]]:
    # record_file_pairs = open_temp_files(rect_records)
    results = ThreadPool(len(rect_records)).imap_unordered(download_file, rect_records)
    return list(results)


def join_images(record_file_pairs: List[Tuple[TileRecord, BytesIO]],
                dimensions: MapDimensions, result_file_name: str):
    record_file_pairs = record_file_pairs
    # To figure out tile pixel dimensions
    (_, file) = record_file_pairs[0]
    with Image.open(file) as img:
        width, height = img.size

    with Image.new('RGB', (dimensions[1] * width, dimensions[0] * height)) as new_im:
        for ((row, col), _), file in record_file_pairs:
            with Image.open(file) as tile_img:
                new_im.paste(tile_img, (col * width, row * height))
        new_im.save(result_file_name)


def generate_tile_records(rect: WorldRect, scale: int) -> MapInfo:
    px_per_degree = math.ceil(397000000 / scale)
    tile_side_dg = (MAX_WIDTH_PX / px_per_degree) * 0.998

    tiles = []
    curr_lat_bottom_dg, curr_lon_left_dg = rect[0]
    while curr_lat_bottom_dg < rect[1][0]:
        # Calculate the actual tile bottom for this row
        pref_lat_top = curr_lat_bottom_dg + tile_side_dg
        mid_latitude = (curr_lat_bottom_dg + pref_lat_top) / 2
        height_to_width_ratio = 1 / math.cos(((mid_latitude / 90) / 2) * math.pi)
        actual_lat_top = curr_lat_bottom_dg + (tile_side_dg / height_to_width_ratio)

        # Create the tiles of this row
        tile_row = []
        while curr_lon_left_dg < rect[1][1]:
            new_tile = ((curr_lat_bottom_dg, curr_lon_left_dg), (actual_lat_top, curr_lon_left_dg + tile_side_dg))
            url = build_rect_url(new_tile, scale)
            tile_row.append(url)
            curr_lon_left_dg += tile_side_dg

        # Save the finished row
        tiles.append(tile_row)

        # Set values for the next row
        curr_lat_bottom_dg = actual_lat_top
        curr_lon_left_dg = rect[0][1]

    # Flatten it
    result = []
    row_count = len(tiles)
    for row_i, row in enumerate(tiles):
        for col_i, tile_url in enumerate(row):
            result.append(((row_count - row_i - 1, col_i), tile_url))

    return (row_count, len(tiles[0])), result


def build_tile_grid(rect: WorldRect, scale: int):
    dimensions, tile_records = generate_tile_records(rect, scale)
    record_file_pairs = download_multiple_tiles(tile_records)
    join_images(record_file_pairs, dimensions, "#a.png")
    # close_temp_files(record_file_pairs)


if __name__ == "__main__":
    build_tile_grid(((49.135403, 16.503313), (49.257874, 16.714003)), 7500)
