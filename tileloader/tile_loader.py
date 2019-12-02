from typing import List, Tuple, BinaryIO
from PIL import Image
import urllib.request
import urllib.error
import requests
import math
from multiprocessing.pool import ThreadPool
from io import BytesIO

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
# Width, height
WH = Tuple[int, int]
# Tile row count, tile col count
MapDimensions = Tuple[RowColPos, WH]
# Complete tile map info
MapInfo = Tuple[MapDimensions, List[TileRecord]]


def build_rect_url(rect: WorldRect, scale: int) -> str:
    return f"https://render.openstreetmap.org/cgi-bin/export?bbox={rect[0][1]},{rect[0][0]},{rect[1][1]},{rect[1][0]}&scale={scale}&format=png"


def download_file(a: Tuple[TileRecord, str]) -> Tuple[TileRecord, BytesIO]:
    record, token = a
    cookies = {"_osm_totp_token": token}
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


def download_multiple_tiles(rect_records: List[TileRecord], token: str) -> List[Tuple[TileRecord, BytesIO]]:
    # record_file_pairs = open_temp_files(rect_records)
    records_with_token = map(lambda rec: (rec, token), rect_records)
    results = ThreadPool(len(rect_records)).imap_unordered(download_file, records_with_token)
    return list(results)


def join_images(record_file_pairs: List[Tuple[TileRecord, BytesIO]],
                dimensions: MapDimensions, result_file_name: str):
    # To figure out tile pixel dimensions
    (_, file) = record_file_pairs[0]
    with Image.open(file) as img:
        width, height = img.size

    img_w, img_h = dimensions[1]
    row_cnt, col_cnt = dimensions[0]

    with Image.new('RGB', (img_w, img_h)) as new_im:
        for ((row, col), _), file in record_file_pairs:
            with Image.open(file) as tile_img:
                # Y needs to be calculated from the bottom instead of from the top
                y_offset = img_h - (row_cnt - row) * height
                new_im.paste(tile_img, (col * width, y_offset))
        new_im.save(result_file_name)


def flatten_tile_rows(tiles: List[List[URL]]) -> Tuple[int, List[TileRecord]]:
    result = []
    row_count = len(tiles)
    for row_i, row in enumerate(tiles):
        for col_i, tile_url in enumerate(row):
            result.append(((row_count - row_i - 1, col_i), tile_url))
    return row_count, result


def generate_tile_records(rect: WorldRect, scale: int, square: bool = False) -> MapInfo:
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

    # Flattened tiles
    row_count, result = flatten_tile_rows(tiles)
    # Resulting image dimensions in px
    map_width_dg: float = rect[1][1] - rect[0][1]
    map_width_px: int = math.floor(px_per_degree * map_width_dg)
    map_height_px = map_width_px
    # TODO calculate height in px
    if square:
        map_width_px = map_height_px = min(map_width_px, map_width_px)
    result_img_sides_px: WH = (map_width_px, map_height_px)
    # Tile row count, tile col count
    tile_counts: RowColPos = (row_count, len(tiles[0]))
    # Combined map dimensions
    dimensions: MapDimensions = (tile_counts, result_img_sides_px)
    # Combined map info
    map_info: MapInfo = (dimensions, result)
    return map_info


def build_tile_grid(rect: WorldRect, scale: int, token: str, square: bool = False):
    dimensions, tile_records = generate_tile_records(rect, scale, square)
    record_file_pairs = download_multiple_tiles(tile_records, token)
    join_images(record_file_pairs, dimensions, "#a.png")
    # close_temp_files(record_file_pairs)


if __name__ == "__main__":
    build_tile_grid(((49.121482, 16.492585), (49.283178, 16.740465)), 10000, "909798", True)
