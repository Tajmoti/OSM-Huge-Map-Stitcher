import math
from multiprocessing.pool import ThreadPool
from PIL import Image
import requests
import tempfile
from typing import IO, Iterator, List, Tuple, Union
from io import BytesIO

MAX_SIZE_PX = 4_000_000
"""Maximum pixel count of a map tile. Hardcoded limit in OSM exporting endpoint."""
MAX_WIDTH_PX = math.sqrt(MAX_SIZE_PX)
"""Maximum side of a square tile in pixels"""

# Latitude, longitude
WorldCoord = Tuple[float, float]
"""A single real-world geographic coordinate"""


def check_valid_coord(coord: WorldCoord):
    """:raises ValueError if latitude or longitude is out of the allowed range"""
    lat, lon = coord
    if lat < -90.0 or lat > 90.0:
        raise ValueError("Latitude must be between -90 and 90")
    if lon < -180.0 or lon > 180.0:
        raise ValueError("Longitude must be between -180 and 180")


# Top left, bottom right
WorldRect = Tuple[WorldCoord, WorldCoord]


def check_and_normalize_rect(rect: WorldRect) -> WorldRect:
    """
    Reorders and returns the passed in coordinates
    in the valid order (top-left, bottom-right).

    :raises ValueError if latitude or longitude is out of the allowed range
    """
    c1, c2 = rect
    check_valid_coord(c1)
    check_valid_coord(c2)
    top_left = min(c1[0], c2[0]), min(c1[1], c2[1])
    bottom_right = max(c1[0], c2[0]), max(c1[1], c2[1])
    return top_left, bottom_right


URL = str
"""Just a good-ol` URL"""

RowColPos = Tuple[int, int]
"""The (row, col) position of a single map tile in the final image"""

TileRecord = Tuple[RowColPos, WorldRect]
"""Position of a tile in the resulting image and its bounding box in degrees"""

ImgDimensions = Tuple[int, int]
"""Width and height in pixels"""

MapDimensions = Tuple[RowColPos, ImgDimensions]
"""Count of rows and columns of map tiles in the map, map dimensions in pixels"""

MapInfo = Tuple[MapDimensions, List[TileRecord]]
"""
Complete information about the map comprised of its dimensions,
scale and a list of tiles that the map is composed of.

The tiles are all of the same size and the image must be built from
top to bottom and from left to right!!!
"""


def build_osm_rect_url(rect: WorldRect, scale: int) -> URL:
    """Generates a URL for a single map tile to be downloaded from the OSM export endpoint."""
    return f"https://render.openstreetmap.org/cgi-bin/export?bbox={rect[0][1]},{rect[0][0]},{rect[1][1]},{rect[1][0]}&scale={scale}&format=png"


OsmTileDownloadArgs = Tuple[WorldRect, int, str, IO]
"""
The info about a tile to be downloaded with the scale, request token
and an IO object where the PNG data should be written into.
The token is included with each OSM endpoint request as '_osm_totp_token'.
"""


def download_osm_tile(args: OsmTileDownloadArgs) -> Union[bool, Exception]:
    """
    Downloads a single OSM map tile into the provided output stream.
    :returns True on success, URLError on failure.
    """
    box, scale, token, io_out = args
    # Cookie is required!
    cookies = {"_osm_totp_token": token}
    url = build_osm_rect_url(box, scale)
    with requests.get(url, cookies=cookies) as response:
        if response.status_code == 400:
            raise ValueError("The token is invalid!")
        elif response.status_code == 500:
            raise ValueError("The scale is out of range!")
        elif not response:
            raise Exception(response.content)
        try:
            for chunk in response.iter_content(chunk_size=512):
                if chunk:
                    io_out.write(chunk)
            io_out.flush()
            return True
        except Exception as e:
            return e


def download_multiple_osm_tiles(rect_records: Iterator[Tuple[TileRecord, IO]], scale: int, token: str) -> None:
    """
    Downloads all of the tiles described by rect_records into their provided IO streams.

    :raises URLError of first failed request (if any).
    """
    mapped_args = list(map(lambda pair: (pair[0], scale, token, pair[1]), rect_records))
    results = ThreadPool(len(mapped_args)).imap(download_osm_tile, mapped_args)
    for result in results:
        if isinstance(result, Exception):
            raise result


def flatten_tile_rows(tiles: List[List[WorldRect]]) -> Tuple[RowColPos, List[TileRecord]]:
    result = []
    row_count = len(tiles)
    for row_i, row in enumerate(tiles):
        for col_i, tile_url in enumerate(row):
            result.append(((row_count - row_i - 1, col_i), tile_url))
    col_count = len(tiles[0])
    return (row_count, col_count), result


def calculate_height_to_width_ratio(latitude_dg: float) -> float:
    return 1 / math.cos(((latitude_dg / 90) / 2) * math.pi)


def calculate_width_coefficients_at_scale(scale: int) -> Tuple[float, float]:
    px_per_degree_width = math.ceil(397000000 / scale)
    tile_width_dg = (MAX_WIDTH_PX / px_per_degree_width) * 0.998
    return px_per_degree_width, tile_width_dg


def calculate_image_dimensions_px(rect: WorldRect, px_per_degree_width: float) -> ImgDimensions:
    # Width
    map_width_dg: float = rect[1][1] - rect[0][1]
    map_width_px: int = math.floor(px_per_degree_width * map_width_dg)
    # Height
    # TODO Calculate height
    # lat_mid_dg: float = (rect[0][0] + rect[1][0]) / 2
    # ratio_in_middle: float = calculate_height_to_width_ratio(lat_mid_dg)
    # map_height_dg: float = rect[1][0] - rect[0][0]
    # map_height_px = (map_width_px / ratio_in_middle)
    return map_width_px, map_width_px


def cut_up_into_tiles(rect: WorldRect, tile_width_dg: float):
    """
    Cuts up the passed in rect into tiles of width tile_width_dg
    and height calculated so that the tiles remain squares.

    The topmost and rightmost tiles are larger than they need to
    be and contain some space out of bounds of rect just so they
    stay squares. This can be (is) cropped when composing
    the final image at bitmap level.
    """
    tiles = []
    curr_lat_bottom_dg, curr_lon_left_dg = rect[0]
    while curr_lat_bottom_dg < rect[1][0]:
        # Calculate the actual tile bottom for this row
        pref_lat_top = curr_lat_bottom_dg + tile_width_dg
        mid_latitude = (curr_lat_bottom_dg + pref_lat_top) / 2
        height_to_width_ratio = calculate_height_to_width_ratio(mid_latitude)
        actual_lat_top = curr_lat_bottom_dg + (tile_width_dg / height_to_width_ratio)

        # Create the tiles of this row
        tile_row = []
        while curr_lon_left_dg < rect[1][1]:
            top_left = (curr_lat_bottom_dg, curr_lon_left_dg)
            bottom_right = (actual_lat_top, curr_lon_left_dg + tile_width_dg)
            tile_row.append((top_left, bottom_right))
            curr_lon_left_dg += tile_width_dg

        # Save the finished row
        tiles.append(tile_row)

        # Set values for the next row
        curr_lat_bottom_dg = actual_lat_top
        curr_lon_left_dg = rect[0][1]
    return tiles


def generate_tile_records_in_scale(rect: WorldRect, scale: int) -> MapInfo:
    """
    Cuts up the bounding box into multiple squares with the OSM image export limit
    in mind and calculates the resulting map dimensions.

    :returns MapInfo composed of the resulting image tile count (row, col count),
             dimensions of the image in pixels and a list of the tile bounding boxes.
    """
    px_per_degree_width, tile_width_dg = calculate_width_coefficients_at_scale(scale)
    tiles = cut_up_into_tiles(rect, tile_width_dg)
    result_img_sides_px = calculate_image_dimensions_px(rect, px_per_degree_width)
    # Tile (row count, col count), flattened tiles
    tile_counts, result = flatten_tile_rows(tiles)
    # Combined map dimensions
    dimensions: MapDimensions = (tile_counts, result_img_sides_px)
    # Combined map info
    map_info: MapInfo = (dimensions, result)
    return map_info


def join_images(record_file_pairs: Iterator[Tuple[RowColPos, IO]],
                dimensions: MapDimensions, result_file_name: str):
    (row_cnt, col_cnt), (img_w, img_h) = dimensions

    with Image.new('RGB', (img_w, img_h)) as new_im:
        for (row, col), io_in in record_file_pairs:
            with Image.open(io_in) as tile_img:
                width, height = tile_img.size
                # Y needs to be calculated from the bottom instead of from the top
                y_offset = img_h - (row_cnt - row) * height
                new_im.paste(tile_img, (col * width, y_offset))
        new_im.save(result_file_name)


def generate_map_in_scale(rect: WorldRect, scale: int, token: str,
                          file_name: str, in_memory: bool = False):
    # Check the params!
    rect_normalized = check_and_normalize_rect(rect)
    # Cut the map into squares!
    dimensions, tile_records = generate_tile_records_in_scale(rect_normalized, scale)

    # Create IO buffer for each tile!
    ios = list(map(lambda _: BytesIO() if in_memory else tempfile.TemporaryFile(), tile_records))
    bounding_boxes = map(lambda tile: tile[1], tile_records)
    # Download the tiles!
    download_multiple_osm_tiles(zip(bounding_boxes, ios), scale, token)
    # Join them!
    img_positions = map(lambda tile: tile[0], tile_records)
    join_images(zip(img_positions, ios), dimensions, file_name)
    # Close the streams!
    for io in ios:
        io.close()


if __name__ == "__main__":
    bb: WorldRect = ((49.121482, 16.492585), (49.283178, 16.740465))
    generate_map_in_scale(bb, 8500, "433343", "brno.png")
