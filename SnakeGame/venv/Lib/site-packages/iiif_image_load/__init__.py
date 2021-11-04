# ----------------------------------------------------------
# Load a iiif image using the image server's tiling feature.
#
# (C) 2021 Scripta Qumranica Electronica
# Created by: Bronson Brown-deVost
# Released under the MIT license
# ----------------------------------------------------------

import asyncio
import aiohttp
from aiohttp import web
from aiohttp import ClientSession
import cv2
import numpy as np
from numpy import ndarray
import backoff
from more_itertools import grouper
from typing import Tuple, List, Dict, Union, Any, Optional


@backoff.on_exception(backoff.expo, aiohttp.web.HTTPServerError, max_time=60)
async def __fetch_img_data(session: ClientSession, headers, tile_data, image_tiles, proxy: str, username: str, password: str) -> None:
    """Get a cv2 image from a URL and insert it into the full-size image."""

    start_y = tile_data['y']
    end_y = start_y + tile_data['height']
    start_x = tile_data['x']
    end_x = start_x + tile_data['width']
    if proxy and username and password:
        proxy_auth = aiohttp.BasicAuth(username, password)
        async with session.get(tile_data['url'], headers=headers, proxy=proxy, proxy_auth=proxy_auth) as response:
            image_data = np.asarray(bytearray(await response.read()), dtype="uint8")
    else:
        async with session.get(tile_data['url'], headers=headers) as response:
            image_data = np.asarray(bytearray(await response.read()), dtype="uint8")
    
    image_tiles[start_y:end_y, start_x:end_x] = cv2.imdecode(image_data, cv2.IMREAD_COLOR)


@backoff.on_exception(backoff.expo, aiohttp.web.HTTPServerError, max_time=60)
async def __fetch_json(session: ClientSession, headers, url: str, proxy: str, username: str, password: str) -> dict:
    """Get JSON data from a URL."""

    if proxy and username and password:
        proxy_auth = aiohttp.BasicAuth(username, password)
        async with session.get(url, headers=headers, proxy=proxy, proxy_auth=proxy_auth) as response:
            return await response.json()
    else:
        async with session.get(url, headers=headers) as response:
            return await response.json()


async def __calculate_image_tiles(session: ClientSession, url: str, headers: dict, crop_x: int = None, crop_y: int = None,
                                  crop_width: int = None, crop_height: int = None, proxy: str = None, username: str = None, password: str = None) -> Tuple[List[Dict[str, Union[Union[str, int], Any]]], str, Optional[int], Optional[int], Union[int, Any], Union[int, Any]]:
    """Get a 2D list of URLs for each image tile in a IIIF image resource."""

    # Download the info.json data
    info_json_url = url if url[-9:] == 'info.json' else f'{url}/info.json'
    info = await __fetch_json(session, headers, info_json_url, proxy, username, password)

    # Get the image dimensions
    orig_y = crop_y if (crop_y is not None and crop_y >= 0 and crop_y < info['height']) else 0
    orig_x = crop_x if (crop_x is not None and crop_x >= 0 and crop_x < info['width']) else 0
    corr_width = crop_width if crop_width is not None else info['width']
    end_x = orig_x + corr_width if orig_x + corr_width <= info['width'] else info['width']
    corr_height = crop_height if crop_height is not None else info['height']
    end_y = orig_y + corr_height if orig_y + corr_height <= info['height'] else info['height']

    # Find the highest quality file the server supports
    format_info = [x for x in info['profile'] if 'formats' in x]
    file_type = 'default.jpg'
    file_ext = '.' + file_type.split('.')[1]
    if len(format_info) > 0:
        if 'formats' in format_info[0]:
            file_type = 'default.tif' if 'tif' in format_info[0]['formats'] \
                else 'default.png' if 'png' in format_info[0]['formats'] \
                else 'default.jp2' if 'jp2' in format_info[0]['formats'] \
                else 'default.webp' if 'webp' in format_info[0]['formats'] \
                else 'default.jpg'

    # Create a array with the URL and placement info for each tile
    if 'sizes' in info and info['sizes'] and 'width' in info['sizes'][0] and 'height' in info['sizes'][0]:
        tile_width = 0
        tile_height = 0
        # Use the largest size image request to reduce load on the server
        for size in info['sizes']:
          tile_width = size['width'] if size['width'] > tile_width else tile_width
          tile_height = size['height'] if size['height'] > tile_height else tile_height
        y = orig_y
        images = []
        while y < end_y:
            x = orig_x
            adj_height = min(tile_height, end_y - y)
            while x < end_x:
                adj_width = min(tile_width, end_x - x)
                img_url = f'{url}/{int(x)},{int(y)},{int(adj_width)},{int(adj_height)}/full/0/{file_type}'
                images.append(
                    {'url': img_url, 'x': x - orig_x, 'y': y - orig_y, 'width': adj_width, 'height': adj_height})
                x = x + tile_width

            y = y + tile_height
        return images, file_ext, orig_x, orig_y, corr_width, corr_height

    else:  # Return a single tile if the server doesn't support tiling
        return [{'url': f'{url}/{orig_x},{orig_y},{corr_width},{corr_height}/full/0/{file_type}', 'x': 0, 'y': 0, 'width': corr_width, 'height': corr_height}], file_ext, orig_x, orig_y, corr_width, corr_height


async def __download_manager(url: str, headers: dict, x: int = None, y: int = None, width: int = None, height: int = None, proxy: str = None, username: str = None, password: str = None, connector=None) -> ndarray:
    """Asynchronously download the image at the URL in the highest possible resolution."""

    async with aiohttp.ClientSession(connector=connector) as session:
        # Collect the tile data and initialize the necessary lists
        tiles, _, _, _, corr_width, corr_height = await __calculate_image_tiles(session, url, headers, x, y, width, height, proxy, username, password)

        # Fetch the tiles async
        image_shape = (corr_height, corr_width, 3)
        image_tiles = np.empty(image_shape, dtype=np.uint8)
        for tile_group in grouper(tiles, 4):
            tasks = []
            for image_details in [x for x in tile_group if x is not None]:
                tasks.append(__fetch_img_data(session, headers, image_details, image_tiles, proxy, username, password))

            # Wait for all requests to complete successfully
            await asyncio.gather(*tasks)
        return image_tiles


def iiif_image_from_url(url: str, headers: dict={}, x: int = None, y: int = None, width: int = None, height: int = None, proxy: str = None, username: str = None, password: str = None, connector=None) -> ndarray:
    """Retrieve the image from a IIIF image resource (do not add any size info like "/full/full/0/default.jpg").

    Example usage:
        ::

        image = iiif_image_from_url('https://iaa.iiifhosting.com/iiif/a00976765308e3d90d13779925e0f7664c1d63a7715ea7097d10b835f4bc090c')

    """

    # Start the event loop for async image retrieval
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(__download_manager(url, headers, x, y, width, height, proxy, username, password, connector))
