import exifread
"""
based on https://gist.github.com/snakeye/fdc372dbf11370fe29eb
"""

def get_key_value(data, key):
    if key in data:
        return data[key]
    return None

def convert_to_degrees(value):
    d = float(value.values[0].num) / float(value.values[0].den)
    m = float(value.values[1].num) / float(value.values[1].den)
    s = float(value.values[2].num) / float(value.values[2].den)
    return d + (m / 60.0) + (s / 3600.0)

path_name = 'IMG_0806.jpg'
# Open image file for reading (binary mode)
f = open(path_name, 'rb')

# Return Exif tags
tags = exifread.process_file(f)

gps_latitude = get_key_value(tags, 'GPS GPSLatitude')
gps_latitude_ref = get_key_value(tags, 'GPS GPSLatitudeRef')
gps_longitude = get_key_value(tags, 'GPS GPSLongitude')
gps_longitude_ref = get_key_value(tags, 'GPS GPSLongitudeRef')

if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
    lat = convert_to_degrees(gps_latitude)
    if gps_latitude_ref.values[0] != 'N':
        lat = 0 - lat
    lon = convert_to_degrees(gps_longitude)
    if gps_longitude_ref.values[0] != 'E':
        lon = 0 - lon
    print('Latitude: {}'.format(lat))
    print('Longitude: {}'.format(lon))
