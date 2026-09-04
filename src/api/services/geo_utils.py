import os
import math
import pandas as pd


_pincode_data = None


def _load_pincodes():
    global _pincode_data
    if _pincode_data is None:
        csv_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..',
            'data', 'reference', 'indian_pincode_lat_lng.csv'
        )
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, dtype={'pincode': str})
            _pincode_data = df.set_index('pincode')[['latitude', 'longitude']].to_dict('index')
        else:
            print("WARNING: pincode csv not found, geo distance will default to 0")
            _pincode_data = {}
    return _pincode_data


def haversine(lat1, lon1, lat2, lon2):
    # standard haversine formula for earth distance
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def calculate_geo_distance(billing_pin: str, shipping_pin: str) -> float:
    if billing_pin == shipping_pin:
        return 0.0

    data = _load_pincodes()
    b = data.get(billing_pin)
    s = data.get(shipping_pin)

    if b is None or s is None:
        return 0.0

    dist = haversine(b['latitude'], b['longitude'], s['latitude'], s['longitude'])
    return round(dist, 2)
