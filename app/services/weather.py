import requests
# -----------------------------
# WEATHER
# -----------------------------
def get_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        res = requests.get(url, timeout=5)
        return res.json()
    except:
        return {"current_weather": {"temperature": 0, "windspeed": 0}}
