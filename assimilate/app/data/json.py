import urllib.request, json

def get_data(url):
    with urllib.request.urlopen(url) as res:
        return json.loads(res.read().decode())

