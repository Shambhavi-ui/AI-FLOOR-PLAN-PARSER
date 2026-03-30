import requests

try:
    with open('floorplan.png', 'rb') as f:
        res = requests.post('http://127.0.0.1:5000/process', files={'image': f})
    
    print(res.status_code)
    if res.status_code == 200:
        data = res.json()
        print('Walls:', len(data.get('walls', [])))
        print('Rooms:', len(data.get('rooms', [])))
    else:
        print('Error:', res.text)
except Exception as e:
    print('Exception:', e)
