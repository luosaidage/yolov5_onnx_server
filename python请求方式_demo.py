import requests

def pick_img(path='ReqFile/bus.jpg'):
    headers = {
        'accept': 'application/json',
        # requests won't add a boundary if this header is set when you pass files=
        # 'Content-Type': 'multipart/form-data',
    }

    files = {
        'file': open(path, 'rb'),
    }

    response = requests.post('http://127.0.0.1:8000/detect/', headers=headers, files=files)

    print (response.text)
    return response.text

pick_img()