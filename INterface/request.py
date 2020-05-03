import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'sepal_length':2, 'sepal_width':1, 'petal_length':6, 'petal_width': 10})

print(r.json())