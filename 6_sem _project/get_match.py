import requests

project_key = 'YOUR_PROJ_KEY'
token = 'YOUR_ACCESS_TOKEN'
key = 'rsaeng_2020_t20_03'
url = "https://api.sports.roanuz.com/v5/cricket/{}/match/{}/".format(project_key,key)
headers = {
    'rs-token': token
}
response = requests.get(url, headers=headers)

print(response.json())