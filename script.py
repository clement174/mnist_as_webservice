import requests
import json

def classify_image(image_url):
    """
        Take url of an image (digit) and send API a request to get the label (from 0 to 9)
    """

    # prepare headers for http request
    content_type = 'img_path/url'
    headers = {'content-type': content_type}

    # send http request with image and receive response
    api_url = 'http://127.0.0.1:5000/prediction'
    response = requests.post(api_url, data=image_url, headers=headers)

    # decode response
    prediction = json.loads(response.text)

    return prediction



if __name__ == '__main__':


    # get image url
    print("Exemple url = https://conx.readthedocs.io/en/latest/_images/MNIST_6_0.png")
    url = input("Enter image url of a digit = ")
    # get label
    while not url[:4] == "http":
        print('Url not valid ({})'.format(url))
        url = input("Enter image url of a digit = ")

    pred = classify_image(url)
    print()
    print("Image label =", pred[-2])