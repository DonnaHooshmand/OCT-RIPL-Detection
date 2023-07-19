import requests
from bs4 import BeautifulSoup
import urllib.parse
import os
import time


def import_data(url, download_directory, batch_size, file_extension):
    response = requests.get(url)
    html_content = response.content

    soup = BeautifulSoup(html_content, "html.parser")
    print('soup')

    # Assuming the download links are in <a> tags with href attribute
    

    # Create a directory to store the downloaded files
    os.makedirs(download_directory, exist_ok=True)
    print('starting download')
    # Find and download the files
    count = 0
    for link in soup.find_all("a"):
        href = link.get("href")
        if href.endswith(file_extension):
            # Construct the absolute URL if the link is relative
            absolute_url = urllib.parse.urljoin(url, href)

            # Download the file
            response = requests.get(absolute_url)
            filename = os.path.join(download_directory, href.split("/")[-1])
            with open(filename, "wb") as file:
                file.write(response.content)
            print('downloaded: ', filename)
            count += 1
            if count % batch_size == 0:
                print(f"Downloaded {count} files. Pausing for a moment...")
                # Add a delay between batches to avoid timeouts or server overload
                time.sleep(5)  # Adjust the delay as needed

image_file_extension = ".tiff"  # Change to the desired file extension
label_file_extension = ".tif"  # Change to the desired file extension

train_url = r'https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/index.html'
train_labels_url = r'https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/index.html'
train_directory = "data\roads_dataset\test_images"
train_labels_directory = "data\roads_dataset\train_labels"
import_data(train_url, train_directory, 10, image_file_extension)
import_data(train_labels_url, train_labels_directory, 10, label_file_extension)
print('finished importing train labels')

valid_url = r'https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/sat/index.html'
valid_labels_url = r'https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/map/index.html'
valid_directory = "data\roads_dataset\valid_images"
valid_labels_directory = "data\roads_dataset\valid_labels"
import_data(valid_url, valid_directory, 10, image_file_extension)
print('finished importing validation images')
import_data(valid_labels_url, valid_labels_directory, 7, label_file_extension)
print('finished importing validation labels')

test_url = r'https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/sat/index.html'
test_labels_url = r'https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/map/index.html'
test_directory = "data\roads_dataset\test_images"
test_labels_directory = "data\roads_dataset\test_labels"
import_data(test_url, test_directory, 10, image_file_extension)
print('finished importing test images')
import_data(test_labels_url, test_labels_directory, 10, label_file_extension)
print('finished importing test labels')



