import urllib.request

print("Downloading weights... (قد ياخد دقيقة)")
urllib.request.urlretrieve(
    "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar",
    "resnet18_places365.pth"
)
print("Done weights!")

print("Downloading class names...")
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt",
    "categories_places365.txt"
)
print("Done! الملفين جاهزين")