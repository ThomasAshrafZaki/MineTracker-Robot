# check_classes.py
with open("categories_places365.txt") as f:
    lines = f.readlines()

# طباعة الكل
for line in lines:
    print(line.strip())