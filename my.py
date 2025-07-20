# -*- coding: utf-8 -*-
"""
Generates class_labels.txt based on filenames in the 'pollen' folder
@author: Dhanesh
"""

import os
import re
from sklearn.preprocessing import LabelEncoder

# Path to your image folder
image_folder = "pollen"

# List all filenames
file_list = os.listdir(image_folder)

# Extract labels from filenames
labels = []
for filename in file_list:
    if filename.endswith(".jpg"):
        # Handles: flower_01.jpg or flower (1).jpg
        match = re.match(r"([a-zA-Z]+)", filename)
        if match:
            labels.append(match.group(1).lower())

# Encode and save unique labels
le = LabelEncoder()
le.fit(labels)

# Write to class_labels.txt
with open("class_labels.txt", "w") as f:
    for label in le.classes_:
        f.write(label + "\n")

print("✅ class_labels.txt created successfully.")
print("📂 Classes found:", list(le.classes_))
