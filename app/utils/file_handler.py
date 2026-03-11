import os
import uuid

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_temp_file(file_bytes, extension):

    filename = f"{uuid.uuid4()}.{extension}"

    path = os.path.join(UPLOAD_DIR, filename)

    with open(path, "wb") as f:
        f.write(file_bytes)

    return path