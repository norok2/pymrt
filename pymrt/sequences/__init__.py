import os  # Miscellaneous operating system interfaces

# import all files in the directory
__all__ = [
    os.path.splitext(filename)[0]
    for filename in os.listdir(os.path.dirname(__file__))
    if os.path.isfile(filename) and filename.endswith('.py')]
