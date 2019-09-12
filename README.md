# ICV-CBIR-Wavelets
simple image feature detector

to run the program, place images in a folder in the same directory as the scripts


# build the database 
python setup.py --images horses --type w

# run queries
python search.py --images horses --query 191.jpg --type w --limit 10
