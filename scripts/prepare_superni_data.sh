# Super-NaturalInstructions dataset
wget -O data/downloads/superni_data.zip https://github.com/allenai/natural-instructions/archive/refs/heads/master.zip
mkdir -p data/downloads/superni
unzip data/downloads/superni_data.zip -d data/downloads/superni
mv data/downloads/superni/natural-instructions-master/ data/eval/superni && rm -r data/downloads/superni data/downloads/superni_data.zip

# Extract classification tasks and edit data
python scripts/extract_classification_tasks.py