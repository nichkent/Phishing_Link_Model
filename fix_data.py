# Import csv functions
import csv

# Open the input and output files
with open('final_dataset.csv', 'r', newline='') as infile, open('final_dataset_final.csv', 'w', newline='') as outfile:
    # Create CSV reader and writer objects
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Process each row in the input file
    for row in reader:
        # Replace empty fields with '0'
        new_row = ['0' if field == '' else field for field in row]
        # Write the updated row to the output file
        writer.writerow(new_row)

# Print if everything ran correctly
print("Empty fields have been replaced with zeros and saved to 'output.csv'.")

