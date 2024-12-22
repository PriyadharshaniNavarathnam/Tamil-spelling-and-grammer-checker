def reformat_tamil_words(input_file, output_file):
    """
    Reformat a file with comma-separated words and frequencies to space-separated format.

    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to the output file.
    """
    try:
        # Open the input file and output file
        with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
            # Process each line in the input file
            for line in infile:
                # Split the line by comma
                word, frequency = line.strip().split(",")
                # Write the reformatted line to the output file
                outfile.write(f"{word} {frequency}\n")
        print(f"File reformatted successfully. Saved as: {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")


input_file_path = "tamilWords.txt"  
output_file_path = "tamilWords_formatted.txt"  

# Call the function
reformat_tamil_words(input_file_path, output_file_path)
