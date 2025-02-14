import re
def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

if __name__ == '__main__':
    translates_txt = 'data/txts/all_translated.txt'
    with open(translates_txt, "r", encoding="utf-8") as txt_file:
        f = txt_file.read()

    with open('data/txts/emoji_free.txt', 'w', encoding="utf-8") as file:
        file.write(remove_emojis(f))

    input_file = "data/txts/emoji_free.txt"   # Change this to your file name
    output_file = "data/txts/cleaned.txt" # Change if needed

    # Read the file and process lines
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Remove leading spaces and "?" signs
    processed_lines = [re.sub(r"^[\s?]+", "", line) for line in lines]

    # Write the processed lines to a new file
    with open(output_file, "w", encoding="utf-8") as file:
        file.writelines(processed_lines)

    print("Processing complete. Check", output_file)
