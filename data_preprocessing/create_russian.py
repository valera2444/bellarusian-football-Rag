import os
import json
from bs4 import BeautifulSoup

def extract_text_from_html(html_content):
    """
    Extracts and returns only the text content from an HTML document.
    :param html_content: A string containing the HTML content
    :return: A string with extracted text
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    
    # Extract text from <div class="text">
    text_divs = [div.get_text(separator=' ', strip=True) for div in soup.find_all("div", class_="text")]
    
    return text_divs

def process_html_files(folder_path, output_json_path):
    """
    Process all HTML files in a folder and save extracted text into a JSON file.
    :param folder_path: Path to the folder containing HTML files
    :param output_json_path: Path to save the output JSON file
    """
    extracted_data = {}
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".html"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                html_content = file.read()
                extracted_data[filename] = extract_text_from_html(html_content)
    
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(extracted_data, json_file, ensure_ascii=False, indent=4)


def split_json_into_txt(json_path, txt_folder, num_parts=7):
    """
    Split a JSON file into multiple TXT files with equal sizes.
    :param json_path: Path to the JSON file
    :param txt_folder: Folder to save the TXT files
    :param num_parts: Number of TXT files to create
    """
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)
    
    with open(json_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        all_text = []
        
        for key, value in data.items():
            all_text.append(f"{key}:\n")
            if isinstance(value, list):
                all_text.extend(value)
            else:
                all_text.append(str(value))
            all_text.append("\n\n")
        
    with open(os.path.join(txt_folder, f"all.txt"), "w", encoding="utf-8") as txt_file:
            txt_file.write("\n".join(all_text))
            
    total_size = len(all_text)
    chunk_size = total_size // num_parts

    if num_parts == 1:
        txt_path = os.path.join(txt_folder, f"all_russian.txt")
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write("\n".join(all_text))
        return

    for i in range(num_parts):
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size if i < num_parts - 1 else total_size

        txt_path = os.path.join(txt_folder, f"part_{i+1}.txt")
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write("\n".join(all_text[start_index:end_index]))


if __name__ == '__main__':

    # Example usage
    folder_path = "data/row_data" # Change this to the folder containing HTML files
    output_json_path = "data/extracted_text.json" # Change this to the desired output JSON file
    process_html_files(folder_path, output_json_path)

    # Example usage
    json_path = "data/extracted_text.json"  # Change to actual JSON file path
    txt_folder = "data/txts"  # Change to output folder path
    split_json_into_txt(json_path, txt_folder,num_parts=1)
