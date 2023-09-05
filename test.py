import os
import argparse
from src.components.pdf_to_text_ocr import XelpOCR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract text from a PDF file using XelpOCR.')
    parser.add_argument(
        '--input_pdf_path',
        type=str,
        required=True,
        help='Path to the input PDF file.')
    parser.add_argument(
        '--output_pdf_path',
        type=str,
        required=True,
        help='Path to the output PDF file.')
    parser.add_argument(
        '--text_folder_path',
        type=str,
        required=True,
        help='Path to the text folder.')
    args = parser.parse_args()

    input_pdf_path = args.input_pdf_path
    output_pdf_path = args.output_pdf_path
    text_folder_path = args.text_folder_path

    xelp_ocr = XelpOCR(input_pdf_path, output_pdf_path, text_folder_path)
    xelp_ocr.start_ocr()
    