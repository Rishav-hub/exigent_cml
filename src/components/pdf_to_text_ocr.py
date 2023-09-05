from io import StringIO
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from PyPDF2 import PdfReader, PdfWriter
from paddleocr import PaddleOCR
import os
import argparse


class XelpOCR:
    def __init__(self, input_pdf_path, output_pdf_path, text_folder_path) -> None:
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en',
                             ocr_version='PP-OCRv4', use_gpu=True)
        self.input_pdf_path = input_pdf_path
        self.output_pdf_path = output_pdf_path
        self.text_folder_path = text_folder_path

    @staticmethod
    def check_orientation(page):
        if page.mediabox[2] - page.mediabox[0] > page.mediabox[3] - page.mediabox[1]:
            return 'Landscape'
        else:
            return 'Portrait'

    def rotate_landscape_pages(self):
        output_string = StringIO()
        resource_manager = PDFResourceManager()
        device = TextConverter(
            resource_manager, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(resource_manager, device)

        landscape_page = []

        if not os.path.isfile(self.input_pdf_path):
            raise FileNotFoundError(f"Input PDF file not found: {self.input_pdf_path}")

        for page_num, page in enumerate(PDFPage.get_pages(open(self.input_pdf_path, 'rb'))):
            interpreter.process_page(page)
            orientation = self.check_orientation(page)

            if orientation == 'Landscape':
                landscape_page.append(page_num)
        
        if len(landscape_page) > 0:
            print("Fixing orientation as some landscape pages were found in PDF")
        else:
            print("No landscape pages found in PDF, no fix needed.")

        device.close()

        pdf_reader = PdfReader(self.input_pdf_path)
        pdf_writer = PdfWriter()

        for page_num, page in enumerate(pdf_reader.pages):
            if page_num in landscape_page:
                pdf_writer.add_page(page.rotate(90))
            else:
                pdf_writer.add_page(page)

        with open(self.output_pdf_path, 'wb') as output_file:
            pdf_writer.write(output_file)

    @staticmethod
    def find_page_coordinates(data):
        max_x = float('-inf')
        min_x = float('inf')
        max_y = float('-inf')
        min_y = float('inf')

        for bbox, _ in data:
            for x, y in bbox:
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
        return max_x, min_x, max_y, min_y

    @staticmethod
    def find_x_minmax(coordinates):
        xmin = float('inf')
        xmax = float('-inf')

        for x, _ in coordinates:
            if x < xmin:
                xmin = x
            if x > xmax:
                xmax = x
        return xmin, xmax

    @staticmethod
    def find_y_minmax(coordinates):
        ymin = float('inf')
        ymax = float('-inf')

        for _, y in coordinates:
            if y < ymin:
                ymin = y
            if y > ymax:
                ymax = y
        return ymin, ymax

    def get_ocr_results(self):
        return self.ocr.ocr(self.output_pdf_path, cls=True)

    def start_ocr(self):
        self.rotate_landscape_pages()
        os.makedirs(self.text_folder_path, exist_ok=True)
        result = self.get_ocr_results()
        for page_no, page_data in enumerate(result):
            sorted_data = sorted(page_data, key=lambda x: x[0][0][1])
            page_xmin, _, _, _ = self.find_page_coordinates(sorted_data)
            grouped_data = []
            current_group = []
            current_group_y = None

            total_char_count = 0
            total_pixel_width = 0
            page_text = ''
            for bbox, text in sorted_data:
                y_coord = bbox[0][1]
                if current_group_y is None:
                    current_group_y = y_coord

                if abs(y_coord - current_group_y) <= 4:
                    current_group.append((bbox, text))
                else:
                    grouped_data.append(current_group)
                    current_group = [(bbox, text)]
                    current_group_y = y_coord
                curr_xmin, curr_xmax = self.find_x_minmax(bbox)
                char_count = len(text[0])
                pixel_width = abs(curr_xmin - curr_xmax)

                total_pixel_width += pixel_width
                total_char_count += char_count

            if total_pixel_width != 0:
                average_char_per_pixel = total_char_count / total_pixel_width
            else:
                average_char_per_pixel = 0.0

            grouped_data.append(current_group)

            for group in grouped_data:
                line = ''
                prev_bbox = None
                sorted_group = sorted(group, key=lambda x: x[0][0][0])
                for bbox, text in sorted_group:
                    curr_xmin, curr_xmax = self.find_x_minmax(bbox)
                    if prev_bbox is None:
                        space_width = int(
                            abs(curr_xmin - page_xmin) * average_char_per_pixel)
                        line += ' ' * space_width
                    if prev_bbox is not None:
                        space_width = int(
                            abs(curr_xmin - prev_xmax) * average_char_per_pixel)
                        line += ' ' * space_width

                    line += text[0]
                    prev_bbox = bbox
                    prev_xmax = curr_xmax
                page_text += '\n' + line

            page_file_path = os.path.join(
                self.text_folder_path, f"{page_no}.txt")
            with open(page_file_path, 'w') as f:
                f.write(page_text)


if __name__ == '__main__':
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