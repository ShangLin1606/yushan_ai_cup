
import os
from models.LangChainModel import LangChainModel
from models.CorpusManager import CorpusManager

import pytesseract
import pymupdf
from PIL import Image
import io
import re

import pdfplumber  # 用於從PDF文件中提取文字的工具

class PDFTextImageExtractor:
    def __init__(self, category:str, source_path:str, min_width:int=500, min_height:int=500, file_counter_for_save = 100):
        self.category = category
        
        self.category_source_path = os.path.join(source_path, category)
        if not os.path.exists(self.category_source_path) or not os.listdir(self.category_source_path):
            raise FileNotFoundError(f"Reference data is missing, the path is incorrect, or the directory is empty: {self.category_source_path}")

        self.min_width = min_width
        self.min_height = min_height
        self.file_counter_for_save = file_counter_for_save
        self.llm_model = LangChainModel()
        self.corpusmanager = CorpusManager(self.category)

    def create_and_save_pdf_content(self) -> dict:
        """
        Extracts content from PDF files in the specified source path and saves the processed data.

        This method iterates over all PDF files in the source directory, extracts text and images from each page, 
        and performs OCR on images meeting the size criteria. It formats and concatenates the extracted text and 
        OCR results, stores the combined output into a corpus dictionary, and saves the data using the CorpusManager.

        Returns:
            dict: A dictionary where each key is the filename (without extension) and the value is the combined 
                text and OCR-processed content from the PDF.
        
        Raises:
            Exception: If any issues occur while processing images or extracting content from PDFs.
        """
        formatted_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'formatted_data')
        os.makedirs(formatted_data_path, exist_ok=True)

        if not os.path.exists(f"{formatted_data_path}/{self.category}_all_text.json"):
            file_ls = self._get_pdf_file_names()
            corpus_dict = {}
            file_counter = 0
            for file in file_ls:
                file_name = os.path.join(self.category_source_path, file)
                page_texts = self._extract_text_from_pdf(file_name)

                with pymupdf.open(file_name) as pdf:
                    for page_number in range(pdf.page_count):
                        page_texts[page_number] = self._remove_page_whitespace(page_texts[page_number])
                        gpt_img_text = self._extract_images_and_ocr(pdf, page_number)
                        page_texts[page_number] += gpt_img_text
                
                corpus_dict[file.replace('.pdf', '')] = "\n---\n".join(page_texts)
            
                file_counter += 1
                if file_counter % self.file_counter_for_save == 0:
                    self.corpusmanager.save_corpus(corpus_dict)
            
            self.corpusmanager.save_corpus(corpus_dict)

            # print(f'The {self.category} PDF data has been extracted and returned as a dictionary, and saved to data/formatted_data/.') 
            
            return  corpus_dict
        
        else:
            # print(f'The {self.category} PDF context dictionary already exists and has been returned from data/formatted_data/ as a dictionary.')
            corpus_dict = self.corpusmanager.load_corpus()
            
            return  corpus_dict

    
    def _get_pdf_file_names(self) -> list[str]:
        return [f for f in os.listdir(self.category_source_path) if f.endswith('.pdf')]

    def _extract_text_from_pdf(self, file_name:str) -> list[str]:
        page_texts = []
        with pdfplumber.open(file_name) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                page_texts.append(text or "")
        return page_texts
    
    def _remove_page_whitespace(self, page_text) -> str:
        page_text = re.sub(r'[\n\r\t]+', ', ', page_text)
        page_text = re.sub(r'\s{2,}', ' ', page_text)
        return page_text
    
    def _extract_images_and_ocr(self, pdf:pymupdf.Document, page_number:int) -> str:
        """
        Extracts images from a specific page of a PDF and performs OCR on images that meet the size criteria.

        This method loads the specified page, identifies all embedded images, and extracts each image for OCR 
        processing if it meets the minimum width and height requirements. The extracted text from the OCR 
        process is formatted and appended to the result.

        Args:
            pdf (pymupdf.Document): The PDF document object being processed.
            page_number (int): The page number from which to extract images.

        Returns:
            str: A string containing the formatted OCR text extracted from all processed images on the page.
        
        Raises:
            Exception: Catches and logs errors encountered during image processing and OCR extraction.
        """
        page = pdf.load_page(page_number)
        img_list = page.get_images(full=True)
        gpt_img_text = ''

        for img in img_list:
            xref = img[0]
            pix = pymupdf.Pixmap(pdf, xref)
            if pix.width > self.min_width and pix.height > self.min_height: 
                try:
                    if pix.n > 3: 
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    img_ocr_text = pytesseract.image_to_string(img, lang="eng+chi_tra")

                    if len(img_ocr_text) >= 30:
                        formatted_text = self._ocr_to_formatted_text(img_ocr_text)
                        gpt_img_text += formatted_text
                except Exception as e:
                    print(f"Error processing image on page {page_number+1}: {e}")
                    continue
        return gpt_img_text
    
    def _ocr_to_formatted_text(self, img_ocr_text:str) -> str:
        """
        Formats the OCR-extracted text using a language model for improved readability and structure.

        This method takes raw OCR text from an image, processes it through a language model to enhance 
        clarity, and returns the formatted result.

        Args:
            img_ocr_text (str): The raw text extracted from an image via OCR.

        Returns:
            str: A string containing the formatted and structured text output from the language model.
        """
        return self.llm_model.get_formatted_text_from_ocr(img_ocr_text)
    


    
if __name__=='__main__':
    category = 'finance'
    source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'reference', category)
    pdf_loader = PDFTextImageExtractor(category, source_path)
    pdf_loader.create_and_save_pdf_content()

    
    
   