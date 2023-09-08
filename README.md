# ChatGPT_PDF prototype

## Overview
This project serves as a prototype that enhances GPT's capabilities by utilizing PDF files as a knowledge source.

## Installation

### Dependencies
This project requires certain Python packages to function correctly. These dependencies can be found in the `requirements.txt` file.

To install these dependencies, ensure you are in the project's root directory and run the following command:

`pip install -r requirements.txt`

Furthermore, you need to create an `.env` file in the root which contains your OPENAI API key like this:

OPENAI_API_KEY= ?

Replace the ? with your API key from the OPENAI API website.

## Database Creation

You need to place your PDF files into the `pdf_files` folder.

To generate the databases, execute `by_sentence.py` and `bookmark_or_page_by_page.py` scripts. Each of these scripts contains inline documentation, explaining their individual operation and purpose.

## Chat Application

You can find the chat application in the `chat_with_pdf.py` file. Running this script will initiate the chat interface.

