import argparse
from processing.caption_generator import ChatGPTImageUploader

def main():
    parser = argparse.ArgumentParser(description="Test ChatGPTImageUploader automation.")
    parser.add_argument("images_folder", nargs="?", default="./images/", help="Path to images folder")
    args = parser.parse_args()

    uploader = ChatGPTImageUploader(args.images_folder)
    uploader.run()


main()
