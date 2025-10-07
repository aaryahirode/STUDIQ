# main.py
import sys
from crawler.lms_scraper import scrape_lms
from crawler.material_downloader import download_materials
from extractor.content_formatter import process_all_materials
from nlp_analysis.keypoint_extractor import generate_keypoints
from evaluation.answer_analyzer import evaluate_answer_file

def help_text():
    print("""
Usage:
  python main.py crawl               -> login & save course list (data/raw_lms_data.json)
  python main.py download            -> download materials into data/materials/
  python main.py extract             -> extract text from downloaded materials
  python main.py keypoints           -> generate keypoints (data/keypoints.json)
  python main.py eval <file> <subject> -> evaluate student answer file (image/pdf) for subject
  python main.py all                 -> run crawl -> download -> extract -> keypoints
""")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        help_text()
        sys.exit(0)

    cmd = sys.argv[1].lower()
    if cmd == "crawl":
        scrape_lms()
    elif cmd == "download":
        download_materials()
    elif cmd == "extract":
        process_all_materials()
    elif cmd == "keypoints":
        generate_keypoints()
    elif cmd == "eval":
        if len(sys.argv) < 4:
            print("python main.py eval <answer_file> <subject_name>")
        else:
            answer_file = sys.argv[2]
            subject = sys.argv[3]
            evaluate_answer_file(answer_file, subject)
    elif cmd == "all":
        scrape_lms()
        download_materials()
        process_all_materials()
        generate_keypoints()
    else:
        help_text()
